import dataloaders_hsi_test
import torch
import numpy as np
from tqdm import tqdm
import argparse
import os
from skimage.restoration import  denoise_nl_means,estimate_sigma
import scipy.io as scio
import time
from model_loader import init_model,load_model
from ops.utils_blocks import block_module
from ops.utils import show_mem, generate_key, save_checkpoint, str2bool, step_lr, get_lr,MSIQA
parser = argparse.ArgumentParser()
#model
parser.add_argument("--mode", type=str, default='sc',help='[group, sc]')
parser.add_argument("--stride", type=int, dest="stride", help="stride size", default=1)
parser.add_argument("--num_filters", type=int, dest="num_filters", help="Number of filters", default=[9,9,9])
parser.add_argument("--kernel_size", type=int, dest="kernel_size", help="The size of the kernel", default=5)
parser.add_argument("--noise_level", type=int, dest="noise_level", help="Should be an int in the range [0,255]", default=25)
parser.add_argument("--unfoldings", type=int, dest="unfoldings", help="Number of LISTA step unfolded", default=24)
parser.add_argument("--patch_size", type=int, dest="patch_size", help="Size of image blocks to process", default=96)
parser.add_argument("--rescaling_init_val", type=float, default=1.0)
parser.add_argument("--lmbda_prox", type=float, default=0.02, help='intial threshold value of lista')
parser.add_argument("--multi_theta", type=str2bool, default=1, help='wether to use a sequence of lambda [1] or a single vector during lista [0]')
parser.add_argument("--gpus", '--list',action='append', type=int, help='GPU')



#data
parser.add_argument("--model_name", type=str, dest="model_name", help="The name of the model to be saved.", default=None)
parser.add_argument("--test_path", type=str, help="Path to the dir containing the testing datasets.", default="/Users/bearshng/Documents/workspaces/MATLAB/denoising/TDL/data/")
parser.add_argument("--tqdm", type=str2bool, default=False)
parser.add_argument("--gt_path", type=str, help="Path to the dir containing the ground truth datasets.", default="gt/")
parser.add_argument("--rs_real", type=str2bool,help="If the input image is remote sensing HSI.", default=0)

#inference
parser.add_argument("--stride_test", type=int, default=12, help='stride of overlapping image blocks [4,8,16,24,48] kernel_//stride')
parser.add_argument("--block_inference", type=str2bool, default=True,help='if true process blocks of large image in paralel')
parser.add_argument("--pad_image", type=str2bool, default=0,help='padding strategy for inference')
parser.add_argument("--pad_block", type=str2bool, default=1,help='padding strategy for inference')
parser.add_argument("--pad_patch", type=str2bool, default=0,help='padding strategy for inference')
parser.add_argument("--no_pad", type=str2bool, default=False, help='padding strategy for inference')
parser.add_argument("--custom_pad", type=int, default=None,help='padding strategy for inference')
parser.add_argument("--verbose", type=str2bool, default=1)
parser.add_argument("--test_batch", type=int, default=1, help='batch size of testing')

args = parser.parse_args()
# os.environ['CUDA_VISIBLE_DEVICES']= '6,7'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
capability = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else os.cpu_count()
#device= torch.device("cpu")
gpus=args.gpus
test_path = [args.test_path]
gt_path = args.gt_path
print(f'test data : {test_path}')
print(f'gt data : {gt_path}')
train_path = val_path = []

noise_std = args.noise_level / 255
loaders = dataloaders_hsi_test.get_dataloaders(test_path,drop_last=True,verbose=True)
model=init_model(kernel_size=args.kernel_size,num_filters=args.num_filters,unfoldings=args.unfoldings,lmbda_prox=args.lmbda_prox,stride=args.stride,multi_theta=args.multi_theta,verbose=args.verbose)
load_model(model_name=args.model_name, model=model)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
if device.type == 'cuda':
    torch.cuda.set_device('cuda:{}'.format(gpus[0]))
if device.type == 'cuda':
    model = torch.nn.DataParallel(model.to(device=device), device_ids=gpus)
model.eval()  # Set model to evaluate mode
num_iters = 0
psnr_tot = 0
stride_test = args.stride_test
loader = loaders['test']
num_iters = 0
psnr_tot = []
ssim_tot = []
fsim_tot = []
ergas_tot = []
sam_tot=[]
tic = time.time()
for batch,fname in tqdm(loader,disable=not args.tqdm):
    batch = batch.to(device=device)
    fname=fname[0]
    print(fname)
    noisy_batch = batch
    if args.noise_level==0:
        sigma_est = np.array(estimate_sigma(noisy_batch.squeeze(0).permute([1, 2, 0]).detach().cpu(), multichannel=True,
                                            average_sigmas=False)).max() * 255
    with torch.set_grad_enabled(False):

        if args.block_inference:
            params = {
                'crop_out_blocks': 0,
                'ponderate_out_blocks': 1,
                'sum_blocks': 0,
                'pad_even': 1,  # otherwise pad with 0 for las
                'centered_pad': 0,  # corner pixel have only one estimate
                'pad_block': args.pad_block,  # pad so each pixel has S**2 estimate
                'pad_patch': args.pad_patch,  # pad so each pixel from the image has at least S**2 estimate from 1 block
                'no_pad': args.no_pad,
                'custom_pad': args.custom_pad,
                'avg': 1}
            block = block_module(args.patch_size, stride_test, args.kernel_size, params)
            batch_noisy_blocks = block._make_blocks(noisy_batch)
            patch_loader = torch.utils.data.DataLoader(batch_noisy_blocks, batch_size=args.test_batch, drop_last=False)
            batch_out_blocks = torch.zeros_like(batch_noisy_blocks)
            for i, inp in enumerate(patch_loader):  # if it doesnt fit in memory
                id_from, id_to = i * patch_loader.batch_size, (i + 1) * patch_loader.batch_size
                batch_out_blocks[id_from:id_to] = model(inp)

            output = block._agregate_blocks(batch_out_blocks)
            # print(torch.isnan(output).sum())
        else:
            output = model(noisy_batch)
        gt = dataloaders_hsi_test.get_gt(gt_path, fname);
        gt = gt.to(device=device)
        if device_name == 'cpu':
            psnr_batch, ssim_batch,fsim_batch, sam_batch,ergas_batch = MSIQA(gt.detach().numpy(),
                                                      output.squeeze(0).detach().numpy())
           #scio.savemat(fname + 'Res.mat', {'output': output.squeeze(0).detach().numpy()})
        else:
            # psnr, ssim, fsim, sam, er
            psnr_batch, ssim_batch,fsim_batch, sam_batch,ergas_batch = MSIQA(gt.detach().cpu().numpy(),                                                      output.squeeze(0).detach().cpu().numpy())
           # scio.savemat(fname + 'Res.mat', {'output': output.squeeze(0).detach().cpu().numpy()})

        psnr_tot.append(psnr_batch)
        ssim_tot.append(ssim_batch)
        sam_tot.append(sam_batch)
        fsim_tot.append(fsim_batch)
        ergas_tot.append(ergas_batch)
        num_iters += 1
        tqdm.write(f'psnr avg {psnr_batch} ssim avg {ssim_batch} sam avg {sam_batch},  fsim avg {fsim_batch}, ergas avg {ergas_batch} ')
tac = time.time()
psnr_mean = np.mean(psnr_tot)
ssim_mean = np.mean(ssim_tot)
sam_mean = np.mean(sam_tot)
fsim_mean= np.mean(fsim_tot)
ergas_mean= np.mean(ergas_tot)


if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)
scio.savemat(args.out_dir + 'Res.mat', {'psnr': psnr_tot,'fsim':fsim_tot,'ergas': ergas_tot, 'ssim': ssim_tot, 'sam': sam_tot})
# psnr_tot = psnr_tot.item()

tqdm.write(
    f'psnr: {psnr_mean:0.4f}  ssim: {ssim_mean:0.4f} sam: {sam_mean:0.4f} fsim: {fsim_mean:0.4f} ergas: {ergas_mean:0.4f}({(tac - tic) / num_iters:0.3f} s/iter)')
