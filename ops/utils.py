import torch
import torch.functional as F
from random import randint
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.measure import compare_ssim, compare_psnr
import dataloaders_hsi_test
# from sewar.full_ref import sam,psnr,
# from ops.gauss import *
from ops.gauss import fspecial_gauss
from scipy import signal
import cv2
import phasepack.phasecong as pc

from scipy import ndimage
def kronecker(A, B):
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))
def gen_bayer_mask(h,w):
    x = torch.zeros(1, 3, h, w)

    x[:, 0, 1::2, 1::2] = 1  # r
    x[:, 1, ::2, 1::2] = 1
    x[:, 1, 1::2, ::2] = 1  # g
    x[:, 2, ::2, ::2] = 1  # b

    return x

def togray(tensor):
    b, c, h, w = tensor.shape
    tensor = tensor.view(b, 3, -1, h, w)
    tensor = tensor.sum(1)
    return tensor

def torch_to_np(img_var):
    return img_var.detach().cpu().numpy()

def plot_tensor(img, **kwargs):
    inp_shape = tuple(img.shape)
    print(inp_shape)
    img_np = torch_to_np(img)
    if inp_shape[1]==3:
        img_np_ = img_np.transpose([1,2,0])
        plt.imshow(img_np_)

    elif inp_shape[1]==1:
        img_np_ = np.squeeze(img_np)
        plt.imshow(img_np_, **kwargs)

    else:
        # raise NotImplementedError
        plt.imshow(img_np, **kwargs)
    plt.axis('off')


def get_mask(A):
    mask = A.clone().detach()
    mask[A != 0] = 1
    return mask.byte()

def sparsity(A):
    return get_mask(A).sum().item()/A.numel()

def soft_threshold(x, lambd):
    return nn.functional.relu(x - lambd,inplace=True) - nn.functional.relu(-x - lambd,inplace=True)
def nn_threshold(x, lambd):
    return nn.functional.relu(x - lambd)

def fastSoftThrs(x, lmbda):
    return x + 0.5 * (torch.abs(x-torch.abs(lmbda))-torch.abs(x+torch.abs(lmbda)))

def save_checkpoint(state,ckpt_path):
    torch.save(state, ckpt_path)

def generate_key():
    return '{}'.format(randint(0, 100000))

def show_mem():
    mem = torch.cuda.memory_allocated() * 1e-6
    max_mem = torch.cuda.max_memory_allocated() * 1e-6
    return mem, max_mem

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def step_lr(optimizer, lr_decay):
    lr = optimizer.param_groups[0]['lr']
    optimizer.param_groups[0]['lr'] = lr * lr_decay

def step_lr_als(optimizer, lr_decay):
    lr = optimizer.param_groups[0]['lr']
    optimizer.param_groups[0]['lr'] = lr * lr_decay
    optimizer.param_groups[1]['lr'] *= lr_decay

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def gen_mask_windows(h, w):
    '''
    return mask for block window
    :param h:
    :param w:
    :return: (h,w,h,w)
    '''
    mask = torch.zeros(2 * h, 2 * w, h, w)
    for i in range(h):
        for j in range(w):
            mask[i:i + h, j:j + w, i, j] = 1

    return mask[h // 2:-h // 2, w // 2:-w // 2, :, :]


def gen_linear_mask_windows(h, w, h_,w_):
    '''
    return mask for block window
    :param h:
    :param w:
    :return: (h,w,h,w)
    '''

    x = torch.ones(1, 1, h - h_ + 1, w - w_ + 1)
    k = torch.ones(1, 1, h_, w_)
    kernel = F.conv_transpose2d(x, k)
    kernel /= kernel.max()
    mask = torch.zeros(2 * h, 2 * w, h, w)
    for i in range(h):
        for j in range(w):
            mask[i:i + h, j:j + w, i, j] = kernel

    return mask[h // 2:-h // 2, w // 2:-w // 2, :, :]

def gen_quadra_mask_windows(h, w, h_,w_):
    '''
    return mask for block window
    :param h:
    :param w:
    :return: (h,w,h,w)
    '''

    x = torch.ones(1, 1, h - h_ + 1, w - w_ + 1)
    k = torch.ones(1, 1, h_, w_)
    kernel = F.conv_transpose2d(x, k) **2
    kernel /= kernel.max()
    mask = torch.zeros(2 * h, 2 * w, h, w)
    for i in range(h):
        for j in range(w):
            mask[i:i + h, j:j + w, i, j] = kernel

    return mask[h // 2:-h // 2, w // 2:-w // 2, :, :]

def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.

    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2, 0, 1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.


def np_to_pil(img_np):
    '''Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)
def Init_DCT(n, m):
    """ Compute the Overcomplete Discrete Cosinus Transform. """
    n=int(n)
    m=int(m)
    Dictionary = np.zeros((n,m))
    for k in range(m):
        V = np.cos(np.arange(0, n) * k * np.pi / m)
        if k > 0:
            V = V - np.mean(V)
        Dictionary[:, k] = V / np.linalg.norm(V)
    # Dictionary = np.kron(Dictionary, Dictionary)
    # Dictionary = Dictionary.dot(np.diag(1 / np.sqrt(np.sum(Dictionary ** 2, axis=0))))
    # idx = np.arange(0, n ** 2)
    # idx = idx.reshape(n, n, order="F")
    # idx = idx.reshape(n ** 2, order="C")
    # Dictionary = Dictionary[idx, :]
    Dictionary = torch.from_numpy(Dictionary).float()
    return Dictionary

def est_noise(y, noise_type='additive'):
    """
    This function infers the noise in a
    hyperspectral data set, by assuming that the
    reflectance at a given band is well modelled
    by a linear regression on the remaining bands.

    Parameters:
        y: `numpy array`
            a HSI cube ((m*n) x p)

       noise_type: `string [optional 'additive'|'poisson']`

    Returns: `tuple numpy array, numpy array`
        * the noise estimates for every pixel (N x p)
        * the noise correlation matrix estimates (p x p)

    Copyright:
        Jose Nascimento (zen@isel.pt) and Jose Bioucas-Dias (bioucas@lx.it.pt)
        For any comments contact the authors
    """
    # def est_additive_noise(r):
    #     small = 1e-6
    #     L, N = r.shape
    #     w=np.zeros((L,N), dtype=np.float)
    #     RR=np.dot(r,r.T)
    #     RRi = np.linalg.pinv(RR+small*np.eye(L))
    #     RRi = np.matrix(RRi)
    #     for i in range(L):
    #         XX = RRi - (RRi[:,i]*RRi[i,:]) / RRi[i,i]
    #         RRa = RR[:,i]
    #         RRa[i] = 0
    #         beta = np.dot(XX, RRa)
    #         beta[0,i]=0;
    #         w[i,:] = r[i,:] - np.dot(beta,r)
    #     Rw = np.diag(np.diag(np.dot(w,w.T) / N))
    #     return w, Rw
    def est_additive_noise(r):
        small = 1e-6
        L, N = r.shape
        w=torch.zeros((L,N), dtype=torch.float,device=r.device)
        RR=r@r.T
        RRi = torch.pinverse(RR+small*torch.eye(L))
        # RRi = np.matrix(RRi)
        for i in range(L):
            XX = RRi - (RRi[:,i]*RRi[i,:]) / RRi[i,i]
            RRa = RR[:,i]
            RRa[i] = 0
            beta =XX@RRa
            beta[i]=0;
            w[i,:] = r[i,:] - np.dot(beta,r)
        Rw = torch.diag(torch.diag((w@w.T) / N))
        return w, Rw

    h, w, numBands = y.shape
    y = torch.reshape(y, (w * h, numBands))
    y = y.T
    L, N = y.shape
    # verb = 'poisson'
    if noise_type == 'poisson':
        sqy = torch.sqrt(y * (y > 0))
        u, Ru = est_additive_noise(sqy)
        x = (sqy - u) ** 2
        w = torch.sqrt(x) * u * 2
        Rw = (w@w.T) / N
    # additive
    else:
        w, Rw = est_additive_noise(y)
    return w.T, Rw.T

    # y = y.T
    # L, N = y.shape
    # #verb = 'poisson'
    # if noise_type == 'poisson':
    #     sqy = np.sqrt(y * (y > 0))
    #     u, Ru = est_additive_noise(sqy)
    #     x = (sqy - u)**2
    #     w = np.sqrt(x)*u*2
    #     Rw = np.dot(w,w.T) / N
    # # additive
    # else:
    #     w, Rw = est_additive_noise(y)
    # return w.T, Rw.T


def hysime(y, n, Rn):
    """
    Hyperspectral signal subspace estimation

    Parameters:
        y: `numpy array`
            hyperspectral data set (each row is a pixel)
            with ((m*n) x p), where p is the number of bands
            and (m*n) the number of pixels.

        n: `numpy array`
            ((m*n) x p) matrix with the noise in each pixel.

        Rn: `numpy array`
            noise correlation matrix (p x p)

    Returns: `tuple integer, numpy array`
        * kf signal subspace dimension
        * Ek matrix which columns are the eigenvectors that span
          the signal subspace.

    Copyright:
        Jose Nascimento (zen@isel.pt) & Jose Bioucas-Dias (bioucas@lx.it.pt)
        For any comments contact the authors
    """
    h, w, numBands = y.shape
    y = torch.reshape(y, (w * h, numBands))
    y=y.T
    n=n.T
    Rn=Rn.T
    L, N = y.shape
    Ln, Nn = n.shape
    d1, d2 = Rn.shape

    x = y - n;

    Ry = y@y.T / N
    Rx = x@x.T/ N
    E, dx, V =torch.svd(Rx)

    Rn = Rn+torch.sum(torch.diag(Rx))/L/10**5 * torch.eye(L)
    Py = torch.diag(E.T@(Ry@E))
    Pn = torch.diag(E.T@(Rn@E))
    cost_F = -Py + 2 * Pn
    kf = torch.sum(cost_F < 0)
    ind_asc = torch.argsort(cost_F)
    Ek = E[:, ind_asc[0:kf]]
    return kf, E # Ek.T ?
def count(M):
    w, Rw = est_noise(M)
    kf, Ek = hysime(M, w, Rw)
    return kf, Ek
def cal_sam(X, Y, eps=2.2204e-16):
    # X = torch.squeeze(X.data).cpu().numpy()
    # Y = torch.squeeze(Y.data).cpu().numpy()
    # X=X*255
    # Y=Y*255
    tmp = (np.sum(X*Y, axis=0) + eps) / ((np.sqrt(np.sum(X**2, axis=0))) * (np.sqrt(np.sum(Y**2, axis=0)))+eps)
    return np.mean(np.real(np.arccos(tmp)))
def cal_psnr(im_true,im_test,eps=13-8):
    c,_,_=im_true.shape
    bwindex = []
    for i in range(c):
        bwindex.append(compare_psnr(im_true[i,:,:], im_test[i,:,:]))
    return  np.mean(bwindex)
def cal_ssim(im_true,im_test,eps=13-8):
    c,_,_=im_true.shape
    bwindex = []
    for i in range(c):
        bwindex.append(ssim(im_true[i,:,:]*255, im_test[i,:,:,]*255))
    return np.mean(bwindex)
def cal_fsim(im_true,im_test,eps=13-8):
    c,_,_=im_true.shape
    fs=fsim(np.transpose(im_true,[1,2,0]),np.transpose(im_test,[1,2,0]))
    return fs
    # bwindex = []
    # for i in range(c):
    #     bwindex.append(fsim(im_true[i,:,:], im_test[i,:,:,]))
    # return np.mean(bwindex)
def fsim(org_img: np.ndarray, pred_img: np.ndarray, T1=0.85, T2=160) -> float:
    """
    Feature-based similarity index, based on phase congruency (PC) and image gradient magnitude (GM)
    There are different ways to implement PC, the authors of the original FSIM paper use the method
    defined by Kovesi (1999). The Python phasepack project fortunately provides an implementation
    of the approach.
    There are also alternatives to implement GM, the FSIM authors suggest to use the Scharr
    operation which is implemented in OpenCV.
    Note that FSIM is defined in the original papers for grayscale as well as for RGB images. Our use cases
    are mostly multi-band images e.g. RGB + NIR. To accommodate for this fact, we compute FSIM for each individual
    band and then take the average.
    Note also that T1 and T2 are constants depending on the dynamic range of PC/GM values. In theory this parameters
    would benefit from fine-tuning based on the used data, we use the values found in the original paper as defaults.
    Args:
        org_img -- numpy array containing the original image
        pred_img -- predicted image
        T1 -- constant based on the dynamic range of PC values
        T2 -- constant based on the dynamic range of GM values
    """
    _assert_image_shapes_equal(org_img, pred_img, "FSIM")

    alpha = beta = 1  # parameters used to adjust the relative importance of PC and GM features
    fsim_list = []
    for i in range(org_img.shape[2]):
        # Calculate the PC for original and predicted images
        pc1_2dim = pc(org_img[:, :, i], nscale=4, minWaveLength=6, mult=2, sigmaOnf=0.55)
        pc2_2dim = pc(pred_img[:, :, i], nscale=4, minWaveLength=6, mult=2, sigmaOnf=0.55)

        # pc1_2dim and pc2_2dim are tuples with the length 7, we only need the 4th element which is the PC.
        # The PC itself is a list with the size of 6 (number of orientation). Therefore, we need to
        # calculate the sum of all these 6 arrays.
        pc1_2dim_sum = np.zeros((org_img.shape[0], org_img.shape[1]), dtype=np.float64)
        pc2_2dim_sum = np.zeros((pred_img.shape[0], pred_img.shape[1]), dtype=np.float64)
        for orientation in range(6):
            pc1_2dim_sum += pc1_2dim[4][orientation]
            pc2_2dim_sum += pc2_2dim[4][orientation]

        # Calculate GM for original and predicted images based on Scharr operator
        gm1 = _gradient_magnitude(org_img[:, :, i], cv2.CV_16U)
        gm2 = _gradient_magnitude(pred_img[:, :, i], cv2.CV_16U)

        # Calculate similarity measure for PC1 and PC2
        S_pc = _similarity_measure(pc1_2dim_sum, pc2_2dim_sum, T1)
        # Calculate similarity measure for GM1 and GM2
        S_g = _similarity_measure(gm1, gm2, T2)

        S_l = (S_pc ** alpha) * (S_g ** beta)

        numerator = np.sum(S_l * np.maximum(pc1_2dim_sum, pc2_2dim_sum))
        denominator = np.sum(np.maximum(pc1_2dim_sum, pc2_2dim_sum))
        fsim_list.append(numerator / denominator)

    return np.mean(fsim_list)

def _assert_image_shapes_equal(org_img: np.ndarray, pred_img: np.ndarray, metric: str):
    msg = (f"Cannot calculate {metric}. Input shapes not identical. y_true shape ="
           f"{str(org_img.shape)}, y_pred shape = {str(pred_img.shape)}")

    assert org_img.shape == pred_img.shape, msg
def _gradient_magnitude(img: np.ndarray, img_depth):
    """
    Calculate gradient magnitude based on Scharr operator
    """
    scharrx = cv2.Scharr(img, img_depth, 1, 0)
    scharry = cv2.Scharr(img, img_depth, 0, 1)

    return np.sqrt(scharrx ** 2 + scharry ** 2)
def _similarity_measure(x, y, constant):
    """
    Calculate feature similarity measurement between two images
    """
    numerator = 2 * x * y + constant
    denominator = x ** 2 + y ** 2 + constant

    return numerator / denominator
# # class Bandwise(object):
# #     def __init__(self, index_fn):
# #         self.index_fn = index_fn
# #
# #     def __call__(self, X, Y):
# #         C = X.shape[-3]
# #         bwindex = []
# #         for ch in range(C):
# #             x = torch.squeeze(X[...,ch,:,:].data).cpu().numpy()
# #             y = torch.squeeze(Y[...,ch,:,:].data).cpu().numpy()
# #             index = self.index_fn(x, y)
# #             bwindex.append(index)
# #         return bwindex
def ssim(img1, img2, cs_map=False):
    """Return the Structural Similarity Map corresponding to input images img1
    and img2 (images are assumed to be uint8)

    This function attempts to mimic precisely the functionality of ssim.m a
    MATLAB provided by the author's of SSIM
    https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255  # bitdepth of image
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu1 = signal.fftconvolve(window, img1, mode='valid')
    mu2 = signal.fftconvolve(window, img2, mode='valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = signal.fftconvolve(window, img1 * img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2 * img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1 * img2, mode='valid') - mu1_mu2
    if cs_map:
        return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                             (sigma1_sq + sigma2_sq + C2)),
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
def mse (GT,P):
	"""calculates mean squared error (mse).

	:param GT: first (original) input image.
	:param P: second (deformed) input image.

	:returns:  float -- mse value.
	"""
	# GT,P = _initial_check(GT,P)
	return np.mean((GT.astype(np.float32)-P.astype(np.float32))**2)
def cal_ergas(im_true,im_test):
    m,n,k=im_true.shape
    mm, nn, kk = im_test.shape
    m=min(m,mm)
    n = min(n, nn)
    k = min(k, kk)
    im_true=im_true[0:m, 0: n, 0: k]*255
    im_test=im_test[0:m, 0: n, 0: k]*255
    ergas = 0;
    for i in range(m):
        ergas = ergas + mse(im_true[i,:,:],im_test[i,:,:]) / np.mean(im_test[i,:,:]);
    return 100 * np.sqrt(ergas / m)
     # ergas
    # [m, n, k] = size(imagery1);
    # [mm, nn, kk] = size(imagery2);
    # m = min(m, mm);
    # n = min(n, nn);
    # k = min(k, kk);
    # imagery1 = imagery1(1:m, 1: n, 1: k);
    # imagery2 = imagery2(1:m, 1: n, 1: k);
    #
    # ergas = 0;
    # for i = 1:k
    # ergas = ergas + mse(imagery1(:,:, i) - imagery2(:,:, i)) / mean2(imagery1(:,:, i));
    # end
    # ergas = 100 * sqrt(ergas / k);
def MSIQA(X, Y):
    # print(X.shape)
    # print(Y.shape)
    psnr = cal_psnr(X, Y)
    ssim = cal_ssim(X, Y)
    fsim=cal_fsim(X, Y)
    sam = cal_sam(X, Y)
    er=cal_ergas(X, Y)

    return psnr, ssim,fsim, sam,er

if __name__ == '__main__':
    gt=dataloaders_hsi_test.get_gt('../','watercolors_ms.mat')
    gt = gt / gt.max()
    res = dataloaders_hsi_test.get_gt('../', 'AvgSigmaResHysime300_6TrueIter6_56_9_5_12_95.mat')
    psnr, ssim, sam=MSIQA(gt.numpy(), res.numpy())
    # hsi = torch.rand(200,200, 198)
    # w, Rw=est_noise(hsi)
    # kf, E= hysime(hsi, w, Rw)
    # print(kf)




