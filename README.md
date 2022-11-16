
# SMDS-Net: Model Guided Spectral-Spatial Network for Hyperspectral Image Denoising

## Fengchao Xiong; Jun Zhou; Shuyin Tao; Jianfeng Lu; Jiantao Zhou; Yuntao Qian

[Link to paper](https://ieeexplore.ieee.org/abstract/document/9855427/)

# Abstract

Deep learning (DL) based hyperspectral images (HSIs) denoising approaches directly learn the nonlinear mapping between noisy and clean HSI pairs. They usually do not consider the physical characteristics of HSIs. This drawback makes the models lack interpretability that is key to understanding their denoising mechanism and limits their denoising ability. In this paper, we introduce a novel model-guided interpretable network for HSI denoising to tackle this problem. Fully considering the spatial redundancy, spectral low-rankness, and spectral-spatial correlations of HSIs, we first establish a subspace-based multidimensional sparse (SMDS) model under the umbrella of tensor notation. After that, the model is unfolded into an end-to-end network named SMDS-Net, whose fundamental modules are seamlessly connected with the denoising procedure and optimization of the SMDS model. This makes SMDS-Net convey clear physical meanings, i.e., learning the low-rankness and sparsity of HSIs. Finally, all key variables are obtained by discriminative training. Extensive experiments and comprehensive analysis on synthetic and real-world HSIs confirm the strong denoising ability, strong learning capability, promising generalization ability, and high interpretability of SMDS-Net against the state-of-the-art HSI denoising methods. The source code and data of this article will be made publicly available at https://github.com/bearshng/smds-net for reproducible research.

# Requirements

We tested the implementation in Python 3.7.




# Datasets

* The ICVL dataset can be downloaded from [Link to Dataset](http://icvl.cs.bgu.ac.il/hyperspectral/).
 
* The 100 HSIs used in our training can be found in [ICVL_train.txt](https://github.com/bearshng/mac-net/blob/master/ICVL_train.txt).


* The 50 HSIs used for testing can be found in [ICVL_test.txt](https://github.com/bearshng/mac-net/blob/master/ICVL_test_gauss.txt).








### test

`python test.py --unfoldings 6  --num_filters 9 --kernel_size 9 --stride_test 12  --test_path 'XXX'  --gt_path 'XXX' --model_name 'trained_model/ICVL_15_ckpt' --gpus 0   --verbose 0  --multi_theta 1  --patch_size 56  --test_batch 12
`


### train


`python train.py --unfoldings 6 --lr 5e-3 --patch_size 56 --train_path 'XXX' --num_filters 9 --kernel_size 9   --log_dir './trained_model' --out_dir './trained_model'  --verbose 0 --multi_theta 1 --validation_every 400  --gpus 1  --noise_level 15  --num_epochs 300 --bandwise 1 --train_batch 2`



# Bibtex

@ARTICLE{9855427,  author={Xiong, Fengchao and Zhou, Jun and Tao, Shuyin and Lu, Jianfeng and Zhou, Jiantao and Qian, Yuntao},  journal={IEEE Transactions on Image Processing},   title={SMDS-Net: Model Guided Spectral-Spatial Network for Hyperspectral Image Denoising},   year={2022},  volume={31},  number={},  pages={5469-5483},  doi={10.1109/TIP.2022.3196826}}


# Contact Information:

Fengchao Xiong: fcxiong@njust.edu.cn

School of Computer Science and Engineering

Nanjing University of Science and Technology
