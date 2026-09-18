#python train.py --epoch 0 --n_epochs 121 --batch_size 8 --checkpoint_interval 10 --dataset_name cvd_100_001 --sample 5000  --ssimori 1 --cvd--lambda_ssim 0.00 --points 3000


import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.utils import save_image
from models_x import *
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from torchsummary import summary
# from swintransformer import *

#from gan_cnn import *
from datasets import *
from discrimintor_trs import *
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import nn as nn
from cvd_function import *
from contrast import *
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from kornia.color import rgb_to_lab
import torchvision.models as models
from utils.color_torch import rgb2yuv_bt709, yuv2rgb_bt709

# generator = models.resnet18(pretrained=False)
# generator.fc = torch.nn.Linear(512, 17 * 17 * 2)
# generator = models.mobilenet_v3_large(pretrained=False, num_classes=17 * 17 * 2)
generator = models.shufflenet_v2_x1_0(pretrained=False, num_classes=17 * 17 * 2)
# generator = models.squeezenet1_1(pretrained=False, num_classes=17 * 17 * 2)
# generator = SwinTransformer(num_classes=17 * 17 * 2)
# CUDA setup
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

generator = generator.cuda()

print("cuda!!!!!!!")
ckpt_dp = torch.load("saved_models/cvd_100_001_P_3000_ori_1_ssimori_1_ssim_0.25_dis_dim_17_2dlut_shufflenet_v2/generator_200.pth")

ckpt = {}
for idx, key in enumerate(ckpt_dp):
    ckpt[key.replace('module.', '')] = ckpt_dp[key]
generator.load_state_dict(ckpt)
generator.eval()

generator_t = Generator_transformer_pathch4_844_48_3_nouplayer_server5()

# CUDA setup
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

generator_t = generator_t.cuda()

print("cuda!!!!!!!")
ckpt_dp = torch.load("saved_models/cvd_100_001_P_labonlyG+global_con+nature_points3000_ori_1_ssimori_1_ssim_0.25/generator_80.pth")

ckpt = {}
for idx, key in enumerate(ckpt_dp):
    ckpt[key.replace('module.', '')] = ckpt_dp[key]
generator_t.load_state_dict(ckpt)
generator_t.eval()





# Configure dataloaders
transforms_ = [
    # transforms.Resize((256, 256), Image.BICUBIC),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]


# data_path = '/path/to/data'
data_path = "/path/to/CVDdataset/Color_cvd_P_experiment_100000"
# data_path = "/path/to/data"
transform_val_list1 = [transforms.Normalize((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0))]
transform_val_list2 = [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
transform_val_list3 = [transforms.Normalize((0, 0, 0), (100, 128, 128))]

trans_compose1 = transforms.Compose(transform_val_list1)
trans_compose2 = transforms.Compose(transform_val_list2)
trans_compose3 = transforms.Compose(transform_val_list3)
# Tensor type
Tensor = torch.cuda.FloatTensor
print(Tensor)

dim=17
def lut_2d(LUT, x):
    x = (x - .5) * 2.
    x = x.permute(0, 2, 3, 1)
    
    result = F.grid_sample(LUT, x, mode='bilinear', padding_mode='border', align_corners=True)
    return result

def generator_train(img,img_name):
    img_yuv = rgb2yuv_bt709(img)
    LUT = generator(img_yuv)
    # LUT = LUT.clip(0,1)


    LUT = LUT.reshape(LUT.shape[0], 2, dim, dim)
    arr = LUT[0].detach().cpu().numpy()
    np.save('lut.npy', arr)
    save_image(LUT[0, 0], "lut_u.png",normalize=True)
    save_image(LUT[0, 1], "lut_v.png",normalize=True)

    uv = lut_2d(LUT, img_yuv[:,1:,:,:])
    out_yuv = torch.cat((img_yuv[:,0:1,:,:],uv), dim=1)
    out = yuv2rgb_bt709(out_yuv)
    out = out.clip(0,1)
    return out 
import random
def sample_images(real_A, img_name):
    """Saves a generated sample from the validation set"""
    fake_t = trans_compose1(generator_t(trans_compose2(real_A))).clip(0,1)
    fake_B = generator_train(real_A,img_name)
    real_A = (real_A).clip(0,1)
    # real_B = trans_compose1(real_B)
    fake_B = (fake_B).clip(0,1)

    # cvd_real = cvd_simulation_tensors(real_B, 0, 100)
    cvd_fake = cvd_simulation_tensors(fake_B, 0, 100)
    cvd_fake_t = cvd_simulation_tensors(fake_t, 0, 100)
    cvd_original = cvd_simulation_tensors(real_A, 0, 100)

    img_sample_1 = torch.cat((real_A.data, fake_B.data,fake_t.data), 3)
    img_sample_2 = torch.cat((cvd_original.data, cvd_fake.data, cvd_fake_t.data), 3)
    # print(real_A.data,cvd_original.data)
    img_sample = torch.cat((img_sample_1.data, img_sample_2.data), 2)

    # img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
    # save_image(img_sample, "result/P_2d_lut_squeezenet1_1/concat/%s" % (img_name),normalize=True)
    # save_image(fake_B, "result/P_2d_lut_squeezenet1_1/pred/%s" % (img_name),normalize=True)


# ----------
#  Training
# ----------
os.makedirs("result/P_2d_lut_squeezenet1_1/",exist_ok=True)
os.makedirs("result/P_2d_lut_squeezenet1_1/concat",exist_ok=True)
os.makedirs("result/P_2d_lut_squeezenet1_1/pred",exist_ok=True)
# os.makedirs("save_result/P_2d_lut_resnet18/ckpt",exist_ok=True)
for img_name in os.listdir(data_path):  
    img = Image.open(os.path.join(data_path, img_name)).convert("RGB")
    img = transforms.Compose(transforms_)(img).unsqueeze(0).cuda()
    with torch.no_grad():
        # print(img_name)
        # print(torch.max(img), torch.min(img))
        sample_images(img, img_name)
    break
