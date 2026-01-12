import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch
import numpy as np
import math
import os


def weights_init_normal_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)

    elif classname.find("BatchNorm2d") != -1 or classname.find("InstanceNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def apply_lut(img, LUT, repeat=False):
    # scale im between -1 and 1 since its used as grid input in grid_sample
    img = (img - .5) * 2.
    # grid_sample expects NxDxHxWx3 (1x1xHxWx3)
    img = img.permute(0, 2, 3, 1)[:, None]
    # add batch dim to LUT
    if repeat:
        LUT = LUT[None].repeat(img.shape[0],1,1,1,1)
    else:
        LUT = LUT[None]
    # apply LUT
    result = F.grid_sample(LUT, img, mode='bilinear', padding_mode='border', align_corners=True)
    # drop added dimensions and permute back
    result = result[:, :, 0]
    #self.LUT, output = self.TrilinearInterpolation(self.LUT, x)
    return result

class resnet18_224(nn.Module):

    def __init__(self, out_dim=5, aug_test=False):
        super(resnet18_224, self).__init__()

        self.aug_test = aug_test
        net = models.resnet18(pretrained=True)
        # self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        # self.std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

        self.upsample = nn.Upsample(size=(224,224),mode='bilinear')
        net.fc = nn.Linear(512, out_dim)
        self.model = net


    def forward(self, x):

        x = self.upsample(x)
        if self.aug_test:
            # x = torch.cat((x, torch.rot90(x, 1, [2, 3]), torch.rot90(x, 3, [2, 3])), 0)
            x = torch.cat((x, torch.flip(x, [3])), 0)
        f = self.model(x)

        return f

############################## Discriminator ##############################

def discriminator_block(in_filters, out_filters, normalization=False):
    """Returns downsampling layers of each discriminator block"""
    layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
    layers.append(nn.LeakyReLU(0.2))
    if normalization:
        layers.append(nn.BatchNorm2d(out_filters))
        #layers.append(nn.BatchNorm2d(out_filters))

    return layers

############################## Classifier ##############################

class Classifier_class2(nn.Module):
    def __init__(self, in_channels=3,lut=5):
        super(Classifier_class2, self).__init__()

        self.model = nn.Sequential(
            nn.Upsample(size=(512,512),mode='nearest'),
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(16),
            *discriminator_block(16, 32, normalization=True),
            *discriminator_block(32, 64, normalization=True),
            *discriminator_block(64, 128, normalization=True),
            *discriminator_block(128, 128, normalization=True),
            *discriminator_block(128, 128),
            #*discriminator_block(128, 128, normalization=True),
            #nn.Dropout(p=0.5),
            nn.Conv2d(128, lut, 8, padding=0),
        )
        self.lr = nn.Linear(lut, 1)

    def forward(self, img_input):
        out = self.model(img_input)
        out1 = out.view(img_input.shape[0], -1)
        out2 = self.lr(out1)
        return out, out2

############################## Traditional 3DLUT ##############################

class Generator3DLUT_hr_lut(nn.Module):
    def __init__(self, dim=33, filename='model/daltnet/lut3d_rgb.txt'):
        super(Generator3DLUT_hr_lut, self).__init__()
        file = open(filename)
        lines = file.readlines()
        buffer = np.zeros((3,dim,dim,dim), dtype=np.float32)
        for i in range(0,dim):
            for j in range(0,dim):
                for k in range(0,dim):
                    n = i * dim*dim + j * dim + k
                    x = lines[n].split()
                    buffer[0,i,j,k] = float(x[0]) / 255.
                    buffer[1,i,j,k] = float(x[1]) / 255.
                    buffer[2,i,j,k] = float(x[2]) / 255.
        self.LUT = nn.Parameter(torch.from_numpy(buffer).requires_grad_(True))

    def forward(self, x):
        # scale im between -1 and 1 since its used as grid input in grid_sample
        x = (x - .5) * 2.
        # grid_sample expects NxDxHxWx3 (1x1xHxWx3)
        x = x.permute(0, 2, 3, 1)[:, None]
        # add batch dim to LUT
        LUT = self.LUT[None].repeat(x.shape[0],1,1,1,1)
        # apply LUT
        result = F.grid_sample(LUT, x, mode='bilinear', padding_mode='border', align_corners=True)
        # drop added dimensions and permute back
        result = result[:, :, 0]
        #self.LUT, output = self.TrilinearInterpolation(self.LUT, x)
        return result

############################## Adaptive 3DLUT ##############################

class Generator3DLUT_identity(nn.Module):
    def __init__(self, dim=33):
        super(Generator3DLUT_identity, self).__init__()
        if dim == 9:
            file = open("model/daltnet/IdentityLUT9.txt", 'r')
        elif dim == 17:
            file = open("model/daltnet/IdentityLUT17.txt", 'r')
        lines = file.readlines()
        buffer = np.zeros((3,dim,dim,dim), dtype=np.float32)

        for i in range(0,dim):
            for j in range(0,dim):
                for k in range(0,dim):
                    n = i * dim*dim + j * dim + k
                    x = lines[n].split()
                    buffer[0,i,j,k] = float(x[0])
                    buffer[1,i,j,k] = float(x[1])
                    buffer[2,i,j,k] = float(x[2])
        self.LUT = nn.Parameter(torch.from_numpy(buffer).requires_grad_(True))

    def forward(self, x):
        x = (x - .5) * 2. # scale im between -1 and 1 since its used as grid input in grid_sample
        x = x.permute(0, 2, 3, 1)[:, None]  # grid_sample expects NxDxHxWx3 (1x1xHxWx3)
        LUT = self.LUT[None].repeat(x.shape[0],1,1,1,1) # add batch dim to LUT
        result = F.grid_sample(LUT, x, mode='bilinear', padding_mode='border', align_corners=True)  # apply LUT
        result = result[:, :, 0]  # drop added dimensions
        return result

############################## TV & MN Calculations ##############################

class TV_3D(nn.Module):
    def __init__(self, dim=33):
        super(TV_3D,self).__init__()

        self.weight_r = torch.ones(3,dim,dim,dim-1, dtype=torch.float)
        self.weight_r[:,:,:,(0,dim-2)] *= 2.0
        self.weight_g = torch.ones(3,dim,dim-1,dim, dtype=torch.float)
        self.weight_g[:,:,(0,dim-2),:] *= 2.0
        self.weight_b = torch.ones(3,dim-1,dim,dim, dtype=torch.float)
        self.weight_b[:,(0,dim-2),:,:] *= 2.0
        self.relu = torch.nn.ReLU()

    def forward(self, LUT):

        dif_r = LUT.LUT[:,:,:,:-1] - LUT.LUT[:,:,:,1:]
        dif_g = LUT.LUT[:,:,:-1,:] - LUT.LUT[:,:,1:,:]
        dif_b = LUT.LUT[:,:-1,:,:] - LUT.LUT[:,1:,:,:]
        tv = torch.mean(torch.mul((dif_r ** 2),self.weight_r)) + torch.mean(torch.mul((dif_g ** 2),self.weight_g)) + torch.mean(torch.mul((dif_b ** 2),self.weight_b))

        mn = torch.mean(self.relu(dif_r)) + torch.mean(self.relu(dif_g)) + torch.mean(self.relu(dif_b))

        return tv, mn

class TV_3D_LUT(nn.Module):  # only code difference is using LUT as value instead of object; for implementation, this is for fused LUT
    def __init__(self, dim=33):
        super(TV_3D_LUT,self).__init__()

        self.weight_r = torch.ones(3,dim,dim,dim-1, dtype=torch.float)
        self.weight_r[:,:,:,(0,dim-2)] *= 2.0
        self.weight_g = torch.ones(3,dim,dim-1,dim, dtype=torch.float)
        self.weight_g[:,:,(0,dim-2),:] *= 2.0
        self.weight_b = torch.ones(3,dim-1,dim,dim, dtype=torch.float)
        self.weight_b[:,(0,dim-2),:,:] *= 2.0
        self.relu = torch.nn.ReLU()

    def forward(self, LUT):
        # [r/g/b, dim, dim, dim]
        # LUT[:,:,:,:-1] - LUT[:,:,:,1:] = (R+G+B)(all points) of diff with redder point
        # LUT[:,:,:,:-1]: 0, 1, ..., n-1
        # LUT[:,:,:,1:]: 1, ..., n
        # each dif_x is 4D tensor of neighbor-wise differences (like subtraction of shifted spaces)
        dif_r = LUT[:,:,:,:-1] - LUT[:,:,:,1:]  # Red direction of LUT
        dif_g = LUT[:,:,:-1,:] - LUT[:,:,1:,:]  # Green direction of LUT
        dif_b = LUT[:,:-1,:,:] - LUT[:,1:,:,:]  # Blue direction of LUT
        tv = torch.mean(torch.mul((dif_r ** 2),self.weight_r)) + torch.mean(torch.mul((dif_g ** 2),self.weight_g)) + torch.mean(torch.mul((dif_b ** 2),self.weight_b))

        mn = torch.mean(self.relu(dif_r)) + torch.mean(self.relu(dif_g)) + torch.mean(self.relu(dif_b))

        return tv, mn


