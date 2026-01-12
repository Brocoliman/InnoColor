import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch
import numpy as np
import math
import sys

# import trilinear

def weights_init_normal_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)

    elif classname.find("BatchNorm2d") != -1 or classname.find("InstanceNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def discriminator_block(in_filters, out_filters, normalization=False):
    """Returns downsampling layers of each discriminator block"""
    layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
    layers.append(nn.LeakyReLU(0.2))
    if normalization:
        layers.append(nn.BatchNorm2d(out_filters))
        #layers.append(nn.BatchNorm2d(out_filters))

    return layers






def discriminator_block2(in_filters, out_filters, normalization=False,act=False):
    """Returns downsampling layers of each discriminator block"""
    layers = [nn.Conv2d(in_filters, out_filters, 3, stride=1, padding=1)]
    
    if normalization:
        # layers.append(nn.InstanceNorm2d(out_filters, affine=True))
        layers.append(nn.BatchNorm2d(out_filters))
    if act:
        layers.append(nn.ReLU())
    else:
        layers.append(nn.LeakyReLU(0.2))
    return layers





class VGG(nn.Module):
    def __init__(self, in_channels=1, mid_channels=64,out_channels=360,norm=True):
        super(VGG, self).__init__()

        self.layer1 = nn.Sequential(
            *discriminator_block2(in_channels, mid_channels, normalization=norm, act=True),
            *discriminator_block2(mid_channels, mid_channels, normalization=norm, act=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )    # 1 -> 64 -> 64
        self.layer2 = nn.Sequential(
            *discriminator_block2(mid_channels, mid_channels*2, normalization=norm, act=True),
            *discriminator_block2(mid_channels*2, mid_channels*2, normalization=norm, act=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )   # 64 - 128 - 128
        self.layer3 = nn.Sequential(
            *discriminator_block2(mid_channels*2, mid_channels*4, normalization=norm, act=True),
            *discriminator_block2(mid_channels*4, mid_channels*4, normalization=norm, act=True),
            *discriminator_block2(mid_channels*4, mid_channels*4, normalization=norm, act=True),
            nn.AvgPool2d(kernel_size=4, stride=4)
        )   # 128 - 256 - 256
        self.layer4 = nn.Sequential(
            *discriminator_block2(mid_channels*4, mid_channels*8, normalization=norm, act=True),
            *discriminator_block2(mid_channels*8, mid_channels*8, normalization=norm, act=True),
            *discriminator_block2(mid_channels*8, mid_channels*8, normalization=norm, act=True),
            nn.AvgPool2d(kernel_size=4, stride=4)
        )
        self.layer5 = nn.Sequential(
            *discriminator_block2(mid_channels*8, mid_channels*8, normalization=norm, act=True),
            *discriminator_block2(mid_channels*8, mid_channels*8, normalization=norm, act=True),
            *discriminator_block2(mid_channels*8, mid_channels*8, normalization=norm, act=True),
            nn.AdaptiveAvgPool2d(output_size=(1,1))
        )
        self.conv = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5,
        )
        self.fc = nn.Sequential(
            nn.Linear(mid_channels*8,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,out_channels),
        )
        

    def forward(self, img_input):
        lr = F.interpolate(img_input, size=(512,512), mode='nearest')
        out = self.conv(lr)
        out = out.view(img_input.shape[0],-1)
        out = self.fc(out)
        return out
class Model_10L64C(nn.Module):
    def __init__(self, in_channels=1, mid_channels=96,out_channels=360,norm=True):
        super(Model_10L64C, self).__init__()

        self.layer1 = nn.Sequential(
            *discriminator_block2(in_channels, mid_channels, normalization=norm, act=True),
            *discriminator_block2(mid_channels, mid_channels, normalization=norm, act=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )    # 1 -> 64 -> 64
        self.layer2 = nn.Sequential(
            *discriminator_block2(mid_channels, mid_channels, normalization=norm, act=True),
            *discriminator_block2(mid_channels, mid_channels, normalization=norm, act=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )   # 64 - 128 - 128
        self.layer3 = nn.Sequential(
            *discriminator_block2(mid_channels, mid_channels, normalization=norm, act=True),
            *discriminator_block2(mid_channels, mid_channels, normalization=norm, act=True),
            nn.AvgPool2d(kernel_size=4, stride=4)
        )   # 128 - 256 - 256
        self.layer4 = nn.Sequential(
            *discriminator_block2(mid_channels, mid_channels, normalization=norm, act=True),
            *discriminator_block2(mid_channels, mid_channels, normalization=norm, act=True),
            nn.AvgPool2d(kernel_size=4, stride=4)
        )
        self.layer5 = nn.Sequential(
            *discriminator_block2(mid_channels, mid_channels, normalization=norm, act=True),
            *discriminator_block2(mid_channels, mid_channels, normalization=norm, act=True),
            nn.AdaptiveAvgPool2d(output_size=(1,1))
        )
        self.conv = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5,
        )
        self.fc = nn.Linear(mid_channels,out_channels)
            
        

    def forward(self, img_input):
        lr = F.interpolate(img_input, size=(256,256), mode='nearest')
        out = self.conv(lr)
        out = out.view(img_input.shape[0],-1)
        out = self.fc(out)
        return out

# # model = VGG(3,64,64,norm=True).cuda().eval()
# model = Model_10L96C(3,64,64).cuda().eval()
# # # VGG(3,64,64,norm=True).cuda().eval()
# # # import netron
# data = torch.rand(1,3,1024,1024).cuda()
# # # torch.onnx.export(model,data,'model.onnx',export_params=True,opset_version=16)

# # # netron.start('model.onnx')
# from thop import profile
# macs, params = profile(model, inputs=(data,))
# print(macs, params)

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
            nn.Dropout(p=0.5),
            nn.Conv2d(128, lut, 8, padding=0),
        )
        self.lr = nn.Linear(lut, 1)

    def forward(self, img_input):
        out = self.model(img_input)
        out1 = out.view(img_input.shape[0], -1)
        out2 = self.lr(out1)
        return out, out2

class Generator3DLUT_hr_lut(nn.Module):
    def __init__(self, dim=33, filename='model/innocolor/lut3d_rgb.txt'):
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
            file = open("IdentityLUT9.txt", 'r')
        elif dim == 17:
            file = open("model/innocolor/IdentityLUT17.txt", 'r')
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



