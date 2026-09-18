import numpy as np

# dim=17
# if dim == 9:
#     file = open("IdentityLUT9.txt", 'r')
# elif dim == 17:
#     file = open("Identity_2DLUT17.txt", 'r')
# lines = file.readlines()
# buffer = np.zeros((2,dim,dim), dtype=np.float32)


# for j in range(0,dim):
#     for k in range(0,dim):
#         n = j * dim + k
#         x = lines[n].split()
#         buffer[0,j,k] = float(x[0])
#         buffer[1,j,k] = float(x[1])

# print(buffer[:,0,0,0])
# print(buffer[:,0,0,1])
# print(buffer[:,0,0,2])
import torch
import cv2
import torch.nn.functional as F
from torchvision.utils import save_image
# lut_path = 'save_result/2d_lut_new/ckpt/Color_001392.pth'
# lut = torch.load(lut_path)
# # print(lut.shape)

# with open("save_result/2d_lut_new/ckpt/Color_001392.txt", 'w') as f:
#     for i in range(17):
#         for j in range(17):
#             f.write('{:.6f}  {:.6f}\n'.format(
#                     lut[0,0,i,j].item(), lut[0,1,i,j].item()))
#             print(lut[0,0,i,j].item(), lut[0,1,i,j].item())
# f = open('save_result/2d_lut_new/ckpt/Color_001392.txt','a')
# for i in range(17):
#     for j in range(17):
#         f.write('{:.6f}  {:.6f}\n'.format(
#                 lut[0,0,i,j].item(), lut[0,1,i,j].item()))
#         print(lut[0,0,i,j].item(), lut[0,1,i,j].item())
# for i in range(0,lut.shape[1]):
#     for j in range(0,lut.shape[2]):
#         for k in range(0,lut.shape[3]):
#             f.write("%f\n"%lut[0,i,j,k].cpu().numpy())
# import torch
# img = cv2.imread("/path/to/save_result/input/Color_001392.bmp")
# img = img[:,:,::-1]
# img = torch.from_numpy(img.copy()).permute(2,0,1).unsqueeze(0)
# img = img.float() / 255.
# print(torch.min(img), torch.max(img))
# from utils.color_torch import *
# img_yuv = rgb2yuv_bt709(img).cuda()
# def lut_2d(LUT, x):
#     x = (x - .5) * 2.
#     x = x.permute(0, 2, 3, 1)
#     # LUT = LUT.reshape(LUT.shape[0], 2, 17, dim)
#     result = F.grid_sample(LUT, x, mode='bilinear', padding_mode='border', align_corners=True)
#     return result
# out_uv = lut_2d(lut, img_yuv[:,1:,:,:])
# out_yuv = torch.cat((img_yuv[:,0:1,:,:], out_uv),dim=1)
# out = yuv2rgb_bt709(out_yuv).clip(0,1)
# from torchvision.utils import save_image
# save_image(out, "/path/to/save_result/gt/Color_001392.bmp",normalize=True)

# lut_path = '/path/to/save_result/yuv/Color_001392.pth'
# lut = torch.load(lut_path)
# # print(lut.shape)

# with open("save_result/2d_lut_new/demo/yuv_Color_001392.txt", 'w') as f:
#     for i in range(256):
#         for j in range(256):
#             f.write('{:.6f}  {:.6f}  {:.6f}\n'.format(
#                     lut[0,0,i,j].item() * 255, lut[0,1,i,j].item() * 255, lut[0,2,i,j].item() * 255))
            # print(lut[0,0,i,j].item(), lut[0,1,i,j].item())

# -----------------------------------compare result of c and pytorch-----------------------------------------------
# f1 = open('lut/rgb.txt', 'r')
# lines1 = f1.readlines()
# f2 = open('save_result/2d_lut_new/demo/out_rgb_Color_001392.txt', 'r')
# lines2 = f2.readlines()
# demo = 0
# for n in range(256*256):
#     x = lines1[n].split()
#     y = lines2[n].split()
#     if abs(float(x[0])-float(y[0])) > 0.5 or abs(float(x[1])-float(y[1])) > 0.5 or abs(float(x[2])-float(y[2])) > 0.5:
#         # print(x)
#         # print(y)
        
#         tmp = max(abs(float(x[0])-float(y[0])), abs(float(x[1])-float(y[1])), abs(float(x[2])-float(y[2])))
#         if tmp > demo:
#             demo = tmp
# print(demo)
# f1.close()
# f2.close()

#--------------------------------------C rgb_lut to torch--------------------------------------------------
f1 = open('lut/rgb.txt', 'r')
lines1 = f1.readlines()
x = torch.rand(1,3,256,256)
for i in range(256):
    for j in range(256):
        list_rgb = lines1[i*256+j].split()
        x[:,0,i,j] = float(list_rgb[0]) / 255.
        x[:,1,i,j] = float(list_rgb[1]) / 255.
        x[:,2,i,j] = float(list_rgb[2]) / 255.
x = x.clip(0,1)
save_image(x, "/path/to/save_result/gt/Color_001392.png",normalize=True)
