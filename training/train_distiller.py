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
import torchvision.models as models
from utils.color_torch import rgb2yuv_bt709, yuv2rgb_bt709
#from gan_cnn import *
from datasets import *
from discrimintor_trs import *
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import nn as nn
from cvd_function import *
from contrast import *

from torch.utils.tensorboard import SummaryWriter
from kornia.color import rgb_to_lab


class SSIMLoss(nn.Module):
    def __init__(self, kernel_size: int = 11, sigma: float = 1.5) -> None:

        """Computes the structural similarity (SSIM) index map between two images.

        Args:
            kernel_size (int): Height and width of the gaussian kernel.
            sigma (float): Gaussian standard deviation in the x and y direction.
        """

        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.gaussian_kernel = self._create_gaussian_kernel(self.kernel_size, self.sigma)

    def forward(self, x: torch.Tensor, y: torch.Tensor, as_loss: bool = True) -> torch.Tensor:

        if not self.gaussian_kernel.is_cuda:
            self.gaussian_kernel = self.gaussian_kernel.to(x.device)

        ssim_map = self._ssim(x, y)

        if as_loss:
            return 1 - ssim_map.mean()
        else:
            return ssim_map

    def _ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        # Compute means
        ux = F.conv2d(x, self.gaussian_kernel, padding=self.kernel_size // 2, groups=3)
        uy = F.conv2d(y, self.gaussian_kernel, padding=self.kernel_size // 2, groups=3)

        # Compute variances
        uxx = F.conv2d(x * x, self.gaussian_kernel, padding=self.kernel_size // 2, groups=3)
        uyy = F.conv2d(y * y, self.gaussian_kernel, padding=self.kernel_size // 2, groups=3)
        uxy = F.conv2d(x * y, self.gaussian_kernel, padding=self.kernel_size // 2, groups=3)
        vx = uxx - ux * ux
        vy = uyy - uy * uy
        vxy = uxy - ux * uy

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        numerator = (2 * ux * uy + c1) * (2 * vxy + c2)
        denominator = (ux ** 2 + uy ** 2 + c1) * (vx + vy + c2)
        return numerator / (denominator + 1e-12)

    def _create_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:

        start = (1 - kernel_size) / 2
        end = (1 + kernel_size) / 2
        kernel_1d = torch.arange(start, end, step=1, dtype=torch.float)
        kernel_1d = torch.exp(-torch.pow(kernel_1d / sigma, 2) / 2)
        kernel_1d = (kernel_1d / kernel_1d.sum()).unsqueeze(dim=0)

        kernel_2d = torch.matmul(kernel_1d.t(), kernel_1d)
        kernel_2d = kernel_2d.expand(3, 1, kernel_size, kernel_size).contiguous()
        return kernel_2d



parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="facades", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--dim", type=int, default=17, help="dim of 3d-lut")
parser.add_argument(
    "--sample_interval", type=int, default=5000, help="interval between sampling of images from generators"
)
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")

parser.add_argument("--cvd", type=int, default=1, help="P 0, D 1")
parser.add_argument("--ori", type=int, default=1, help="naturalness calculate between fake and original")
parser.add_argument("--ssimori", type=int, default=1, help="ssim calculate between fake and original")
parser.add_argument("--lambda_ssim", type=float, default=0.25, help="the weightness of ssim")
parser.add_argument("--points", type=int, default=0.00, help="the weightness of ssim")


opt = parser.parse_args()
print(opt)
device = torch.device("cuda")
cuda = True if torch.cuda.is_available() else False
# print(cuda)
# exit()
# Loss functions



criterion_contrast = torch.nn.L1Loss()


# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 100
lambda_contrast = 1
lambda_global = 1
lambda_ssim = opt.lambda_ssim


# Calculate output of image discriminator (PatchGAN)
# patch = (1, opt.img_height // 2 ** 5, opt.img_width // 2 ** 5)
# patch = (1,)
# patch = (1, opt.img_height // 2 ** 5, opt.img_width // 2 ** 5)
patch = (opt.img_height // 2 ** 5 * opt.img_width // 2 ** 5, 1)
# Initialize generator and discriminator
global_points = opt.points
CVD_type = opt.cvd
dis_index = 0
if CVD_type == 1:
    dis_index = 1
else:
    dis_index = 0
# word_suffix = [
# '_P_%d_ori_%s_ssimori_%s_ssim_%s_dis_dim_%s_finetune'%(opt.points,opt.ori,opt.ssimori,opt.lambda_ssim, opt.dim),
# '_D_%d_ori_%s_ssimori_%s_ssim_%s_dis_dim_%s_finetune'%(opt.points,opt.ori,opt.ssimori,opt.lambda_ssim, opt.dim),
# #'_D_labonlyG+nature_points%d_COLOR_patch4_844_48_3_nouplayer_norma_%s_ori_%s__ssimori_%s_n_%s_ssim_%s'%(opt.points,opt.norma,opt.ori,opt.ssimori,opt.lambda_nature,opt.lambda_ssim,)
# ]
word_suffix = [
'_P_%d_ori_%s_ssimori_%s_ssim_%s_dis_dim_%s_2dlut_shufflenet_v2'%(opt.points,opt.ori,opt.ssimori,opt.lambda_ssim, opt.dim),
'_D_%d_ori_%s_ssimori_%s_ssim_%s_dis_dim_%s_2dlut_shufflenet_v2'%(opt.points,opt.ori,opt.ssimori,opt.lambda_ssim, opt.dim),
#'_D_labonlyG+nature_points%d_COLOR_patch4_844_48_3_nouplayer_norma_%s_ori_%s__ssimori_%s_n_%s_ssim_%s'%(opt.points,opt.norma,opt.ori,opt.ssimori,opt.lambda_nature,opt.lambda_ssim,)
]

print(word_suffix)

# generator = models.resnet18(pretrained=False)
# generator.fc = torch.nn.Linear(512, opt.dim * opt.dim * 2)
# generator = models.mobilenet_v3_large(pretrained=False, num_classes=opt.dim * opt.dim * 2)
generator = models.shufflenet_v2_x1_0(pretrained=False, num_classes=opt.dim * opt.dim * 2)
# generator = models.squeezenet1_1(pretrained=False, num_classes=opt.dim * opt.dim * 2)
# generator = SwinTransformer(num_classes=opt.dim * opt.dim * 2)
print(generator)


print(len(word_suffix),dis_index,word_suffix[dis_index])
os.makedirs("images/ssim/%s" % opt.dataset_name + word_suffix[dis_index], exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name + word_suffix[dis_index], exist_ok=True)


# CUDA setup
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
TV2_LUT = TV_2D_LUT(dim=opt.dim)
if cuda:
    generator = nn.DataParallel(generator).to(device)
    # ckpt_dp = torch.load("saved_models/cvd_100_001_D_3000_ori_1_ssimori_1_ssim_0.25_dis_dim_17/generator_100.pth")
    # generator.load_state_dict(ckpt_dp)
    criterion_contrast.cuda()
    # criterion_contrast.to(device)
    # criterion_contrast.cuda()
    print("cuda!!!!!!!")
    TV2_LUT.cuda()
    TV2_LUT.weight_u = TV2_LUT.weight_u.type(Tensor)
    TV2_LUT.weight_v = TV2_LUT.weight_v.type(Tensor)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Configure dataloaders
transforms_ = [
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

# dataloader = DataLoader(
#     ImageDataset_single("/public/CHEN/Nature_images", transforms_=transforms_, mode ="None"),
#     batch_size=opt.batch_size,
#     shuffle=True,
#     num_workers=8,
# )

if CVD_type == 1:
    data_path = "/path/to/CVDdataset/Color_cvd_D_experiment_100000"
else:
    data_path = "/path/to/CVDdataset/Color_cvd_P_experiment_100000"

dataloader = DataLoader(
    ImageDataset_distiller(data_path, transforms_=transforms_, mode ="None"),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=12,
)
# val_dataloader = DataLoader(
#     ImageDataset_single("../Code_dataset/%s" % opt.dataset_name, transforms_=transforms_, mode="val"),
#     batch_size=10,
#     shuffle=True,
#     num_workers=1,
# )
val_dataloader = DataLoader(
    ImageDataset_single(data_path, transforms_=transforms_, mode ="None"),
    batch_size=20,
    shuffle=True,
    num_workers=16,
)
transform_val_list1 = [transforms.Normalize((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0))]
transform_val_list2 = [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
transform_val_list3 = [transforms.Normalize((0, 0, 0), (100, 128, 128))]

trans_compose1 = transforms.Compose(transform_val_list1)
trans_compose2 = transforms.Compose(transform_val_list2)
trans_compose3 = transforms.Compose(transform_val_list3)
# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
print(Tensor)
def lut_2d(LUT, x):
    x = (x - .5) * 2.
    x = x.permute(0, 2, 3, 1)
    
    result = F.grid_sample(LUT, x, mode='bilinear', padding_mode='border', align_corners=True)
    return result

def generator_train(img):
    img_yuv = rgb2yuv_bt709(img)
    LUT = generator(img_yuv)
    LUT = LUT.reshape(LUT.shape[0], 2, opt.dim, opt.dim)
    uv = lut_2d(LUT, img_yuv[:,1:,:,:])
    out_yuv = torch.cat((img_yuv[:,0:1,:,:],uv), dim=1)
    out = yuv2rgb_bt709(out_yuv).clip(0,1)
    return out, LUT


# ----------
#  Training
# ----------

prev_time = time.time()
D_Loss, G_Loss = [], []

writer = SummaryWriter("tensorboard/%s" % opt.dataset_name + word_suffix[dis_index], flush_secs=30)

ssimloss_funtion = SSIMLoss(kernel_size=11)

for epoch in range(opt.epoch+1, opt.n_epochs):
    Epochs = range(0, epoch + 1 - opt.epoch)
    D_losses = []
    G_losses = []
    nature_loss = []
    local_loss = []
    global_loss = []
    ssim_loss = []
    nature_weight = []

    # if epoch > 100:
    #     dataloader = dataloader_later
    for i, batch in enumerate(dataloader):

        real_A = Variable(batch['input'].type(Tensor))
        gt = Variable(batch['gt'].type(Tensor))
        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()
        # print(torch.max(real_A), torch.min(real_A),torch.max(gt), torch.min(gt))
        # GAN loss
        fake_B, LUT = generator_train(real_A)
        # fake_B = trans_compose2(fake_B)
        loss_distiller = criterion_contrast(fake_B, gt)
        loss_tv = 0
        loss_mn = 0
        for b in range(real_A.shape[0]):

            tv, mn = TV2_LUT(LUT[b, :, :, :])
            loss_tv += tv
            loss_mn += mn
        loss_G = loss_distiller + loss_tv * 0.0001 + loss_mn * 10



        loss_G.backward()

        optimizer_G.step()


        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        # D_losses.append(loss_D.item())
        G_losses.append(loss_G.item())
        # d_l, g_l = np.array(D_losses).mean(), np.array(G_losses).mean()
        g_l = np.array(G_losses).mean()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d]  [G: %f,distiller: %f,tv: %f,mn: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_G.item(),
                loss_distiller.item(),
                loss_tv.item() *  0.0001,
                loss_mn.item() * 10,
                time_left,
            )
        )
        # print(len(dataloader), 22222)
        # If at sample interval save image
        # if batches_done % opt.sample_interval == 0:
        #     sample_images(batches_done)

    G_Loss.append(g_l)
    # print(min(nature_loss),max(nature_loss),"\n")
    # print(min(local_loss), max(local_loss), "\n")
    # print(min(global_loss), max(global_loss), "\n")

    writer.add_scalars('loss/G_D_loss', {"G_loss": g_l,
                                         }, epoch)

    writer.add_scalars('loss/Generator_loss', {"loss_dis": loss_distiller.item(),
                                               "loss_tv": loss_tv.item() * 0.0001,
                                                "loss_mn": loss_mn.item() * 10
                                               }, epoch)


    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(),
                   "saved_models/%s/generator_%d.pth" % (opt.dataset_name + word_suffix[dis_index], epoch))  

writer.close()