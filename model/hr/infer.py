import argparse
import os
import time
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable

from model.daltnet.models_x import *
from model.daltnet.datasets import *

dim=17
epoch = 190
outer_dir = "checkpoint"
model_dir = "hr"
input_color_space = "sRGB"
threshold = 0.3

############################## Setup things ##############################

# Models
LUT0 = Generator3DLUT_hr_lut(dim=dim)

# CUDA or not settings
# Determine the available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move all models and LUTs to the correct device
LUT0 = LUT0.to(device)

LUT0.eval()


############################## Evaluation functions ##############################

# Update function to ensure all tensors match
def generator(img, bin_class=True):
    img = img.to(device)  # Move input to same device as model
    LUT = torch.clip(LUT0.LUT, 0, 1)

    # Apply LUT
    img = (img - .5) * 2.
    img = img.permute(0, 2, 3, 1)[:, None]
    LUT = LUT[None].to(device)  # Ensure LUT is also on same device

    result = F.grid_sample(LUT, img, mode='bilinear', padding_mode='border', align_corners=True)
    combine_A = result[:, :, 0]

    return combine_A

