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
model_dir = "daltnet"
input_color_space = "sRGB"
threshold = 0.3

############################## Setup things ##############################


# Loss
criterion_pixelwise = torch.nn.MSELoss()

# Models
LUT0 = Generator3DLUT_hr_lut(dim=dim)
LUT1 = Generator3DLUT_identity(dim=dim)
LUT2 = Generator3DLUT_identity(dim=dim)
LUT3 = Generator3DLUT_identity(dim=dim)
LUT4 = Generator3DLUT_identity(dim=dim)
classifier = Classifier_class2(lut=5)

# CUDA or not settings
# Determine the available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move all models and LUTs to the correct device
LUT0 = LUT0.to(device)
LUT1 = LUT1.to(device)
LUT2 = LUT2.to(device)
LUT3 = LUT3.to(device)
LUT4 = LUT4.to(device)
classifier = classifier.to(device)
criterion_pixelwise.to(device)

# Ensure all LUTs are on the correct device
LUTs = torch.load(os.path.join(outer_dir, model_dir, f"LUTs_{epoch}.pth"), map_location=device, weights_only=True)
classifier.load_state_dict(torch.load(os.path.join(outer_dir, model_dir, f"classifier_{epoch}.pth"), map_location=device, weights_only=True))


############################## Evaluation functions ##############################

# Update function to ensure all tensors match
def generator(img, bin_class=True):
    img = img.to(device)  # Move input to same device as model

    pred, out2 = classifier(img)  # Ensure model processes correctly
    pred = pred.squeeze()

    if out2 < threshold and bin_class:
        LUT = LUT0.LUT
    else:
        LUT = pred[0] * LUT0.LUT + pred[1] * LUT1.LUT + pred[2] * LUT2.LUT + pred[3] * LUT3.LUT + pred[4] * LUT4.LUT

    LUT = torch.clip(LUT, 0, 1)

    # Apply LUT
    img = (img - .5) * 2.
    img = img.permute(0, 2, 3, 1)[:, None]
    LUT = LUT[None].to(device)  # Ensure LUT is also on same device

    result = F.grid_sample(LUT, img, mode='bilinear', padding_mode='border', align_corners=True)
    combine_A = result[:, :, 0]

    return combine_A



############################## Evaluation ############################## 


#test_speed()
