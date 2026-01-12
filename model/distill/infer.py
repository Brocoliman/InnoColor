import os
import sys
from settings import *
os.chdir(FOLDER)

# Add the parent directory to sys.path
sys.path.append(FOLDER)

import argparse
import os
import time
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from model.distill.color_torch import *
from torchvision import models
import torchvision.transforms.functional as TF
import torch.nn.functional as F

import matplotlib.pyplot as plt
import cv2
import numpy as np
dim=17
epoch = 100
outer_dir = "checkpoint"
model_dir = "distill"
input_color_space = "sRGB"


############################## Setup things ##############################


# Loss
criterion_pixelwise = torch.nn.MSELoss()

# Models
model = models.resnet18(pretrained=False, num_classes=17*17*2)

# CUDA or not settings
# Determine the available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# Move all models and LUTs to the correct device
ckpt_dp = torch.load(os.path.join(outer_dir, model_dir, f"generator_{epoch}.pth"), map_location=device, weights_only=True)
ckpt = {}
for idx, key in enumerate(ckpt_dp):
    ckpt[key.replace('module.', '')] = ckpt_dp[key]

model.load_state_dict(ckpt)
model.eval()

############################## Evaluation functions ##############################

# Update function to ensure all tensors match
def lut_2d(LUT,x):
    x = (x - 0.5) * 2.
    x = x.permute(0, 2, 3, 1)
    LUT = LUT.reshape(LUT.shape[0],2,17,17)
    result = F.grid_sample(LUT, x, mode='bilinear', padding_mode='border', align_corners=True)
    return result

def generator(img, bin_class=True):
    img = img.to(device)  # Move input to same device as model
    img_resize = TF.resize(img.clip(0,1), (256,256))
    img_yuv = rgb2yuv_bt709(img)
    img_resize_yuv = rgb2yuv_bt709(img_resize)
    LUT = model(img_resize_yuv)
    uv = lut_2d(LUT, img_yuv[:,1:,:,:])
    result_yuv = torch.cat((img_yuv[:,0:1,:,:], uv), dim=1)
    return yuv2rgb_bt709(result_yuv)

def tensor_to_numpy(tensor, normalize=True):
    # Ensure tensor is on CPU and detached
    tensor = tensor.cpu().detach()
    
    # Normalize if requested
    if normalize:
        tensor = tensor.clamp(0, 1)  # Ensure values are in [0,1]
    
    # Check the number of dimensions and handle accordingly
    if len(tensor.shape) == 4:  # Shape: (batch, channels, height, width)
        return tensor.permute(0, 2, 3, 1)[0].numpy()
    elif len(tensor.shape) == 3:  # Shape: (channels, height, width)
        return tensor.permute(1, 2, 0).numpy()
    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}. Expected 3D or 4D tensor.")

def visualize_and_save_components(input_img, uv_tensor, result_rgb, save_path):
    # Load and preprocess the input image
    input_img = cv2.imread(input_img)
    if input_img is None:
        raise ValueError(f"Failed to load image at {input_img}. Check the file path and format.")
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB) / 255.0  # Normalize to [0,1]
    input_img = torch.from_numpy(input_img).permute(2, 0, 1).float().unsqueeze(0)  # Shape: (1, C, H, W)

    # Convert input RGB to YUV for component extraction
    input_yuv = rgb2yuv_bt709(input_img)
    Y, U, V = input_yuv[0, 0], input_yuv[0, 1], input_yuv[0, 2]  # Extract Y, U, V channels

    # Extract U, V from uv_tensor
    uv_U, uv_V = uv_tensor[0, 0], uv_tensor[0, 1]  # Assuming uv_tensor has shape (1, 2, H, W)

    # Convert tensors to numpy for visualization
    Y_np = tensor_to_numpy(Y.unsqueeze(0), normalize=True).squeeze()  # Ensure 2D (H, W) for grayscale
    U_np = tensor_to_numpy(U.unsqueeze(0), normalize=True).squeeze()  # Ensure 2D (H, W) for grayscale
    V_np = tensor_to_numpy(V.unsqueeze(0), normalize=True).squeeze()  # Ensure 2D (H, W) for grayscale
    uv_U_np = tensor_to_numpy(uv_U.unsqueeze(0), normalize=True).squeeze()  # Ensure 2D (H, W) for grayscale
    uv_V_np = tensor_to_numpy(uv_V.unsqueeze(0), normalize=True).squeeze()  # Ensure 2D (H, W) for grayscale
    result_rgb_np = tensor_to_numpy(result_rgb, normalize=True)  # Should be (H, W, 3) for RGB

    # Create a figure with no padding or spacing between subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), gridspec_kw={'wspace': 0, 'hspace': 0})
    
    # Hide all axes, titles, and labels
    for ax in axes.flat:
        ax.axis('off')

    # Plot the images without titles or text
    axes[0, 0].imshow(Y_np, cmap='gray')  # Input Y - Grayscale
    axes[0, 1].imshow(U_np, cmap='viridis')  # Input U - Viridis
    axes[0, 2].imshow(V_np, cmap='inferno')  # Input V - Inferno
    axes[1, 0].imshow(uv_U_np, cmap='viridis')  # UV U - Viridis
    axes[1, 1].imshow(uv_V_np, cmap='inferno')  # UV V - Inferno
    axes[1, 2].imshow(result_rgb_np)  # Final RGB Result

    # Adjust layout to remove any padding or spacing
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)

    # Save the combined image without text or spacing
    plt.savefig(f'{save_path}/yuv_decomposition.png', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

