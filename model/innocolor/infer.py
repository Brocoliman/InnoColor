import os
import torch

from model.daltnet.datasets import *
from model.innocolor.color_torch import *
from model.innocolor.models_x import *
from model.innocolor.SAM2UNet import SAM2UNet
dim=17
epoch = 2000
outer_dir = "checkpoint"
model_dir = "innocolor"
input_color_space = "sRGB"
threshold = 0.3

############################## Setup things ##############################


# Loss
criterion_pixelwise = torch.nn.MSELoss()
SOD = SAM2UNet()
# Models
LUT0 = Generator3DLUT_hr_lut(dim=dim)
LUT1 = Generator3DLUT_identity(dim=dim)
LUT2 = Generator3DLUT_identity(dim=dim)
LUT3 = Generator3DLUT_identity(dim=dim)
LUT4 = Generator3DLUT_identity(dim=dim)
classifier = Classifier_class2(lut=5)
RAnet = Model_10L64C(4,64,1,norm=True)
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
SOD = SOD.to(device)
RAnet = RAnet.to(device)

# Ensure all LUTs are on the correct device
LUTs = torch.load(os.path.join(outer_dir, model_dir, f"LUTs_{epoch}.pth"), map_location=device)
classifier.load_state_dict(torch.load(os.path.join(outer_dir, model_dir, f"classifier_{epoch}.pth"), map_location=device))
classifier.eval()
ckpt_dp = torch.load(os.path.join(outer_dir, model_dir, f"generator_{200}.pth"), map_location=device)
ckpt = {}
for idx, key in enumerate(ckpt_dp):
    ckpt[key.replace('module.', '')] = ckpt_dp[key]
RAnet.load_state_dict(ckpt)
RAnet.eval()
SOD.load_state_dict(torch.load(os.path.join(outer_dir, model_dir, f"SAM2UNet-SOD.pth"), map_location=device))
SOD.eval()

############################## Evaluation functions ##############################

# Update function to ensure all tensors match
def generator(img, bin_class=True):
    img = img.to(device)  # Move input to same device as model
    img_size = img[0, 0, :, :].shape  # Extract shape as (height, width)
    img_resize = TF.resize(img.clip(0, 1), (352, 352))
    img_resize = TF.normalize(img_resize, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    Mask, _, _ = SOD(img_resize)
    Mask = F.upsample(Mask, size=(int(img_size[0]), int(img_size[1])), mode='bilinear', align_corners=True)
    Mask = Mask.sigmoid().data
    Mask[Mask > 0.5] = 1
    Mask[Mask <= 0.5] = 0

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
    combine_A_hsv = rgb2hsv_torch(combine_A)
    hue_rotate = RAnet(torch.cat([combine_A_hsv, Mask], dim=1)).unsqueeze(2).unsqueeze(2)
    hue_new = combine_A_hsv[:, 0:1, :, :] + hue_rotate
    hue_new = torch.where(hue_new < 0., hue_new + 1, torch.where(hue_new > 1., hue_new - 1, hue_new))
    result_hsv = torch.cat([hue_new, combine_A_hsv[:, 1:2, :, :], combine_A_hsv[:, 2:3, :, :]], dim=1)
    result_rgb = hsv2rgb_torch(result_hsv)
    result = combine_A * (1 - Mask) + result_rgb * Mask
    return result

############################## Evaluation ############################## 


#test_speed()
