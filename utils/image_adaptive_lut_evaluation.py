import argparse
import os
import time
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable

from models_x import *
from datasets import *

os.chdir('/home/brocolimanx/Desktop/Image-Adaptive-3DLUT-master')

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=2000, help="epoch to load the saved checkpoint")
parser.add_argument("--dim", type=int, default=17, help="the dimension of the lut")
parser.add_argument("--dataset_name", type=str, default="colourblindness", help="name of the dataset")
parser.add_argument("--input_color_space", type=str, default="sRGB", help="input color space: sRGB or XYZ")
parser.add_argument("--model_dir", type=str, default="1015", help="directory of saved models")
parser.add_argument("--outer_dir", type=str, default="./saved_models/%s", help="path to save model")
opt = parser.parse_args()
opt.model_dir = opt.model_dir

############################## Setup things ##############################

with open('threshold.txt') as thres_file:
    threshold = float(thres_file.read())

# Loss
criterion_pixelwise = torch.nn.MSELoss()

# Models
LUT0 = Generator3DLUT_hr_lut(dim=opt.dim)
LUT1 = Generator3DLUT_identity(dim=opt.dim)
LUT2 = Generator3DLUT_identity(dim=opt.dim)
LUT3 = Generator3DLUT_identity(dim=opt.dim)
LUT4 = Generator3DLUT_identity(dim=opt.dim)
classifier = Classifier_class2(lut=5)

# CUDA or not settings
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
if torch.cuda.is_available():
    LUT0 = LUT0.cuda()
    LUT1 = LUT1.cuda()
    LUT2 = LUT2.cuda()
    LUT3 = LUT3.cuda()
    LUT4 = LUT4.cuda()
    classifier = classifier.cuda()
    criterion_pixelwise.cuda()

# Load pretrained models
LUTs = torch.load((opt.outer_dir+"/LUTs_%d.pth") % (opt.model_dir, opt.epoch), map_location='cuda')
LUT0.load_state_dict(LUTs["0"])
LUT1.load_state_dict(LUTs["1"])
LUT2.load_state_dict(LUTs["2"])
LUT3.load_state_dict(LUTs["3"])
LUT4.load_state_dict(LUTs["4"])
LUT0.eval()
LUT1.eval()
LUT2.eval()
LUT3.eval()
LUT4.eval()
classifier.load_state_dict(torch.load((opt.outer_dir+"/classifier_%d.pth") % (opt.model_dir, opt.epoch), map_location='cuda'))
classifier.eval()

if opt.input_color_space == 'sRGB':
    dataloader = DataLoader(
        ImageDataset_sRGB(r"./data/data/"+opt.dataset_name,  mode="test"),
        batch_size=1,
        shuffle=False,
        # num_workers=1,
    )
elif opt.input_color_space == 'XYZ':
    assert ValueError("Must be sRGB")

############################## Evaluation functions ##############################

def generator(img, bin_class=True):
    # generated fused LUT
    img = img.cuda()
    pred, out2 = classifier(img)
    pred = pred.squeeze()
    if out2 < threshold and bin_class:
        LUT = LUT0.LUT
    else:
        LUT = pred[0] * LUT0.LUT + pred[1] * LUT1.LUT + pred[2] * LUT2.LUT + pred[3] * LUT3.LUT + pred[4] * LUT4.LUT
    # Clip (for camera cases)
    LUT = torch.clip(LUT, 0, 1)
    # apply LUT
    img = (img - .5) * 2.
    img = img.permute(0, 2, 3, 1)[:, None]
    LUT = LUT[None]
    result = F.grid_sample(LUT, img, mode='bilinear', padding_mode='border', align_corners=True)
    combine_A = result[:, :, 0]
    
    return combine_A, out2, pred

def generator_wrapper(img, bin_class=True):
    return generator(img, bin_class)[0]

############################## Evaluation ############################## 


#test_speed()
