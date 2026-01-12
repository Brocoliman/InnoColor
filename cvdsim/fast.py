from PIL import Image
import numpy as np
import pandas as pd
import sys, os, csv
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch
import time


#################### Setup
# Input
dim = 17

use_cuda = torch.cuda.is_available()

# Current working lut
lut = None

# Matrix (11x3x3)
m_match = np.array([i*0.1 for i in range(0, 11)])
deut = np.array(
    [
        [
            [1.000000,	0.000000,	-0.000000],
            [0.000000,	1.000000,	0.000000], 
            [-0.000000,	-0.000000,	1.000000],
        ],
        [
            [0.866435,	0.177704,	-0.044139],
[0.049567,	0.939063,	0.011370],
[-0.003453,	0.007233,	0.996220],
        ],
        [[0.760729,	0.319078,	-0.079807],
[0.090568,	0.889315,	0.020117],
[-0.006027,	0.013325,	0.992702],],
        [[0.675425,	0.433850,	-0.109275],
[0.125303,	0.847755,	0.026942],
[-0.007950,	0.018572,	0.989378]	,],
        [[0.605511,	0.528560,	-0.134071],
[0.155318,	0.812366,	0.032316],
[-0.009376,	0.023176,	0.986200],],
        [[0.547494,	0.607765,	-0.155259],
[0.181692,	0.781742,	0.036566],
[-0.010410,	0.027275,	0.983136],],
        [[0.498864,	0.674741,	-0.173604],
[0.205199,	0.754872,	0.039929],
[-0.011131,	0.030969,	0.980162],],
        [[0.457771,	0.731899,	-0.189670],
[0.226409,	0.731012,	0.042579],
[-0.011595,	0.034333,	0.977261],],
        [[0.422823,	0.781057,	-0.203881],
[0.245752,	0.709602,	0.044646],
[-0.011843,	0.037423,	0.974421],],
        [[0.392952,	0.823610,	-0.216562],
[0.263559,	0.690210,	0.046232],
[-0.011910,	0.040281,	0.971630],],
        [[0.367322,	0.860646,	-0.227968],
[0.280085,	0.672501,	0.047413],
[-0.011820,	0.042940,	0.968881],]
    ]
)
prot = np.array(
        [
            [[1.0, 0.0, -0.0], [0.0, 1.0, 0.0], [-0.0, -0.0, 1.0]],
            [
                [0.856167, 0.182038, -0.038205],
                [0.029342, 0.955115, 0.015544],
                [-0.00288, -0.001563, 1.004443],
            ],
            [
                [0.734766, 0.334872, -0.069637],
                [0.05184, 0.919198, 0.028963],
                [-0.004928, -0.004209, 1.009137],
            ],
            [
                [0.630323, 0.465641, -0.095964],
                [0.069181, 0.890046, 0.040773],
                [-0.006308, -0.007724, 1.014032],
            ],
            [
                [0.539009, 0.579343, -0.118352],
                [0.082546, 0.866121, 0.051332],
                [-0.007136, -0.011959, 1.019095],
            ],
            [
                [0.458064, 0.679578, -0.137642],
                [0.092785, 0.846313, 0.060902],
                [-0.007494, -0.016807, 1.024301],
            ],
            [
                [0.38545, 0.769005, -0.154455],
                [0.100526, 0.829802, 0.069673],
                [-0.007442, -0.02219, 1.029632],
            ],
            [
                [0.319627, 0.849633, -0.169261],
                [0.106241, 0.815969, 0.07779],
                [-0.007025, -0.028051, 1.035076],
            ],
            [
                [0.259411, 0.923008, -0.18242],
                [0.110296, 0.80434, 0.085364],
                [-0.006276, -0.034346, 1.040622],
            ],
            [
                [0.203876, 0.990338, -0.194214],
                [0.112975, 0.794542, 0.092483],
                [-0.005222, -0.041043, 1.046265],
            ],
            [
                [0.152286, 1.052583, -0.204868],
                [0.114503, 0.786281, 0.099216],
                [-0.003882, -0.048116, 1.051998],
            ],
        ]
)
trit = np.array(
        [
            [[1.0, 0.0, -0.0], [0.0, 1.0, 0.0], [-0.0, -0.0, 1.0]],
            [
                [0.92667, 0.092514, -0.019184],
                [0.021191, 0.964503, 0.014306],
                [0.008437, 0.054813, 0.93675],
            ],
            [
                [0.89572, 0.13333, -0.02905],
                [0.029997, 0.9454, 0.024603],
                [0.013027, 0.104707, 0.882266],
            ],
            [
                [0.905871, 0.127791, -0.033662],
                [0.026856, 0.941251, 0.031893],
                [0.01341, 0.148296, 0.838294],
            ],
            [
                [0.948035, 0.08949, -0.037526],
                [0.014364, 0.946792, 0.038844],
                [0.010853, 0.193991, 0.795156],
            ],
            [
                [1.017277, 0.027029, -0.044306],
                [-0.006113, 0.958479, 0.047634],
                [0.006379, 0.248708, 0.744913],
            ],
            [
                [1.104996, -0.046633, -0.058363],
                [-0.032137, 0.971635, 0.060503],
                [0.001336, 0.317922, 0.680742],
            ],
            [
                [1.193214, -0.109812, -0.083402],
                [-0.058496, 0.97941, 0.079086],
                [-0.002346, 0.403492, 0.598854],
            ],
            [
                [1.257728, -0.139648, -0.118081],
                [-0.078003, 0.975409, 0.102594],
                [-0.003316, 0.501214, 0.502102],
            ],
            [
                [1.278864, -0.125333, -0.153531],
                [-0.084748, 0.957674, 0.127074],
                [-0.000989, 0.601151, 0.399838],
            ],
            [
                [1.255528, -0.076749, -0.178779],
                [-0.078411, 0.930809, 0.147602],
                [0.004733, 0.691367, 0.3039],
            ],
        ]
)


# Functions
def generate_3dlut(matrix: np.array) -> np.array:
        if matrix.shape != (3, 3):
            raise ValueError("Matrix must be 3x3")

        grid = np.linspace(0, 1, dim)
        lut = np.zeros((dim, dim, dim, 3))

        for r in range(dim):
            for g in range(dim):
                for b in range(dim):
                    rgb = np.array([grid[r], grid[g], grid[b]])
                    transformed_rgb = np.dot(matrix, rgb)
                    #transformed_rgb = np.clip(transformed_rgb, 0, 1)
                    lut[r, g, b] = transformed_rgb

        return lut

def write_lut(lut: np.array, lut_path: str) -> None:
    lut_dim = lut.shape[0]
    
    with open(lut_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['R', 'G', 'B', 'R_out', 'G_out', 'B_out'])  # Header
        
        for r in range(lut_dim):
            for g in range(lut_dim):
                for b in range(lut_dim):
                    r_in, g_in, b_in = r / (lut_dim - 1), g / (lut_dim - 1), b / (lut_dim - 1)
                    r_out, g_out, b_out = lut[r, g, b]
                    writer.writerow([r_in, g_in, b_in, r_out, g_out, b_out])

def read_lut(filename: str):
    global lut
    df = pd.read_csv(filename)
    lut = np.zeros((dim, dim, dim, 3)) # R x G x B x Output
    
    for _, row in df.iterrows():
        r, g, b = int(row['R'] * (dim - 1)), int(row['G'] * (dim - 1)), int(row['B'] * (dim - 1))
        lut[r, g, b] = row[['R_out', 'G_out', 'B_out']]
    
    lut = torch.tensor(lut, dtype=torch.float32)
    if use_cuda: lut = lut.cuda()
    lut = lut.permute(3, 2, 1, 0)[None] # N x Output x R x G x B

#################### LUT 

def get_lut_fn(dir: str, lvl: float):
    if lvl == 0:
        return f'lut{dir}0-0.txt'
    if lvl == 1:
        return f'lut{dir}1-0.txt'
    return f"lut{dir}{lvl}".replace(".", '-')+'.txt'

# Create LUT
def make_lut(dir: str, lvl: float, lutdir: str = r'cvdsim/lut'):
    # Define LUT Path
    lut_path = os.path.join(lutdir, get_lut_fn(dir, lvl))
    if dir=='d': m = deut
    if dir=='p': m = prot
    if dir=='t': m = trit
    m_interp = np.array([
        [np.interp(lvl, m_match, m[:,0,0]), np.interp(lvl, m_match, m[:,0,1]), np.interp(lvl, m_match, m[:,0,2])],
        [np.interp(lvl, m_match, m[:,1,0]), np.interp(lvl, m_match, m[:,1,1]), np.interp(lvl, m_match, m[:,1,2])],
        [np.interp(lvl, m_match, m[:,2,0]), np.interp(lvl, m_match, m[:,2,1]), np.interp(lvl, m_match, m[:,2,2])]
    ])
    lut = generate_3dlut(m_interp)
    write_lut(lut, lut_path)


#################### Image

def change_setting(dir: str, lvl: float, lutdir: str = r'cvdsim/lut'):
    lut_path = os.path.join(lutdir, get_lut_fn(dir, lvl))
    read_lut(lut_path)
    global lut

def sim_image(image: torch.Tensor, name='') -> torch.Tensor: # image in PIL RGB format: Image.open(fp).convert('RGB')
    # Open Image
    if use_cuda: image = image.cuda()
    image = image * 2 - 1

    # Format dimensions CxHxW -> NxDxHxWxC
    #image = torch.tensor(image, dtype=torch.float32)[None, None]
    image = image[None]
    image = image.permute(0, 1, 3, 4, 2)

    # Apply LUT
    res = F.grid_sample(lut, image, padding_mode='border', align_corners=True) # align_corners is CRUCIAL
    res = res.permute(0, 2, 1, 3, 4) # NxCxDxHxW -> NxDxCxHxW
    res = res.squeeze(0) # DxCxHxW
    res = res.clamp(0, 1)

    # Save Transform: 
    return res

if __name__ == "__main__":
    for i in range(0, 11):
        make_lut('d', i/10)
        make_lut('p', i/10)
        make_lut('t', i/10)