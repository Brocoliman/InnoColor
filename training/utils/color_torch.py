import torch

def rgb2yuv_bt709(x):
    r = x[:, 0, :, :]
    g = x[:, 1, :, :]
    b = x[:, 2, :, :]
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    u = -0.1146 * r - 0.3854 * g + 0.5 * b + 0.5
    v = 0.5 * r  - 0.4542 * g - 0.0458 * b + 0.5
    yuv = torch.cat((y.unsqueeze(1), u.unsqueeze(1), v.unsqueeze(1)), dim=1)
    return yuv

def yuv2rgb_bt709(x):
    y = x[:, 0, :, :]
    u = x[:, 1, :, :]
    v = x[:, 2, :, :]
    r = y + 1.5748 * (v-0.5)
    g = y - 0.1873 * (u-0.5) - 0.4681 * (v-0.5)
    b = y + 1.8556 * (u-0.5)
    rgb = torch.cat((r.unsqueeze(1), g.unsqueeze(1), b.unsqueeze(1)), dim=1)
    return rgb



