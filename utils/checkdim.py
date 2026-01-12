def checkdim(x, exp=None):
    res = None
    shape = list(x.shape)
    if x.dim() == 3: # channels, height, width
        # find dimension that is 3 (that is channel dim)
        chdim = shape.index(3)
        assert chdim in [0, 2], f"Channel dimension should not be in middle of the tensor: {shape}"
        # width should be greater than height
        if chdim == 0: # CHW or CWH
            if shape[1] <= shape[2]:
                res = 'CHW'
            else:
                res = 'CWH'
        elif chdim == 2: # HWC or WHC
            if shape[1] > shape[0]:
                res = 'HWC'
            else:
                res = 'WHC'
    if x.dim() == 4:
        assert x.shape[0] == 1, f"Batch size should be 1: {shape}"
        # find dimension that is 3 (that is channel dim)
        chdim = shape.index(3)
        assert chdim in [1, 3], f"Channel dimension should be either index 1 or 3: {shape}"
        # width should be greater than height
        if chdim == 1: # BCHW or BCWH
            if shape[2] <= shape[3]:
                res = 'BCHW'
            else:
                res = 'BCWH'
        elif chdim == 3: # NHWC or NWHC
            if shape[2] > shape[1]:
                res = 'BHWC'
            else:
                res = 'BWHC'
                
    if exp is not None: assert res == exp, f"Image shape {res}, {shape} does not match expected shape {exp}"
    return res


if __name__ == "__main__":
    import torch
    a = torch.zeros([1, 3, 1290, 1080])
    print(checkdim(a))