import time
from PIL import Image
import torch

class Process:
    def __init__(self, M, S, H, modelonly=False, topil=True, optimize=True):
        self.M, self.S, self.H = M, S, H
        self.modelonly = modelonly
        self.topil = topil
        self.optimize = optimize

    def __call__(self, img):
        return self.forward(img)

    def forward(self, img):
        #t_start = time.time()
        # Ensure input is a CHW torch tensor
        img = img.float() / 255.0  # Normalize to [0,1]
        img = img.unsqueeze(0)  # Add batch dimension if missing
        #t_preprocess = time.time()
        img = img.cuda()

        # Apply the model
        if self.modelonly:
            img = self.M(img)
        else:
            img = self.H(self.S(self.M(img)), img)

        # Remove batch dimension
        img = img.squeeze(0)
        img = (img * 255).clamp(0, 255)


        if self.topil:
            
            img = img.byte()

            #t_denorm = time.time()

            # Convert (C, H, W) → (H, W, C)
            
            img = img.permute(1, 2, 0).contiguous().to("cpu", non_blocking=self.optimize)  # Ensures memory contiguity before NumPy conversion
            #t_np = time.time()

            # Convert to PIL Image
            img_pil = Image.fromarray(img.numpy())

            #t_pil = time.time()

            # print(f"Total: {t_pil-t_start:0.3f}s | "
            #     f"Preprocess: {t_preprocess-t_start:.3f}s | "
            #     f"Model: {t_model-t_preprocess:.3f}s | "
            #     f"Denorm: {t_denorm-t_model:.3f}s | "
            #     f"Contig + Numpy Convert: {t_np-t_denorm:.3f}s | "
            #     f"PIL Convert: {t_pil-t_np:.3f}s")

            return img_pil  # Always return a PIL image

        return img  # Return tensor if topil=False


