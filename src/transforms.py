import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image

import kornia.color as KC
import kornia.filters as KF
import kornia.augmentation as KA

class DualChannelLaplaceTransform:
    def __init__(self, train=True, sharpen_strength=1.5, contrast_clip=(0.01, 0.99)):
        self.train = train
        self.sharpen_strength = sharpen_strength
        self.contrast_clip = contrast_clip
        self.to_tensor = T.ToTensor()

        # Augmentations are ONLY used if train=True
        if self.train:
            self.augment = torch.nn.Sequential(
                KA.RandomAffine(degrees=2.5, translate=(0.02, 0.02), p=0.5),
                KA.RandomBrightness((0.9, 1.1), p=0.5), # Fixed syntax for modern Kornia
                KA.RandomContrast((0.9, 1.1), p=0.5),
                KA.RandomGaussianNoise(mean=0.0, std=0.01, p=0.2),
            )

    def _contrast_stretch(self, img, low=0.01, high=0.99):
        """Standardises pixel intensities to bring out hidden text outline."""
        if img.dim() == 4:
            return torch.stack([self._contrast_stretch(single_img, low, high) for single_img in img])
            
        stretched = torch.empty_like(img)
        for c in range(img.shape[0]):
            flat = img[c].flatten()
            low_val = flat.kthvalue(int(flat.numel() * low)).values
            high_val = flat.kthvalue(int(flat.numel() * high)).values
            stretched[c] = torch.clamp((img[c] - low_val) / (high_val - low_val + 1e-6), 0, 1)
        return stretched

    def __call__(self, img, device="cpu"):
        # Step 1: Uniform conversion to PyTorch Tensor
        if isinstance(img, Image.Image):
            tensor = self.to_tensor(img).unsqueeze(0)  # [1, C, H, W]
        elif isinstance(img, np.ndarray):
            if img.ndim == 3:
                tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
            else:
                tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
        elif isinstance(img, torch.Tensor):
            tensor = img.unsqueeze(0) if img.ndim == 3 else img
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

        # Send to the correct device (CPU or CUDA) to prevent Kornia crashes
        tensor = tensor.to(device)

        # Step 2: Grayscale
        if tensor.shape[1] == 3:
            gray = KC.rgb_to_grayscale(tensor)
        else:
            gray = tensor

        # Step 3: Augmentation (Only happens when training)
        if self.train:
            gray = self.augment(gray)

        # Step 4: Core Processing (Happens in BOTH training and execution)
        gray = self._contrast_stretch(gray, *self.contrast_clip)

        # Step 5: Sharpening (Happens in BOTH training and execution)
        blurred = KF.gaussian_blur2d(gray, (3, 3), (1.0, 1.0))
        sharpened = torch.clamp(gray + self.sharpen_strength * (gray - blurred), 0.0, 1.0)

        return sharpened.squeeze(0)  # Returns clean [1, H, W] tensor
