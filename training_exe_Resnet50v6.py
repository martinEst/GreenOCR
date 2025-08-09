import os
import gc
import cv2
import time
import torch
import string

import numpy as np
import torch.nn as nn
import kornia.augmentation as K

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms
import torchvision.models as models

from PIL import Image


from torch.nn.utils import clip_grad_norm_

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as T


from torchvision.transforms import functional as TF
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import GradScaler, autocast


from torchvision import transforms
from torchvision import models
from torchvision.models import resnet50, ResNet50_Weights


from PIL import Image, ImageFilter
from PIL import Image

from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo


import albumentations as A
from albumentations.pytorch import ToTensorV2
from kornia.augmentation import AugmentationSequential
import torchvision.models as models


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



import numpy as np
from PIL import Image
from torch.utils.data import Dataset



from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import Dataset
from PIL import Image
import numpy as np



IMG_HEIGHT = 32  
IMG_WIDTH = 2048  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## training data has some of germanic scripts and they contain more characters than classical english

ctc_loss_fn = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
#germanic_chars = ['ſ','ſ','ä', 'ö', 'ü', 'Ä', 'Ö', 'Ü', 'ß', 'å', 'Å', 'æ', 'Æ', 'ø', 'Ø']
vocab = ['<blank>'] + ['ſ','—','“','„','’','ô','é']+ list(string.ascii_letters + string.digits + string.punctuation + " ") + ['ä', 'ö', 'ü', 'Ä', 'Ö', 'Ü', 'ß', 'å', 'Å', 'æ', 'Æ', 'ø', 'Ø']
#vocab = ";ſäöüÄÖÜßåÅæÆøØabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?- '"
num_classes = len(vocab) + 1  # +1 for CTC blank
# Add CTC blank at index 0
BLANK_INDEX = 0





import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import torch
import torch
import torchvision.transforms as T
import kornia as K
import kornia.color as KC
import kornia.filters as KF
import kornia.enhance as KE
import kornia.geometry.transform as KG
import torch
import torchvision.transforms as T
import kornia as K
import kornia.color as KC
import kornia.filters as KF
import kornia.enhance as KE
import kornia.augmentation as KA
import kornia.geometry.transform as KG
from PIL import Image

from PIL import Image
import os
import numpy as np
import torch
import torchvision.transforms as T
import kornia.augmentation as KA
import kornia.filters as KF
import kornia.color as KC
import kornia.geometry.transform as KG



import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import kornia.color as KC
import kornia.geometry as KG
import kornia.augmentation as KA
import kornia.filters as KF
import torch
import torch.nn as nn
import torchvision.transforms as T
import kornia.augmentation as KA
import kornia.color as KC
import kornia.filters as KF
import kornia.geometry.transform as KG
from PIL import Image
import numpy as np
gc.collect()  # Python garbage collection
torch.cuda.empty_cache()  # frees cached memory (not allocated memory)

import torch
import torchvision.transforms as T
import kornia.augmentation as KA
import kornia.color as KC
import kornia.geometry.transform as KG
import kornia.filters as KF
from PIL import Image
import numpy as np

class DualChannelLaplaceTransform:
    def __init__(self, target_height=64, max_width=None, 
                 train=True, output_channels=1,  # <-- NEW FLAG
                 sharpen_strength=1.5, contrast_clip=(0.01, 0.99)):
        """
        output_channels: 1 (single channel) or 2 (sharpened + laplace)
        """
        self.target_height = target_height
        self.max_width = max_width
        self.train = train
        self.output_channels = output_channels
        self.sharpen_strength = sharpen_strength
        self.contrast_clip = contrast_clip
        self.to_tensor = T.ToTensor()

        # Augmentations only used during training
        self.augment = torch.nn.Sequential(
            KA.RandomAffine(degrees=2.5, translate=(0.02, 0.02), p=0.5),
            KA.RandomBrightness(0.1, p=0.5),
            KA.RandomContrast(0.1, p=0.5),
            KA.RandomGaussianNoise(mean=0.0, std=0.01, p=0.2),
        )

    def __call__(self, img):
        
       

        # Step 1: Convert to tensor
        if isinstance(img, Image.Image):
            tensor = self.to_tensor(img).unsqueeze(0)  # [1,C,H,W]
        elif isinstance(img, np.ndarray):
            tensor = torch.from_numpy(img).float().unsqueeze(0)
        elif isinstance(img, torch.Tensor):
            if img.ndim == 3:
                tensor = img.unsqueeze(0)
            else:
                tensor = img
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

        # Step 2: Grayscale
        if tensor.shape[1] == 3:
            gray = KC.rgb_to_grayscale(tensor)
        else:
            gray = tensor

        # Step 3: Proportional resize
        gray = self._resize_proportional(gray)

        # Step 4: Augmentation (if training)
        if self.train:
            gray = self.augment(gray)

        # Step 5: Contrast stretch
        gray = self._contrast_stretch(gray, *self.contrast_clip)

        # Step 6: Sharpen
        blurred = KF.gaussian_blur2d(gray, (3, 3), (1.0, 1.0))
        sharpened = torch.clamp(gray + self.sharpen_strength * (gray - blurred), 0.0, 1.0)

        if self.output_channels == 1:
            # Single channel output
            return sharpened.squeeze(0)  # [1,H,W] -> remove batch dim for dataset __getitem__

        # Step 7: Laplacian (only if output_channels == 2)
        laplace = KF.laplacian(sharpened, kernel_size=3)

        # Step 8: Extract first channel from each (safe for B>1)
        sharpened_1ch = sharpened[:, 0:1, :, :]
        laplace_1ch = laplace[:, 0:1, :, :]

        # Step 9: Combine into [2,H,W]
        combined = torch.cat([sharpened_1ch.squeeze(0), laplace_1ch.squeeze(0)], dim=0)


        #img_tensor = transform(images[0]).unsqueeze(0).cuda() 
        #from torchvision.transforms.functional import to_pil_image
        #to_pil_image(sharpened[0]).show(title="Channel 0 (Sharpened)")


        return combined

    def _resize_proportional(self, img):
        """Resize keeping aspect ratio to target height."""
        _, _, h, w = img.shape
        scale = self.target_height / h
        new_w = int(round(w * scale))
        if self.max_width:
            new_w = min(new_w, self.max_width)
        return KG.resize(img, (self.target_height, new_w), interpolation='bilinear')

    def _contrast_stretch(self, img, low=0.01, high=0.99):
        if img.dim() == 4:
            return torch.stack([self._contrast_stretch(single_img, low, high) for single_img in img])
        stretched = torch.empty_like(img)
        for c in range(img.shape[0]):
            flat = img[c].flatten()
            low_val = flat.kthvalue(int(flat.numel() * low)).values
            high_val = flat.kthvalue(int(flat.numel() * high)).values
            stretched[c] = torch.clamp((img[c] - low_val) / (high_val - low_val + 1e-6), 0, 1)
        return stretched



class SingleChannelTransform:
    def __init__(self, target_height=64, max_width=None, train=True, sharpen_strength=1.5, contrast_clip=(0.01, 0.99)):
        self.target_height = target_height
        self.max_width = max_width
        self.train = train
        self.sharpen_strength = sharpen_strength
        self.contrast_clip = contrast_clip
        self.to_tensor = T.ToTensor()

        # Augmentations only during training
        self.augment = torch.nn.Sequential(
            KA.RandomAffine(degrees=2.5, translate=(0.02, 0.02), p=0.5),
            KA.RandomBrightness(0.1, p=0.5),
            KA.RandomContrast(0.1, p=0.5),
            KA.RandomGaussianNoise(mean=0.0, std=0.01, p=0.2),
        )

    def __call__(self, img):
        # Step 1: Convert to tensor
        if isinstance(img, Image.Image):  
            tensor = self.to_tensor(img).unsqueeze(0)  # [1, C, H, W]
        elif isinstance(img, np.ndarray):
            tensor = torch.from_numpy(img).float().unsqueeze(0)
            if tensor.ndim == 3 and tensor.shape[0] != 1:
                tensor = tensor.mean(dim=0, keepdim=True).unsqueeze(0)  
        elif isinstance(img, torch.Tensor):
            if img.ndim == 3:  
                tensor = img.unsqueeze(0)  
            elif img.ndim == 4:
                tensor = img  
            else:
                raise ValueError(f"Unexpected tensor shape: {img.shape}")
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

        # Step 2: Ensure grayscale
        if tensor.shape[1] == 3:
            gray = KC.rgb_to_grayscale(tensor)
        else:
            gray = tensor  

        # Step 3: Proportional resize
        gray = self._resize_proportional(gray)

        # Step 4: Augment (train only)
        if self.train:
            gray = self.augment(gray)

        # Step 5: Contrast normalization
        gray = self._contrast_stretch(gray, *self.contrast_clip)

        # Step 6: Sharpen
        blurred = KF.gaussian_blur2d(gray, (3, 3), (1.0, 1.0))
        sharpened = gray + self.sharpen_strength * (gray - blurred)
        sharpened = torch.clamp(sharpened, 0.0, 1.0)

        # Output single channel [1, H, W]
        return sharpened.squeeze(0)

    def _resize_proportional(self, img):
        """Resize keeping aspect ratio to target height."""
        _, _, h, w = img.shape
        scale = self.target_height / h
        new_w = int(round(w * scale))
        if self.max_width:
            new_w = min(new_w, self.max_width)
        return KG.resize(img, (self.target_height, new_w), interpolation='bilinear')

    def _contrast_stretch(self, img, low=0.01, high=0.99):
        if img.dim() == 4:  # batch mode
            return torch.stack([self._contrast_stretch(single_img, low, high) for single_img in img])

        stretched = torch.empty_like(img)
        for c in range(img.shape[0]):
            flat = img[c].flatten()
            low_val = flat.kthvalue(int(flat.numel() * low)).values
            high_val = flat.kthvalue(int(flat.numel() * high)).values
            stretched[c] = torch.clamp((img[c] - low_val) / (high_val - low_val + 1e-6), 0, 1)
        return stretched



# -----------------------------
# Line Segmentation
# -----------------------------



# -----------------------------
# AUGMENTATION
# -----------------------------



#few different albumentations approaches
#------------------
# albumentations
# several different function to try  

def get_train_transform(image_height=32, image_max_width=2048):
    return A.Compose([
        A.Resize(height=image_height, width=image_max_width, interpolation=1, always_apply=True),  # Resize to fixed height
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.ISONoise(p=0.2),
        ], p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.Downscale(scale_min=0.3, scale_max=0.7, interpolation=0, p=0.2),
        A.ToGray(p=1.0),  # Ensure grayscale
        A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
        ToTensorV2()
    ])



# kornia






#---------------------------------------------------------------------------------------------------
# encode/decode characters

# Char to index (starting from 1 because 0 is reserved for blank)
char_to_idx = {char: idx + 1 for idx, char in enumerate(vocab)}

# Index to char (inverse)
idx_to_char = {idx + 1: char for idx, char in enumerate(vocab)}
idx_to_char[BLANK_INDEX] = ''  # blank is mapped to empty string



def decode_target(tensor, idx2char):
    return ''.join([idx2char.get(int(i), '?') for i in tensor])



def encode_text(text):
    return [char_to_idx[c] for c in text if c in char_to_idx]


def encode_text(text, char_to_idx):
    """Convert string to list of integer indices (no blank)"""
    return [char_to_idx[c] for c in text if c in char_to_idx]

def decode_prediction(preds, idx_to_char, blank=0):
    """Greedy decoding: remove duplicates and blanks"""
    
    pred_texts = []
    for pred in preds:  # each prediction is a list of indices
        string = []
        prev = None
        
        if pred != blank and pred != prev:
            string.append(idx_to_char.get(pred, ''))
            prev = pred
        pred_texts.append(''.join(string))
    return pred_texts


def ctc_greedy_decoder(preds, blank=0):
    decoded = []
    prev = blank
    for p in preds:
        if p != prev and p != blank:
            decoded.append(p.item())
        prev = p
    return decoded



##--------------------------------------------------------
## attempt to keep execution on going, if limited memory. 
##--------------------------------------------------------
def clear_gpu_memory():
        torch.cuda.empty_cache()
        gc.collect()
        

def wait_until_enough_gpu_memory(min_memory_available, max_retries=10, sleep_time=5):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(torch.cuda.current_device())

    for _ in range(max_retries):
        info = nvmlDeviceGetMemoryInfo(handle)
        if info.free >= min_memory_available:
            break
        print(f"Waiting for {min_memory_available} bytes of free GPU memory. Retrying in {sleep_time} seconds...")
        time.sleep(sleep_time)
    else:
        raise RuntimeError(f"Failed to acquire {min_memory_available} bytes of free GPU memory after {max_retries} retries.")


#---------------------------------------------------------------------------------------------------

##--------------------------------------------------------
## ctc 
##--------------------------------------------------------


def ctc_loss(y_pred, labels, input_lengths, label_lengths):
    """
    Compute CTC loss with validation.
    
    Args:
        y_pred (torch.Tensor): Model output of shape (batch, seq_len, num_classes).
        labels (torch.Tensor): Target sequences of shape (batch, max_label_len).
        input_lengths (torch.Tensor): Length of each input sequence.
        label_lengths (torch.Tensor): Length of each target sequence.
    
    Returns:
        torch.Tensor: CTC loss.
    """
    # Validate lengths
    max_input_length = y_pred.size(1)  # seq_len = W/16
    if (label_lengths > input_lengths).any() or (label_lengths > max_input_length).any():
        raise ValueError(f"label_lengths ({label_lengths}) must be <= input_lengths ({input_lengths}) "
                        f"and <= output seq_len ({max_input_length})")
    
    y_pred = y_pred.log_softmax(2)  # Apply log_softmax
    return torch.nn.functional.ctc_loss(
        y_pred.transpose(0, 1),  # (seq_len, batch, num_classes)
        labels,
        input_lengths,
        label_lengths,
        blank=0,  # Assume blank is index 0
        reduction='mean'
    )




def ctc_collate_fn(batch, total_stride=4):
    images, labels = zip(*batch)
    
    max_w = max(img.shape[-1] for img in images)
    C, H = images[0].shape[:2]  # assume consistent channels & height
    
    padded_images_tensor = torch.zeros(len(images), C, H, max_w, dtype=images[0].dtype)
    for i, img in enumerate(images):
        padded_images_tensor[i, :, :, :img.shape[-1]] = img
    
    targets = torch.cat(labels)
    input_lengths = torch.tensor([img.shape[-1] // total_stride for img in images], dtype=torch.long)
    target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)

    return padded_images_tensor, targets, input_lengths, target_lengths




#---------------------------------------------------------------------------------------------------
#
##--------------------------------------------------------
## classes 
##--------------------------------------------------------
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch

import os
from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np

import os
from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np

class OCRDataset(Dataset):
    def __init__(self, label_file, image_dir, transform=None):
        """
        Args:
            label_file (str): Path to label file in format 'image_name|text'
            image_dir (str): Directory with images
            char_to_idx (dict): Mapping from characters to indices
            transform (callable, optional): Transform applied to the image
        """
        self.image_dir = image_dir
        self.transform = transform
        self.samples = []

        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                image_name, text = line.strip().split("|")
                self.samples.append((image_name, text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, text = self.samples[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Load grayscale image
        img_pil = Image.open(img_path).convert("L")

        # Convert to tensor [1, H, W]
        img_np = np.array(img_pil, dtype=np.float32)
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)  # [1, H, W]

        # Apply transform if provided
        if self.transform:
            img_tensor = self.transform(img_tensor)

        # Encode label
        label_tensor = torch.tensor(
            [char_to_idx[c] for c in text],
            dtype=torch.long
        )

        return img_tensor, label_tensor




class CRNN(nn.Module):
    def __init__(self, num_classes, img_height=64, img_width=2048, in_channels=1, freeze_cnn=False):
        super(CRNN, self).__init__()
        self.normalize = transforms.Normalize(mean=[0.5] * in_channels, std=[0.5] * in_channels)
        
        self.img_height = img_height
        self.img_width = img_width
        self.in_channels = in_channels
        self.num_classes = num_classes

        resnet = models.resnet50(pretrained=False)
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])

        if freeze_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False

        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, img_height, img_width)
            dummy_output = self.cnn(dummy_input)
            _, c, h, w = dummy_output.size()
            self.sequence_length = w
            self.feature_dim = c  # height collapsed to 1 later

        self.lstm1 = nn.LSTM(self.feature_dim, 256, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.normalize(x)
        x = self.cnn(x)

        # Collapse height to 1 → [B, C, 1, W]
        x = F.adaptive_avg_pool2d(x, (1, None))

        # Convert to sequence format → [B, W, C]
        x = x.squeeze(2).permute(0, 2, 1)

        # Pass through BiLSTMs
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        # Final prediction
        x = self.fc(x)

        return F.log_softmax(x, dim=2)


#---------------------------------------------------------------------------------------------------

## training / execution of model 


#init
model = CRNN(num_classes=num_classes,img_height=64,img_width=2048,in_channels=1)




#---------------------------------------------------------------------------------------------------
# exe = True # means model is applied 



exe = True
epocs_iterat = 30


if(exe):
    #transform = DualChannelLaplaceTransform(resize=(32, 2048))
    transform = DualChannelLaplaceTransform(train=False)

   # optimizer = torch.optim.Adam(model.parameters())  # Now includes RNN and FC
    """     dataloader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=ctc_collate_fn) """
    
    state = model.load_state_dict(torch.load("/home/martinez/TUS/DISSERT/models/crnn_ctc_model_DvJhB_30_ep_29.pth"))  # or your path
    model = model.to("cuda")
    print("state keys:")

    
    image = Image.open('/home/martinez/TUS/DISSERT/data/customImages/enhanced_image.jpg').convert('RGB')  # grayscale
    img_tensor = transform(image).unsqueeze(0).cuda() 
    #from torchvision.transforms.functional import to_pil_image
   #to_pil_image(img_tensor[0]).show(title="Channel 0 (Sharpened)")

    """  from torchvision.transforms.functional import to_pil_image

        img_single = img_tensor[0]

        # View each channel separately
        to_pil_image(img_single[0]).show(title="Channel 0 (Sharpened)")
        to_pil_image(img_single[1]).show(title="Channel 1 (Laplace)")
    
        """
    preds = None
    #with torch.no_grad():
    
    output = model(img_tensor)  # Output: (T, B, num_classes)
    log_probs = torch.nn.functional.log_softmax(output, dim=2)
      #  preds = ctc_greedy_decoder(log_probs)
      # Return single prediction
    preds = log_probs.argmax(dim=2)
    print(preds)
    decoded_indices = ctc_greedy_decoder(preds[0])
    print(decoded_indices)
    
    ocrTxt = decode_prediction (decoded_indices, idx_to_char)
    
    joined_text = ''.join(ocrTxt)
    cleaned_text = ' '.join(joined_text.split())
    print(cleaned_text)
    print("exit")
    import sys
    sys.exit()





else:

 

# Create dataset
    import random
    import string
    s = string.ascii_letters

    random_level2 = random.choices(s, k=5)
    res = ''.join(random_level2)

    transform = DualChannelLaplaceTransform(train=True)
 
    train_dataset = OCRDataset(
    label_file="data/latest/matchingLabelFileV2.txt",
    image_dir="data/latest/images_all/",
    #char_to_idx=char_to_idx,
    #laplace_dir="data/latest/lapace_all/",
    transform=transform
)


    dataloader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=ctc_collate_fn,
        num_workers=os.cpu_count(),       # Suggestion: speed up data loading
        pin_memory=True                   # Optional: improves transfer to GPU if using CUDA
    )

    # Define loss function
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)  


    images, targets, input_lengths, target_lengths = next(iter(dataloader))

    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    device = torch.device("cuda")
    model = model.to(device)    

    
    """  from torchvision.transforms.functional import to_pil_image

        img_single = img_tensor[0]

        # View each channel separately
        to_pil_image(img_single[0]).show(title="Channel 0 (Sharpened)")
        to_pil_image(img_single[1]).show(title="Channel 1 (Laplace)")
    
        """
    
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
    #train(model, dataloader, optimizer, ctc_loss, device, 2);


    best_val_loss = float('inf')
    # Usage example
    for epoch in range(epocs_iterat):  # or more
        model.train()
        total_loss = 0

        min_memory_available = 2 * 1024 * 1024 * 1024  # 2GB
        #clear_gpu_memory()
        #wait_until_enough_gpu_memory(min_memory_available)

        for images, targets, input_lengths, target_lengths in dataloader:
            clear_gpu_memory()
            wait_until_enough_gpu_memory(min_memory_available)
 
            try:
                #torch.cuda.memory_summary(device=None, abbreviated=False)
                #torch.cuda.empty_cache()
                if images is None:
                    print("⚠️ Skipping None batch")
                    continue

                images = images.to(device)
                #targets = targets.to(device)
                #input_lengths = input_lengths.to(device)
                target_lengths = target_lengths.to(device).clone().detach()
                #target_lengths = target_lengths.to(device)
                print("target_lengths dtype:", target_lengths.dtype)
                print("target_lengths values:", target_lengths)
                print("Sum:", target_lengths.sum())

                if targets.dim() == 2:
                    # Still batched, need to flatten
                    clean_targets = []
                    for i in range(targets.size(0)):
                        l = target_lengths[i]
                        t = targets[i]
                        if t.dim() == 0:
                            t = t.unsqueeze(0)
                        clean_targets.append(t[:l])
                    clean_targets = torch.cat(clean_targets).to(device)

                elif targets.dim() == 1:
                    # Already flat — just check it's consistent
                    if targets.shape[0] != target_lengths.sum():
                        print("❌ Already flat targets but lengths don't match")
                        continue
                    clean_targets = targets.to(device)

                else:
                    print("❌ Unknown target shape:", targets.shape)
                    continue

                # Final safety check
                if clean_targets.numel() == 0:
                    print("⚠️ Empty clean_targets — skipping batch.")
                    continue

                # Set final targets to clean version
                targets = clean_targets

            
                logits = model(images)  # (B, T, C)
                log_probs = F.log_softmax(logits, dim=2)  # Apply log softmax over classes (C)

                # Permute to (T, B, C) for CTC loss
                log_probs = log_probs.permute(1, 0, 2)  # (T, B, C)

                # Input lengths (T should be log_probs.size(0) after permute)
                T = log_probs.size(0)
                B = log_probs.size(1)
                #input_lengths = torch.full(size=(B,), fill_value=T, dtype=torch.long).to(device)
                input_lengths = torch.full((B,), log_probs.size(0), dtype=torch.long).to(device)
                print("Sum4:", target_lengths.sum())
                if any(target_lengths > input_lengths):
                    print(f"⚠️ Skipping batch: some target_lengths > T={T}")
                    continue
                safe_targets = targets.detach()
                safe_input_lengths = input_lengths.detach()
                #safe_target_lengths = target_lengths.detach()
                safe_target_lengths = target_lengths.detach().clone()
                valid_indices = [i for i, l in enumerate(target_lengths) if l > 0]
                # Apply filtering
                images = images[valid_indices]
                safe_targets = torch.cat([safe_targets[i].unsqueeze(0) for i in valid_indices], dim=0) if safe_targets.ndim > 1 else safe_targets
                safe_input_lengths = safe_input_lengths[valid_indices]
                safe_target_lengths = safe_target_lengths[valid_indices]


                # Filter out samples with target_length == 0
                valid_indices = [i for i, l in enumerate(safe_target_lengths) if l > 0]

                #loss = ctc_loss(log_probs, safe_targets, safe_input_lengths, safe_target_lengths)
                loss = ctc_loss_fn(log_probs, safe_targets, safe_input_lengths, safe_target_lengths)


                # Compute loss safely
                

                if loss is not None:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                elif torch.isnan(loss) or torch.isinf(loss):
                        print("⚠️ Skipping invalid loss")
                        continue
                else:
                    print("⚠️ Skipped batch due to invalid loss")
                    continue
                



                assert targets.shape[0] == target_lengths.sum()
                
                total_loss += loss.item()
            except RuntimeError as e:
                print(f"🔥 Skipping batch due to error: {e}")
                torch.cuda.empty_cache()
                continue
        

        avg_loss = total_loss / len(dataloader)
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            print("saving epocs")
            torch.save(model.state_dict(), "/home/martinez/TUS/DISSERT/models/crnn_ctc_model_"+res+"_"+str(epocs_iterat)+"_ep_"+str(epoch)+".pth")
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

