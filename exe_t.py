import os
import gc
import time
import torch
import string
import random
import numpy as np
import torch.nn as nn

import sys
import kornia as K
import kornia.color as KC
import kornia.filters as KF
import kornia.enhance as KE
import kornia.augmentation as K
import kornia.augmentation as KA
import torchvision.transforms as T
import kornia.geometry.transform as KG
import torchvision.transforms.functional as TF


from torch.utils.data import Dataset
from PIL import Image


from torch.utils.data import Dataset



import torchvision.transforms as T
import kornia.color as KC
import kornia.geometry as KG
import kornia.augmentation as KA
import kornia.filters as KF
import torch.nn as nn
import torchvision.transforms as T
import kornia.augmentation as KA
import kornia.color as KC
import kornia.filters as KF
import kornia.geometry.transform as KG
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms
import torchvision.models as models

from PIL import Image


import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import os
import torch

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

import torchvision.transforms as T
import kornia.augmentation as KA
import kornia.color as KC
import kornia.geometry.transform as KG
import kornia.filters as KF
from PIL import Image
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet

from realesrgan import RealESRGANer
from PIL import Image
import torch
import os
import cv2

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import unicodedata


# any input image or training image width can vary but height is fixed
IMG_HEIGHT = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## training data has some of germanic scripts and they contain more characters than classical english



#some special characters that appeared in germanic manuscripts, maybe should be avoided now
ctc_loss_fn = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)


""" vocab = ['<blank>'] +  list("abcdefghijklmnopqrstuvwxyz") + \
        list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + \
        list("0123456789") + \
        list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ")  # include space

 """

#vocab = build_vocab()
vocab = ['<blank>'] + list("ÈĒĖēėěęĚĘëéèÉÊËðÐŊŋ") + list("£§êàâé£§⊥")+['£','ſ','—','“','„','’','ô','é']+ list(string.ascii_letters + string.digits + string.punctuation + " ") + ['ä', 'ö', 'ü', 'Ä', 'Ö', 'Ü', 'ß', 'å', 'Å', 'æ', 'Æ', 'ø', 'Ø']

num_classes = len(vocab) + 1  # +1 for CTC blank
# Add CTC blank at index 0
BLANK_INDEX = 0








gc.collect()  # Python garbage collection
torch.cuda.empty_cache()  # frees cached memory (not allocated memory)



  

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

""" def get_train_transform(image_height=32, image_max_width=2048):
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

 """

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


def ctc_greedy_decoder_batch(preds, idx_to_char, blank=0):
    """
    Greedy CTC decoding (batch version).
    - Skips blanks
    - Deduplicates only consecutive repeats (not across blanks)
    - Uses idx_to_char to map indices back to string
    """
    pred_texts = []
    decoded = []
    prev = blank
    for p in preds:               # loop over timesteps
        p = p.item() if hasattr(p, "item") else int(p)
        if p != prev and p != blank:
            decoded.append(idx_to_char.get(p, ''))
        prev = p
    pred_texts.append("".join(decoded))
    return pred_texts




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
from PIL import Image, ImageOps
import os

def pad_to_required_width(img, target_length, downsample_factor=32, pad_color=255):
    min_width = target_length * downsample_factor
    w, h = img.size
    if w < min_width:
        pad_width = min_width - w
        # Pad right only
        padding = (0, 0, pad_width, 0)
        img = ImageOps.expand(img, padding, fill=pad_color)
    return img

# Example usage
#img = Image.open("sample.png").convert("L")
#text = "some label here"
#padded_img = pad_to_required_width(img, target_length=len(text))
#padded_img.save("sample_padded.png")

##--------------------------------------------------------
## ctc 
##--------------------------------------------------------

def collate_fn(batch):
    images, texts, widths = zip(*batch)
    max_w = max(widths)
    
    padded_imgs = []
    for img in images:
        pad_w = max_w - img.shape[2]
        padded = torch.nn.functional.pad(img, (0, pad_w, 0, 0), value=1.0)  # pad right with white
        padded_imgs.append(padded)
    target_lengths = torch.tensor([len(label) for label in texts], dtype=torch.long)
    targets = torch.cat(texts)
    input_lengths = torch.tensor([img.shape[-1] for img in images], dtype=torch.long)

    return torch.stack(padded_imgs),targets, input_lengths,target_lengths

#    return padded_images_tensor, targets, input_lengths, target_lengths """


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

def resize_keep_aspect(img, target_h=64):
    """
    Resize image to target height while keeping aspect ratio.
    Does not pad or crop — only resizes proportionally.

    Args:
        img: input image (grayscale or color)
        target_h: desired height
    """
    h, w = img.shape[:2]

    # Compute new width maintaining aspect ratio
    new_w = int(w * (target_h / h))
    resized = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

    return resized

#---------------------------------------------------------------------------------------------------
#
##--------------------------------------------------------
## classes 
##--------------------------------------------------------
import math
import random
from torch.utils.data import Sampler

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

        # avoid resizing - stretching or proportsional 
        # Step 3: Proportional resize 
        #gray = self._resize_proportional(gray)

        # Step 4: Augmentation (if training)
        if self.train:
            gray = self.augment(gray)

        # Step 5: Contrast stretch
        gray = self._contrast_stretch(gray, *self.contrast_clip)

        # Step 6: Sharpen
        blurred = KF.gaussian_blur2d(gray, (3, 3), (1.0, 1.0))
        sharpened = torch.clamp(gray + self.sharpen_strength * (gray - blurred), 0.0, 1.0)

        return sharpened.squeeze(0)  # [1,H,W] -> remove batch dim for dataset __getitem__
                


class OCRDataset(Dataset):
    def __init__(self, label_file, image_dir, transform=None, dataSrc = None):
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
        #format is folder name, filename , text 
        if dataSrc == "En" :
            with open(label_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        folder_image_name, text = line.strip().split(" ",1)
                        folder, image_name =  folder_image_name.strip().split(",")

                        self.samples.append((folder+"/"+image_name+".png", text))
                    except:
                        print(line)
        else:   ##old way
            with open(label_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        image_name, text = line.strip().split("|")
                        self.samples.append((image_name, text))
                    except:
                        print(line)

                    

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, text = self.samples[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Load grayscale image
        img_pil = Image.open(img_path).convert("L")
        img_pil = pad_to_required_width(img_pil, target_length=len(text))
        w, h = img_pil.size
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
        
        return img_tensor, label_tensor, w


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch
import torch.nn as nn
import torchvision.models as models
import glob
class CRNN(nn.Module):
    def __init__(self, num_classes, img_height=64, in_channels=1, freeze_cnn=False):
        super(CRNN, self).__init__()
        self.normalize = transforms.Normalize(mean=[0.5] * in_channels, std=[0.5] * in_channels)
        
        self.img_height = img_height
        self.in_channels = in_channels
        self.num_classes = num_classes


        resnet = models.resnet50(pretrained=True)

        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])

        if freeze_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False

        with torch.no_grad():
            #dummy_input = torch.zeros(1, in_channels, 100)
            dummy = torch.zeros(1, in_channels, img_height, 100)  # 100 = arbitrary width

            dummy_output = self.cnn(dummy)
            _, c, h, w = dummy_output.size()
            self.sequence_length = w
            self.feature_dim = c  # height collapsed to 1 later

        
        self.lstm1 = nn.LSTM(self.feature_dim, 256, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        self.lstm3 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
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

        return x


#---------------------------------------------------------------------------------------------------

## training / execution of model 
def ctc_greedy_decoder(preds, blank=0):
    decoded = []
    prev = blank
    for p in preds:
        if p != prev and p != blank:
            decoded.append(p.item())
        prev = p
    return decoded



#init
model = CRNN(num_classes=num_classes,img_height=64,in_channels=1)


myvars = {}
with open("ref.cnf") as myfile:
    for line in myfile:
        name, var = line[:-1].strip().partition("=")[::2]
        myvars[name.strip()] = var.strip()


exe = True

#pathModel = myvars['model_interface']


    
import re
if(exe):

    
    transform = DualChannelLaplaceTransform(train=False)

   # optimizer = torch.optim.Adam(model.parameters())  # Now includes RNN and FC
    """     dataloader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=ctc_collate_fn) """
    
   
    for singleModel in glob.glob( "models/*.pth"):
                
        """      regexp = re.compile(r'.*_[5-90-9]_.*')
                if regexp.search(singleModel):
                    print('matched')
            """
   
        model = torch.load(singleModel, map_location="cpu")
        state_dict = model.state_dict()
        #state = model.load_state_dict(torch.load(singleModel))  # or your path
        model = model.to("cuda")
        print("state keys:")

        
        for file in glob.glob( myvars['imgTest']+"/*.png"):

            print("IMG",file)
            #enchance input image ,using RealESRGANer outscale it 8, image will be rebuilt with some of the realesrg magic 
            # and then downsample it to height 64 , need for our ocr model

            # Step 1: Load image (OpenCV loads in BGR format)
            img = cv2.imread(file, cv2.IMREAD_COLOR)

            # Step 2: Create the model
            mod = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

            # Step 3: Create the RealESRGANer instance
            upsampler = RealESRGANer(
                scale=4,
                model_path='/home/martinez/d/weights/RealESRGAN_x4plus.pth', 
                model=mod,
                tile=0,  # Set to >0 for tiled inference on large images
                tile_pad=10,
                pre_pad=0,
                half=False  # Set True if using half-precision and CUDA
            )

            # Step 4: Super-resolve
            output, _ = upsampler.enhance(img, outscale=1)
            print("Original shape:", img.shape)
            print("Upscaled shape:", output.shape) 

            #target_w, target_h = 32, 64
            #downscaled = cv2.resize(output, (target_w, target_h), interpolation=cv2.INTER_AREA)
            output_proportional = resize_keep_aspect(output, target_h=64)  #return 3 channel RGB

            #?? can we increase training data from height 64 to 98 ? 

            gray = cv2.cvtColor(output_proportional, cv2.COLOR_BGR2GRAY)  # or COLOR_RGB2GRAY depending on your color format
        # gray = cv2.cvtColor(output_proportional, cv2.COLOR_BGR2GRAY)  # or COLOR_RGB2GRAY depending on your color format
            tensor = torch.from_numpy(gray).float() / 255.0  # [H, W]
            tensor = tensor.unsqueeze(0).unsqueeze(0)        # [B=1, C=1, H, W]
            # Convert to tensor
            #tensor = torch.from_numpy(gray).float() / 255.0    # normalize 0–1
            #tensor = tensor.unsqueeze(0).unsqueeze(0)   
            #maybe we should train on RGB ?

            #tmp solution 
            #cv2.imwrite("/tmp/enchcned_downsample.png", output_proportional)

            #image = Image.open('/home/martinez/TUS/DISSERT/data/latest/all_images2/Bennett__8e_down.png').convert('L')  # grayscale

            #image2 = Image.open('/home/martinez/TUS/DISSERT/data/customImages/a01-000u-00.png').convert('RGB')  # grayscale
            #img_tensor = transform(image2).unsqueeze(0).cuda() 

            img_tensor = transform(tensor).unsqueeze(0).cuda() 

        
            #from torchvision.transforms.functional import to_pil_image
            #to_pil_image(img_tensor[0]).show(title="Channel 0 (Sharpened)")

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
            
        
        
            #ocrTxt = ctc_greedy_decoder_batch (decoded_indices, idx_to_char)
            ocrTxt = "".join(idx_to_char[i] for i in decoded_indices)
            print(ocrTxt)

            #ocrTxt = ctc_greedy_decoder_batch (decoded_indices, idx_to_char)
            
            joined_text = ''.join(ocrTxt)
            cleaned_text = ' '.join(joined_text.split())
            print(cleaned_text)
    