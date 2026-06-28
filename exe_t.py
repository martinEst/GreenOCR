import os
import gc
import cv2

import random

import time
import torch
import glob
import numpy as np
import torch.nn as nn
import kornia.color as KC
import kornia.filters as KF
import torch.nn.functional as F
import kornia.augmentation as K
import kornia.augmentation as KA
import torchvision.transforms as T
import torchvision.models as models

#torchvision
from torchvision import models
from torchvision import transforms

#torch

from torch.utils.data import Dataset


#PIL
from PIL import Image, ImageOps

#albumentations

#kornia


#pynvml

#basicsr
from basicsr.archs.rrdbnet_arch import RRDBNet

#realesrgan
from realesrgan import RealESRGANer

#local src
from src.config import VOCAB


# any input image or training image width can vary but height is fixed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Hardware & Memory Settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

# this is for my super weak GPU optimization
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True       # important for varying image widths!
torch.autograd.set_detect_anomaly(False)    # True when debugging NaNs

#we only fixing height, not widht
IMG_HEIGHT = 64
BLANK_INDEX = 0  # Explicitly mapped to index 0
num_classes = len(VOCAB)  # (Blank is already included inside VOCAB)


# Cleaning up VRAM before loading the model
gc.collect()  # Python garbage collection
torch.cuda.empty_cache()  # frees cached memory (not allocated memory)

#character mapping
char_to_idx = {char: idx for idx, char in enumerate(VOCAB)}
idx_to_char = {idx: char for idx, char in enumerate(VOCAB)}




def encode_text(text):
    """Converts a raw string into a list of integer indices using the global map."""
    return [char_to_idx[c] for c in text if c in char_to_idx]

def decode_target(tensor, idx2char=idx_to_char):
    """Converts a PyTorch tensor of indices back into a readable string."""
    return ''.join([idx2char.get(int(i), '?') for i in tensor])





##--------------------------------------------------------
## attempt to keep execution on going, if limited memory. 
##--------------------------------------------------------





def resize_keep_aspect(img, target_h=64):
    """
    Resizes an incoming image so its height matches the network requirement (64 pixels).
    Crucially scales the width proportionally to prevent the text from squishing or stretching.
    """
    # 1. Extract the current height and width from the image array dimensions
    h, w = img.shape[:2]

    # 2. Calculate the scaling multiplier (target height divided by original height)
    # Then multiply the original width by this ratio to find the proportional new width
    new_w = int(w * (target_h / h))
    
    # 3. SAFETY CHECK: If an image is extremely narrow, the math could round down to 0 width.
    # A width of 0 will crash OpenCV's resize function instantly, so we force a minimum of 1 pixel.
    if new_w < 1: 
        new_w = 1
        
    # 4. Perform the high-quality bicubic interpolation resize operation
    # cv2.resize expects the dimensional tuple argument formatted as (Width, Height)
    resized = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_CUBIC)
    
    return resized

#---------------------------------------------------------------------------------------------------

import torchvision.transforms as T
from PIL import Image

# Assuming Kornia imports are set up like this in your script:
import kornia.color as KC
import kornia.filters as KF

from src.transforms import DualChannelLaplaceTransform
train_transform = DualChannelLaplaceTransform(train=True)

#---------------------------------------------------------------------------------------------------



#init
model = CRNN(num_classes=num_classes,img_height=64,in_channels=1,freeze_cnn=True)


myvars = {}
with open("ref.cnf") as myfile:
    for line in myfile:
        name, var = line[:-1].strip().partition("=")[::2]
        myvars[name.strip()] = var.strip()


exe = True

#pathModel = myvars['model_interface']


    
if(exe):

    
    transform = DualChannelLaplaceTransform(train=False)

   # optimizer = torch.optim.Adam(model.parameters())  # Now includes RNN and FC
    """     dataloader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=ctc_collate_fn) """
    
   
    for singleModel in glob.glob( "models/*.pth"):
        print("model",singleModel)                
        """      regexp = re.compile(r'.*_[5-90-9]_.*')
                if regexp.search(singleModel):
                    print('matched')
            """
   
        model = torch.load(singleModel, map_location="cuda")
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

            tensor = torch.from_numpy(gray).float() / 255.0  # [H, W]
            tensor = tensor.unsqueeze(0).unsqueeze(0)        # [B=1, C=1, H, W]
           

            img_tensor = transform(tensor).unsqueeze(0).cuda() 
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
        
            ocrTxt = "".join(idx_to_char[i] for i in decoded_indices)
            print(ocrTxt)
            
            joined_text = ''.join(ocrTxt)
            cleaned_text = ' '.join(joined_text.split())
            print(cleaned_text)
    