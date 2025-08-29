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

from pathlib import Path

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
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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

punctuation_chars = [chr(i) for i in range(sys.maxunicode)
                     if unicodedata.category(chr(i)).startswith("P")]

#some special characters that appeared in germanic manuscripts, maybe should be avoided now
ctc_loss_fn = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)



vocab = ['<blank>'] + list("ÈĒĖēėěęĚĘëéèÉÊËðÐŊŋ") + list("£§êàâé£§⊥")+['£','ſ','—','“','„','’','ô','é']+ list(string.ascii_letters + string.digits + string.punctuation + " ") + ['ä', 'ö', 'ü', 'Ä', 'Ö', 'Ü', 'ß', 'å', 'Å', 'æ', 'Æ', 'ø', 'Ø']
""" 
vocab = ['<blank>'] +  punctuation_chars + list("abcdefghijklmnopqrstuvwxyz") + \
        list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + \
        list("0123456789") + \
        list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ")  # include space

 """

num_classes = len(vocab) + 1  # +1 for CTC blank
# Add CTC blank at index 0
BLANK_INDEX = 0



gc.collect()  # Python garbage collection
torch.cuda.empty_cache()  # frees cached memory (not allocated memory)

 




#---------------------------------------------------------------------------------------------------
# encode/decode characters

# Char to index (starting from 1 because 0 is reserved for blank)
char_to_idx = {char: idx + 1 for idx, char in enumerate(vocab)}

# Index to char (inverse)
idx_to_char = {idx + 1: char for idx, char in enumerate(vocab)}
idx_to_char[BLANK_INDEX] = ''  # blank is mapped to empty string

import torch

""" def merge_models(model1, model2, alpha=0.5):
    
    state1 = model1.state_dict()
    state2 =model2.state_dict()

    merged_state = {}
    for k in state1.keys():
        merged_state[k] = (alpha * state1[k] + (1 - alpha) * state2[k])

    # create a new model instance (same architecture as model1)
    
    #merged_model = CRNN(num_classes=num_classes,img_height=64,in_channels=1)

    merged_model = type(model1)(
    num_classes=num_classes,
    img_height=64,
    in_channels=1
    )
    merged_model.load_state_dict(merged_state)

    

    merged_model.load_state_dict(merged_state)
    return merged_model """


import torch
import copy


import torch

def merge_task_specific_models(models, alphas=None, merge_layer4=False):
    """
    Merge multiple CRNN models, only merging LSTM + classifier layers
    and optionally the last ResNet block (cnn.5).

    Args:
        models (list of nn.Module): list of models to merge
        alphas (list of float, optional): weights for each model, must sum to 1
        merge_layer4 (bool): if True, merge cnn.5 block as well

    Returns:
        nn.Module: merged model (modifies first model in list in-place)
    """
    assert len(models) > 1, "Need at least two models to merge"
    num_models = len(models)

    if alphas is None:
        alphas = [1.0 / num_models] * num_models
    else:
        assert len(alphas) == num_models, "Length of alphas must match number of models"
        assert abs(sum(alphas) - 1.0) < 1e-5, "Alphas must sum to 1"

    merged_model = models[0]

    with torch.no_grad():
        # loop through parameters
        for name, param in merged_model.named_parameters():
            # merge only task-specific layers
            if ("lstm" in name) or ("fc" in name) or (merge_layer4 and "cnn.5" in name):
                # weighted sum of corresponding parameters
                merged_data = sum(alpha * m.state_dict()[name].data for alpha, m in zip(alphas, models))
                param.data.copy_(merged_data)

    return merged_model


def merge_models(model1, model2, alpha=0.5):
    with torch.no_grad():
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            p1.data.copy_(alpha * p1.data + (1 - alpha) * p2.data)
    
    # free model2 if not needed anymore
    model2 = None
    clear_gpu_memory()
    
    return model1 




def decode_target(tensor, idx2char):
    return ''.join([idx2char.get(int(i), '?') for i in tensor])



def encode_text(text):
    return [char_to_idx[c] for c in text if c in char_to_idx]


def encode_text(text, char_to_idx):
    """Convert string to list of integer indices (no blank)"""
    return [char_to_idx[c] for c in text if c in char_to_idx]

def ctc_greedy_decoder_batch(preds, idx_to_char, blank=0):

    pred_texts = []
    for seq in preds:               # loop over batch
        decoded = []
        prev = blank
        for p in seq:               # loop over timesteps
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


def chunk_dataloaders(dataloaders, n_chunks=10):
    """Split dataloaders list into n_chunks groups."""
    chunk_size = math.ceil(len(dataloaders) / n_chunks)
    return [dataloaders[i:i+chunk_size] for i in range(0, len(dataloaders), chunk_size)]

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
        
       

        # to tensor
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

        # to gayscale
        if tensor.shape[1] == 3:
            gray = KC.rgb_to_grayscale(tensor)
        else:
            gray = tensor

        # avoid resizing - stretching or proportsional 
        # this part is done separatley and differently in other script
        # when resising with aspect ratio
        #gray = self._resize_proportional(gray)

        # augmentation only (if training)
        if self.train:
            gray = self.augment(gray)

        # Some conctrast
        gray = self._contrast_stretch(gray, *self.contrast_clip)

        # bit sharpening
        blurred = KF.gaussian_blur2d(gray, (3, 3), (1.0, 1.0))
        sharpened = torch.clamp(gray + self.sharpen_strength * (gray - blurred), 0.0, 1.0)

        return sharpened.squeeze(0)  # [1,H,W] -> remove batch dim for dataset __getitem__
                


class OCRDataset(Dataset):
    def __init__(self, label_file, image_dir, transform=None, dataSrc = None):

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
class DictDataset(Dataset):
    def __init__(self, datasets_dict, transform=None):
        # Flatten dictionary into list of (dataset_name, path, label)
        self.samples = []
       # for name, items in datasets_dict.items():
        for entry in datasets_dict:
            path, label = entry.split("|", 1)  # split only on first pipe
            self.samples.append((path, label))
    
        self.transform = transform

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        img_name, text = self.samples[idx]
        img_path = img_name

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
            for name, param in self.cnn.named_parameters():
                if "layer4" not in name: ## layer 4 trainiable. keep 1-3 as resnet have
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
        x = F.adaptive_avg_pool2d(x, (1, None))       # Collapse height to 1 → [B, C, 1, W]
        x = x.squeeze(2).permute(0, 2, 1) # Convert to sequence format → [B, W, C]
  
        # Pass through BiLSTMs
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        # Final prediction
        x = self.fc(x)

        return x



#---------------------------------------------------------------------------------------------------

## training / execution of model 


#init
model = CRNN(num_classes=num_classes,img_height=64,in_channels=1,freeze_cnn=True)

def ctc_greedy_decoder(preds, blank=0):
    decoded = []
    prev = blank
    for p in preds:
        if p != prev and p != blank:
            decoded.append(p.item())
        prev = p
    return decoded
 
def make_infinite(dataloader):
    while True:
        for batch in dataloader:
            yield batch


#---------------------------------------------------------------------------------------------------
# exe = True # means model is applied 



exe = False
epocs_iterat = 500


#preaparing similar widht dataloaders
import os
import glob
from PIL import Image
from collections import defaultdict
# Directory containing images
myvars = {}
dictionary = defaultdict()
dictionaryModels = defaultdict()


dictonaryLabels = defaultdict()
def iterate_over_dict():
    for key in dictionary:
        print(key, dictionary[key])

    
with open("ref.cnf") as myfile:
    for line in myfile:
        name, var = line[:-1].strip().partition("=")[::2]
        myvars[name.strip()] = var.strip()


label_file = None
dir_path  = None

""" 
if[myvars['data'] == "custom"]:
    label_file=myvars['label_file_custom']
    dir_path = myvars['imgFolder_custom']+'/*.png'
elif [myvars['data'] == "IAM64"]:
    label_file=myvars['label_file_IAM64']
    dir_path = myvars['imgFolder_IAM64']+'/*/*.png'
 """
print("preparing IAM64")    

label_file=myvars['label_file_IAM64']
dir_path = myvars['imgFolder_IAM64']+'/*/*.png'
with open(label_file, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            folder_image_name, text = line.strip().split(" ",1)
            folder, image_name =  folder_image_name.strip().split(",")
            dictonaryLabels.setdefault(folder+"/"+image_name, []).append(text)
        except:
            print(line)


listing = glob.glob(dir_path)
for file_name in listing:

    with Image.open(file_name) as img:
        path = Path(file_name)
        os.path.split(path.parent.absolute())[1]
        label = os.path.split(path.parent.absolute())[1]+"/"+ os.path.basename(path)
        label = label.replace(".png","")
        dictionary.setdefault(img.size, []).append(file_name+"|"+"".join((dictonaryLabels[label])))

#--------------------------------------------------
## process custom
label_file=myvars['label_file_custom']
dir_path = myvars['imgFolder_custom']+'/*.png'
#just to be consistent I use same approach. can be moved to common function
#but actually label file already is 'normalized' so this here might not be optimized in future. 
#if myvars['data'] == "custom":
    #mapping location of image to label for IAM64
print("preparing custom")    
with open(label_file, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            folder_image_name, text = line.strip().split("|",1)
            dictonaryLabels.setdefault(folder_image_name, []).append(text)                
        except:
            print(line)
            continue


# Group images by their dimension; tuple -> list
listing = glob.glob(dir_path)
for file_name in listing:

    # Construct full file path
    # Open and find the image size
    with Image.open(file_name) as img:
        path = Path(file_name)
        os.path.split(path.parent.absolute())[1]
        label = os.path.basename(path)
        #label = label.replace(".png","")
        #print(label)
        
        if label in dictonaryLabels:
            dictionary.setdefault(img.size, []).append(file_name+"|"+"".join((dictonaryLabels[label])))

        #print(type(img.size))
        #print(f"{file_name}: {img.size}")


#-----------------------------------------------




## process ICDAR
label_file=myvars['label_file_ICDAR']
dir_path = myvars['imgFolder_ICDAR']+'/*.png'
#just to be consistent I use same approach. can be moved to common function
#but actually label file already is 'normalized' so this here might not be optimized in future. 
#if myvars['data'] == "custom":
    #mapping location of image to label for IAM64
print("preparing ICDAR")
with open(label_file, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            folder_image_name, text = line.strip().split("|",1)
            dictonaryLabels.setdefault(folder_image_name, []).append(text)                
        except:
            #print(line)
            #print("exception")
            continue


# Group images by their dimension; tuple -> list
listing = glob.glob(dir_path)
for file_name in listing:

    # Construct full file path
    # Open and find the image size
    with Image.open(file_name) as img:
        path = Path(file_name)
        os.path.split(path.parent.absolute())[1]
        label = os.path.basename(path)
        #label = label.replace(".png","")
        #print(label)
        
        if label in dictonaryLabels:
            dictionary.setdefault(img.size, []).append(file_name+"|"+"".join((dictonaryLabels[label])))

        #print(type(img.size))
        #print(f"{file_name}: {img.size}")




print("Dataloaders")



transform = DualChannelLaplaceTransform(train=True)

# generate loaders
dataloaders  = []
for key,value in sorted(dictionary.items()): 
    #print(key)
    dataloaders.append(DataLoader(DictDataset(value,transform),   prefetch_factor=2,  batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=2,pin_memory=True, persistent_workers=True  ))
    



s = string.ascii_letters






# Define loss function
criterion = nn.CTCLoss(blank=0, zero_infinity=True)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,      # safer start
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0
)


#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Cosine annealing scheduler (smooth decay over total epochs)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=400,   # number of epochs until LR restarts (here: full run)
    eta_min=1e-6 # minimum LR at the end
)



#optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

device = torch.device("cuda")
model = model.to(device)    



import itertools
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
#train(model, dataloader, optimizer, ctc_loss, device, 2);
min_memory_available = 2 * 1024 * 1024 * 1024  # 2GB

clear_gpu_memory()
wait_until_enough_gpu_memory(min_memory_available)
#predefined_loss = 2.0
# Usage example
#infinite_loaders = [make_infinite(dl) for dl in dataloader_groups]

from torch.utils.data import ConcatDataset, DataLoader

print("training")
scaler = torch.cuda.amp.GradScaler()   # put this before training loop
#res1 = None
#res2 = None
countr = 0
objectSaved = False
res = None
best_models = []
for batch in  itertools.chain(*dataloaders):
    if countr > 0:
       best_models.append(dictionaryModels[res]) 
    if len(best_models)==2:
        #merge
        #save
        #leave in list one merged element as single element
        
        print(dictionaryModels[res+"_path"])

        #model = merge_models(best_models[0],best_models[1], alpha=0.8)
        merged_model = merge_task_specific_models([best_models[0], best_models[1]], alphas=[0.5, 0.5], merge_layer4=True)
        best_models.clear()

        best_models.append(merged_model)
        print("epocs",epoch+1)
        
        #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # reset optimizer
        if countr % 1000 == 0:
            torch.save(best_models[0], dictionaryModels[res+"_path"])
        dictionaryModels.clear()
    
    random_level2 = random.choices(s, k=5)
    res = ''.join(random_level2)

    model.train()
    best_val_loss = float('inf')
    
    countr+=1
    print(countr)   
    for epoch in range(epocs_iterat):
        total_loss = 0.0
        num_batches = 0

        images, targets,input_lengths, target_lengths  = batch

        if images is None or targets is None:
            print("⚠️ Skipping None batch")
            continue

        images = images.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=False):  
            logits = model(images.float())   # full forward in FP32 
        with torch.cuda.amp.autocast():  # 🔹 mixed precision zone
            # Forward
            #logits = model(images)              # (B, T, C)
            log_probs = F.log_softmax(logits, dim=2)
            log_probs = log_probs.permute(1, 0, 2)  # (T, B, C) for CTC

            B = log_probs.size(1)
            T = log_probs.size(0)
            input_lengths = torch.full((B,), T, dtype=torch.long, device=device)

            # Safety: skip batch if any target too long
            if (target_lengths > input_lengths).any():
                print(f"⚠️ Skipping batch: target longer than input T={T}")
                continue

            # Compute CTC loss
            loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)

        # Skip invalid loss
        if torch.isnan(loss) or torch.isinf(loss):
            print("⚠️ Skipping batch due to invalid loss")
            continue

        # Backward with scaler
        torch.cuda.empty_cache()

        loss.backward()
        optimizer.step()
        #scaler.update()

        total_loss += loss.item()
        num_batches += 1

        # Average loss per epoch
        avg_loss = total_loss / max(1, num_batches)
        print(f"Epoch {epoch+1}: avg_loss = {avg_loss}")

        # Save best model
        if avg_loss < best_val_loss :
            best_val_loss = avg_loss
            print("FOUND new best losss value")
            print(f"models/crnn_ctc_model_epoch{res+"_"+str(epoch+1)}.pth")
            model_path = f"models/crnn_ctc_model_epoch{res+"_"+str(countr)+"_"+str(epoch+1)}.pth"
            #best_models[0]= ((model.state_dict(), model_path))  ## this should replace with each new one
            dictionaryModels[res]=model
            dictionaryModels[res+"_path"]=model_path
            #dictionaryModels.setdefault(res, []).append(model.state_dict())
            #dictionaryModels.setdefault(res+"_path", []).append(model_path)
            modelInTraining = res
            #torch.save(model.state_dict(), model_path)
            #print(f"✅ Saved new best model: {model_path}")
    
   
