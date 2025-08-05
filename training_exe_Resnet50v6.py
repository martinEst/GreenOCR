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

class DualChannelLaplaceTransform:
    def __init__(self, target_height=64, max_width=2048, train=True, sharpen_strength=1.5, contrast_clip=(0.01, 0.99)):
        self.target_height = target_height
        self.max_width = max_width
        self.train = train
        self.sharpen_strength = sharpen_strength
        self.contrast_clip = contrast_clip
        self.to_tensor = T.ToTensor()

        # Augmentations only used during training
        self.augment = nn.Sequential(
            KA.RandomAffine(degrees=2.5, translate=(0.02, 0.02), p=0.5),
            KA.RandomBrightness(0.1, p=0.5),
            KA.RandomContrast(0.1, p=0.5),
            KA.RandomGaussianNoise(mean=0.0, std=0.01, p=0.2),
        )

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

    def __call__(self, img):
        # Step 1: Convert input to tensor [1, C, H, W]
        if isinstance(img, Image.Image):
            tensor = self.to_tensor(img).unsqueeze(0)
        elif isinstance(img, np.ndarray):
            if img.ndim == 2:  # grayscale
                img = np.expand_dims(img, axis=0)
                tensor = torch.from_numpy(img).float().unsqueeze(0)
            elif img.ndim == 3:  # HWC
                tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
            else:
                raise ValueError(f"Unexpected ndarray shape: {img.shape}")
        elif isinstance(img, torch.Tensor):
            if img.ndim == 3:  # CHW
                tensor = img.unsqueeze(0)
            elif img.ndim == 4:
                tensor = img
            else:
                raise ValueError(f"Unexpected tensor shape: {img.shape}")
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

        # Step 2: Grayscale if RGB
        if tensor.shape[1] == 3:
            gray = KC.rgb_to_grayscale(tensor)
        else:
            gray = tensor

        # Step 3: Resize proportionally
        gray = self._resize_proportional(gray)

        # Step 4: Augmentation (training only)
        if self.train:
            gray = self.augment(gray)

        # Step 5: Contrast stretch
        gray = self._contrast_stretch(gray, *self.contrast_clip)

        # Step 6: Sharpen
        blurred = KF.gaussian_blur2d(gray, (3, 3), (1.0, 1.0))
        sharpened = torch.clamp(gray + self.sharpen_strength * (gray - blurred), 0.0, 1.0)

        # Step 7: Laplacian
        laplace = KF.laplacian(sharpened, kernel_size=3)

        # Step 8: Stack → 2 channels
        #        combined = torch.cat([sharpened.squeeze(0), laplace.squeeze(0)], dim=0)
        sharpened_1ch = sharpened[:, 0:1, :, :]  # [B,1,H,W]
        laplace_1ch   = laplace[:, 0:1, :, :]    # [B,1,H,W]

        # Stack into 2 channels
        combined = torch.cat([sharpened_1ch.squeeze(0),
                              laplace_1ch.squeeze(0)], dim=0)  # [2,H,W]
        return combined


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



#---------------------------------------------------------------------------------------------------
#before feeding image into model, maybe slightly modify it ? 

def preprocess_image_safe(img_path, target_size=(2048, 32)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Slight denoise, but not aggressive
    img = cv2.bilateralFilter(img, 9, 75, 75)


    
    # Convert `a` to a PIL Image
    # 2. Denoising (Median + Bilateral)
    img = cv2.medianBlur(img, 3)
    print("c Image shape after resize:", img.shape)


    # 3. Contrast Enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)






    # Resize
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

    # Normalize to [-1, 1]
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5

    # Convert to tensor
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().cuda()

    return img_tensor


#---------------------------------------------------------------------------------------------------

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



def ctc_collate_fn(batch):
    images, labels = zip(*batch)
    
    # Pad image widths to the max width in batch
    max_w = max(img.shape[-1] for img in images)
    padded_images = []
    for img in images:
        if img.dim() == 4 and img.size(0) == 1:
            img = img.squeeze(0)  # remove fake batch dim if exists
        c, h, w = img.shape  # support multi-channel: c can be 2, 3, 4...
        pad_w = max_w - w
        padded = torch.nn.functional.pad(img, (0, pad_w), value=0)
        padded_images.append(padded)
    
    images_tensor = torch.stack(padded_images)  # [B, 1, 32, max_W]

    # Flatten labels into 1D target for CTC
    targets = torch.cat(labels)
    input_lengths = torch.tensor([img.shape[-1] // 4 for img in padded_images], dtype=torch.long)
    target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)

    return images_tensor, targets, input_lengths, target_lengths





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

class OCRDataset(Dataset):
    def __init__(self, label_file, image_dir, laplace_dir=None, transform=None, target_size=(2048, 32),
                 kornia_augmenter=None, return_type='tensor'):
        """
        Args:
            laplace_dir: Directory with corresponding Laplacian images (optional)
        """
        self.target_size = target_size  # (W, H)
        self.char_to_idx = char_to_idx
        self.image_dir = image_dir
        self.laplace_dir = laplace_dir
        self.samples = []
        self.transform = transform
        self.kornia_augmenter = kornia_augmenter

        with open(label_file, 'r', encoding='utf-8') as f:
            try:
                for line in f:
                    image_name, text = line.strip().split("|")
                    self.samples.append((image_name, text))
            except RuntimeError as e:
                print(f" Error  batch due to error: {e}")
                

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, text = self.samples[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Load grayscale image
        img_pil = Image.open(img_path).convert("L")
        img_pil = img_pil.resize(self.target_size, Image.BICUBIC)
        image_np = np.array(img_pil, dtype=np.float32) #/ 255.0  # shape: (H, W)

        # Load Laplacian image if available
        if self.laplace_dir:
            lap_path = os.path.join(self.laplace_dir, img_name)
            lap_pil = Image.open(lap_path).convert("L")
            lap_pil = lap_pil.resize(self.target_size, Image.BICUBIC)
            laplace_np = np.array(lap_pil, dtype=np.float32) #/ 255.0  # shape: (H, W)
        else:
            laplace_np = np.zeros_like(image_np, dtype=np.float32)

        # Stack grayscale image and laplacian as channels: [2, H, W]
        combined_np = np.stack([image_np, laplace_np], axis=0)
        combined_tensor = torch.from_numpy(combined_np).float()  # [2, H, W]

        # Kornia augmentations expect [B, C, H, W]
        if self.kornia_augmenter:
            combined_tensor = self.kornia_augmenter(combined_tensor.unsqueeze(0))  # [1, 2, H, W]
            combined_tensor = combined_tensor.squeeze(0)  # back to [2, H, W]

        # Optional PyTorch transform (e.g., normalization)
        #print("Before transform:", combined_tensor.shape)
        if self.transform:
            combined_tensor = self.transform(combined_tensor)
        #print("After transform:", combined_tensor.shape)
        # Encode label
        label = self.encode_text(text)
        return combined_tensor, label

    def encode_text(self, text):
        # Converts a string to a tensor of character indices
        return torch.tensor([self.char_to_idx[c] for c in text], dtype=torch.long)


class CRNN(nn.Module):
    def __init__(self, num_classes, img_height=64, img_width=2048, in_channels=2, freeze_cnn=False):
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
            self.feature_dim = c #self.feature_dim = c * h #with pooling h collapse to 1
        """ 
        self.lstm = nn.Sequential(
            nn.LSTM(self.feature_dim, 256, bidirectional=True, batch_first=True),
            nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        ) """
        self.lstm1 = nn.LSTM(self.feature_dim, 256, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)


        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        
        x = self.normalize(x)  # <-- Use the transforms.Normalize object
        x = self.cnn(x)

        batch_size, c, h, w = x.size()
        # Collapse the height dimension to 1 to make it sequence-friendly
        x = F.adaptive_avg_pool2d(x, (1, None))  # shape: [B, C, 1, W]
        x = x.squeeze(2).permute(0, 2, 1)        # shape: [B, W, C]

        #x = x.permute(0, 3, 1, 2).reshape(batch_size, w, c * h)
        #x, _ = self.lstm(x)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.fc(x)

        return F.log_softmax(x, dim=2)




#
##--------------------------------------------------------
## if using Kornia augmentation 
##--------------------------------------------------------

""" class CombinedTransform:
    def __init__(self, apply_enhancement=True):
        self.apply_enhancement = apply_enhancement

    def __call__(self, img):
        img = img.convert("L")  # ensure grayscale

        if self.apply_enhancement:
            img = img.convert("L")  # force grayscale
            img = TF.adjust_contrast(img, 32.0)
            img = TF.adjust_brightness(img, 21.6)
            img = TF.adjust_sharpness(img, 2.0)
            img = TF.equalize(img)

        img = TF.resize(img, (32, 2048))
        img = TF.to_tensor(img)
        img = TF.normalize(img, (0.5,), (0.5,))
        return img

 """
""" class CombinedTransform:
    def __init__(self, img_height=32):
        self.img_height = img_height
        self.to_tensor = transforms.ToTensor()
        self.augmenter = OCRKorniaAugmentations()
        self.normalize = transforms.Normalize(mean=[0.5], std=[0.5])  # example normalization

    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        w, h = image.size
        new_w = int(self.img_height * (w / h))
        image = image.resize((new_w, self.img_height), Image.BILINEAR)
        x = self.to_tensor(image)  # PIL → Tensor
        x = self.augmenter(x.unsqueeze(0))  # Kornia expects [B,C,H,W]
        x = x.squeeze(0)  # Remove batch dimension
        x = self.normalize(x)
        return x
    
 """
""" 
class OCRKorniaAugmentations(nn.Module):
    def __init__(self):
        super().__init__()
        self.kornia_augmenter = AugmentationSequential(
            K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.5),
            K.RandomAffine(degrees=5, scale=(0.9, 1.1), shear=5, p=0.5),
            K.RandomMotionBlur(kernel_size=3, angle=10.0, direction=0.5, p=0.5),
            K.RandomErasing(scale=(0.02, 0.2), ratio=(0.3, 3.3), p=0.3),
            data_keys=["input"],
            same_on_batch=False
        )

    def forward(self, x):
        return self.kornia_augmenter(x)
 """
  
#---------------------------------------------------------------------------------------------------

## training / execution of model 


#init
model = CRNN(num_classes=num_classes,img_height=64,img_width=2048,in_channels=2)




#---------------------------------------------------------------------------------------------------
# exe = True # means model is applied 

#---------------------------------------------------------------------------------------------------
class EnhanceOCRImage:
    def __call__(self, img):
        # Ensure grayscale
        img = img.convert('L')

        # Apply contrast and brightness  us
         # img = TF.invert(img)

        img = TF.adjust_contrast(img, 1.5)
        img = TF.adjust_brightness(img, 1.2)
        img = TF.adjust_sharpness(img, 1.8)
        img = TF.equalize(img)
        
        # Optional: apply mild sharpening
        img = img.filter(ImageFilter.SHARPEN)

        # Optional: histogram equalization (can help with faded text)
        img = TF.equalize(img)

        return img

exe = False
epocs_iterat = 10


if(exe):
    #transform = DualChannelLaplaceTransform(resize=(32, 2048))
    transform = DualChannelLaplaceTransform(train=False)
   # optimizer = torch.optim.Adam(model.parameters())  # Now includes RNN and FC
    """     dataloader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=ctc_collate_fn) """
    
    state = model.load_state_dict(torch.load("/home/martinez/TUS/DISSERT/models/crnn_ctc_model0308a_70_epocs.pth"))  # or your path
    model = model.to("cuda")
    print("state keys:")

    
    image = Image.open('/home/martinez/TUS/DISSERT/data/customImages/b06-045-02-02.png').convert('RGB')  # grayscale
    img_tensor = transform(image).unsqueeze(0).cuda() 
    from torchvision.transforms.functional import to_pil_image

    img_single = img_tensor[0]

    # View each channel separately
    to_pil_image(img_single[0]).show(title="Channel 0 (Sharpened)")
    to_pil_image(img_single[1]).show(title="Channel 1 (Laplace)")
    # img_tensor = preprocess_image_safe('/home/martinez/TUS/DISSERT/data/customImages/random.png')
   


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




    """   val_dataset = OCRDataset(
            label_file="data/annotations/matchingLabelFile.txt",
            image_dir="data/merged_data_python/",
            transform=standard_transform,
            kornia_augmenter=None  # no augmentation for val/test
        ) """
    #transform = CombinedTransform(apply_enhancement=True)
    transform = DualChannelLaplaceTransform(train=True)
   # kornia_augmenter = OCRKorniaAugmentations()
 
    train_dataset = OCRDataset(
    label_file="data/latest/matchingLabelFileV2.txt",
    image_dir="data/latest/images_all/",
    laplace_dir="data/latest/lapace_all/",
    transform=transform
)


    dataloader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=ctc_collate_fn,
        num_workers=4,       # Suggestion: speed up data loading
        pin_memory=True      # Optional: improves transfer to GPU if using CUDA
    )

    # Define loss function
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)  


    images, targets, input_lengths, target_lengths = next(iter(dataloader))

    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    device = torch.device("cuda")
    model = model.to(device)    

    
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
            torch.save(model.state_dict(), "/home/martinez/TUS/DISSERT/models/crnn_ctc_model0308a_"+str(epocs_iterat)+"_ep_"+str(epoch)+".pth")
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

