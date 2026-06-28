import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class OCRDataset(Dataset):
    def __init__(self, label_file, image_dir, char_to_idx, transform=None, dataSrc=None):
        self.image_dir = image_dir
        self.transform = transform
        self.char_to_idx = char_to_idx
        self.samples = []

        if dataSrc == "En":
            with open(label_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        folder_image_name, text = line.strip().split(" ", 1)
                        folder, image_name = folder_image_name.strip().split(",")
                        self.samples.append((os.path.join(folder, f"{image_name}.png"), text))
                    except Exception:
                        continue
        else: 
            with open(label_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        image_name, text = line.strip().split("|")
                        self.samples.append((image_name, text))
                    except Exception:
                        continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, text = self.samples[idx]
        img_path = os.path.join(self.image_dir, img_name)

        img_pil = Image.open(img_path).convert("L")
        img_np = np.array(img_pil, dtype=np.float32) / 255.0  
        
        if self.transform:
            img_tensor = self.transform(img_np)
        else:
            img_tensor = torch.from_numpy(img_np).unsqueeze(0)  

        w = img_tensor.shape[-1]
        label_tensor = torch.tensor(
            [self.char_to_idx[c] for c in text if c in self.char_to_idx],
            dtype=torch.long
        )
        return img_tensor, label_tensor, w

#google ai
def collate_fn(batch, downsample_factor=32):
    """
    Combines individual data samples into a single uniform batch for training.
    Handles dynamic right-side padding and converts widths into model sequence steps.
    """
    # 1. Unzip the batch tuple into separate lists of tensors
    # Expected dimensions per image tensor in the batch: [Channels (1), Height (64), Width]
    images, texts, widths = zip(*batch)
    
    # 2. Find the widest image in this specific batch to set our padding target
    max_w = max(widths)
    
    padded_imgs = []
    input_lengths = []
    
    # 3. Process each sample to align them perfectly to the max width
    for img in images:
        # Calculate how many pixels of padding this specific image needs on the right side
        # img.shape[2] accesses the width dimension of the [1, 64, Width] tensor
        pad_w = max_w - img.shape[2]
        
        # Pad only the right side (0 left, pad_w right, 0 top, 0 bottom)
        # We use value=1.0 because 1.0 represents a clean white background in normalized tensors
        padded = torch.nn.functional.pad(img, (0, pad_w, 0, 0), value=1.0)
        padded_imgs.append(padded)
        
        # 4. MATH CRITICAL TO CTC LOSS: Calculate the timeline steps outputted by the CNN.
        # Your ResNet-50 backbone downsamples the raw width dimension by exactly 32x.
        # We divide the RAW width by 32 so PyTorch's CTCLoss knows the true length of the model's output array.
        network_seq_len = img.shape[2] // downsample_factor
        input_lengths.append(network_seq_len)
        
    # 5. Pack the target text metrics into long tensors for PyTorch's CTCLoss
    # target_lengths tracks how many total characters are inside each label string
    target_lengths = torch.tensor([len(label) for label in texts], dtype=torch.long)
    
    # Flatten the list of individual label index tensors into one single continuous 1D tensor
    targets = torch.cat(texts)
    
    # Convert our calculated network timeline step counts into a tensor
    input_lengths_tensor = torch.tensor(input_lengths, dtype=torch.long)

    # 6. Stack the padded images into a unified 4D tensor: [BatchSize, Channels, Height, MaxWidth]
    return torch.stack(padded_imgs), targets, input_lengths_tensor, target_lengths
