from realesrgan import RealESRGANer
from PIL import Image
import torch
import os
import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet

import cv2
import numpy as np
trainingDirectory = '/home/martinez/TUS/DISSERT/data/latest/ICD_DATA/prepareForTraining/allImg'
targetLabelFile = '/home/martinez/TUS/DISSERT/data/latest/ICD_DATA/prepareForTraining/targets.txt'
fileTranscriptionFolder = '/home/martinez/TUS/DISSERT/data/latest/ICD_DATA/Transcriptions'
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import gc
import time

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



def resize_keep_aspect(img, target_h=64):

    h, w = img.shape[:2]

    # Compute new width maintaining aspect ratio
    new_w = int(w * (target_h / h))
    resized = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

    return resized



min_memory_available = 2 * 1024 * 1024 * 1024  # 2GB

# Example:
import glob
import random
import string
import os.path
from pathlib import Path
s = string.ascii_letters

random_level2 = random.choices(s, k=5)
res = ''.join(random_level2)    

for file in glob.glob("/home/martinez/TUS/DISSERT/data/latest/ICD_DATA/Lines/*.png"):
    
    
    #if os.path.isfile(trainingDirectory+"/" + os.path.basename(file)):
         #continue
    
    clear_gpu_memory()
    wait_until_enough_gpu_memory(min_memory_available)
    
    # Step 1: Load image (OpenCV loads in BGR format)
    img = cv2.imread(file, cv2.IMREAD_COLOR)

    # Step 2: Create the model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

    # Step 3: Create the RealESRGANer instance
    upsampler = RealESRGANer(
        scale=4,
        model_path='/home/martinez/d/weights/RealESRGAN_x4plus.pth', 
        model=model,
        tile=0,  # Set to >0 for tiled inference on large images
        tile_pad=10,
        pre_pad=0,
        half=False  # Set True if using half-precision and CUDA
    )

    # Step 4: Super-resolve
    torch.cuda.empty_cache()
    print("running file ",file)
    #output, _ = upsampler.enhance(img, outscale=2)
    print("Original shape:", img.shape)
    #print("Upscaled shape:", output.shape)
    #target_w, target_h = 32, 64
    #downscaled = cv2.resize(output, (target_w, target_h), interpolation=cv2.INTER_AREA)
    output_proportional = resize_keep_aspect(img, target_h=64)
    newFileName = trainingDirectory+"/" + res+"_"+os.path.basename(file); 
    #read transcription file 
    label = (Path(fileTranscriptionFolder) /os.path.basename(file).replace(".png",".txt")).read_text()
    
    # Step 5: Save result
    cv2.imwrite(newFileName, output_proportional)

    #step 6 add record to targets text file
    with open(targetLabelFile, 'a') as fd: 
        fd.write(f'{ os.path.basename(newFileName)+"|"+label}\n')
    

