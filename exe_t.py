import os
import gc
import cv2
import torch
import glob


from src.config import VOCAB
from src.transforms import DualChannelLaplaceTransform
from src.model import CRNN
from src.utils import ctc_greedy_decoder, resize_keep_aspect
# Import your clean modular building blocks
from src.config import VOCAB
from src.transforms import DualChannelLaplaceTransform
from src.model import CRNN
from src.utils import ctc_greedy_decoder, resize_keep_aspect, clear_gpu_memory

#basicsr
from basicsr.archs.rrdbnet_arch import RRDBNet

#realesrgan
from realesrgan import RealESRGANer

#local src
from src.config import VOCAB





# Hardware & Memory Settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

# this is for my super weak GPU optimization
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True       # important for varying image widths!
torch.autograd.set_detect_anomaly(False)    # True when debugging NaNs



# Cleaning up VRAM before loading the model
gc.collect()  # Python garbage collection
torch.cuda.empty_cache()  # frees cached memory (not allocated memory)





# 1. Parse your configuration file paths
myvars = {}
with open("ref.cnf", "r", encoding="utf-8") as myfile:
    for line in myfile:
        if "=" in line:
            name, var = line.strip().split("=", 1)
            myvars[name.strip()] = var.strip()

# 2. Map standard global indices
char_to_idx = {char: idx for idx, char in enumerate(VOCAB)}
idx_to_char = {idx: char for idx, char in enumerate(VOCAB)}
num_classes = len(VOCAB)

# 3. Instantiate your core model framework shell once
# (Using the identical class definition that training utilizes)
model_shell = CRNN(num_classes=num_classes, img_height=64, in_channels=1, freeze_cnn=True)
model_shell = model_shell.to("cuda")

# 4. Initialize the heavy Real-ESRGAN upsampler ONCE outside the loop
# This completely stops your home GPU VRAM from leaking and crashing!
try:
    # Ensure external dependencies are accessible locally
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    
    esrgan_network = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path='/home/martinez/d/weights/RealESRGAN_x4plus.pth', 
        model=esrgan_network,
        tile=0,  
        tile_pad=10,
        pre_pad=0,
        half=False  # Switch to True if your GPU supports half-precision FP16
    )
    use_upscaler = True
    print("Real-ESRGAN Super-Resolution engine successfully loaded.")
except ImportError:
    use_upscaler = False
    print("Warning: Real-ESRGAN libraries missing. Proceeding with standard interpolation scaling.")

# 5. Initialize evaluation transform rules
inference_transform = DualChannelLaplaceTransform(train=False)

# -------------------------------------------------------------------------
# Execution Cycle
# -------------------------------------------------------------------------
# Find all trained checkpoint paths saved inside your models directory
checkpoint_files = sorted(glob.glob("models/*.pth"))
test_image_paths = glob.glob(os.path.join(myvars['imgTest'], "*.png"))

for checkpoint_path in checkpoint_files:
    print(f"\n--- Evaluating Model Checkpoint: {checkpoint_path} ---")
    
    # SAFE WEIGHT LOADING: Load weights safely directly into the active shell
    try:
        # If your old checkpoints saved the entire model object, we use this fallback
        checkpoint_data = torch.load(checkpoint_path, map_location="cuda")
        if isinstance(checkpoint_data, dict):
            # Normal weights state dict
            model_shell.load_state_dict(checkpoint_data)
        else:
            # Legacy object fallback
            model_shell.load_state_dict(checkpoint_data.state_dict())
    except Exception as error:
        print(f"Skipping model {checkpoint_path} due to directory path mismatch structure: {error}")
        continue
        
    model_shell.eval() # Set model to validation evaluation mode
    
    # CRITICAL PERFORMANCE FIX: Freeze gradient calculations entirely during inference!
    with torch.no_grad():
        for file in test_image_paths:
            print(f"Processing Image: {file}")

            # Step 1: Load Raw input asset (OpenCV reads in native BGR format)
            img = cv2.imread(file, cv2.IMREAD_COLOR)
            if img is None:
                continue

            # Step 2: Super-resolve image via the global instance
            if use_upscaler:
                output, _ = upsampler.enhance(img, outscale=1)
            else:
                output = img.copy()

            # Step 3: Resize height proportionally down to 64 pixels 
            output_proportional = resize_keep_aspect(output, target_h=64)

            # Step 4: Convert 3-channel colour space down to clean Grayscale
            gray = cv2.cvtColor(output_proportional, cv2.COLOR_BGR2GRAY)

            # Step 5: Convert grayscale array to a normalized float array [0, 1]
            # Feed raw normalized array directly to your clean Kornia transform block
            img_tensor = inference_transform(gray, device="cuda") # Output shape: [1, 64, Width]
            
            # Add back the singular global batch dimension required by the CRNN model layer
            img_tensor = img_tensor.unsqueeze(0) # Ultimate shape matching model: [1, 1, 64, Width]

            # Step 6: Generate OCR Sequence Predictions
            predictions = model_shell(img_tensor) # Output matrix shape: [1, OutputSequenceLength, NumClasses]
            
            # Step 7: Collapse matrix selections via argmax along the class axis
            preds_indices = predictions.argmax(dim=2) # Shape: [1, OutputSequenceLength]
            
            # Step 8: Parse character positions using our safe greedy decoder logic
            decoded_indices = ctc_greedy_decoder(preds_indices[0])
            
            # Step 9: Reconstruct string data from map dictionary indices
            ocr_text = "".join(idx_to_char.get(idx, '?') for idx in decoded_indices)
            
            # Step 10: Clean trailing layouts or sentence spaces
            cleaned_text = " ".join(ocr_text.split())
            print(f"Resulting Text Output: {cleaned_text}")
            
    clear_gpu_memory() # Clean your VRAM cache right after running predictions for the model