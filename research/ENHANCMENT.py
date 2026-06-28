from realesrgan import RealESRGANer
from PIL import Image
import torch

import cv2
#from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# Step 1: Load image (OpenCV loads in BGR format)
img = cv2.imread("/home/martinez/Prioly/abc.jpeg", cv2.IMREAD_COLOR)

# Step 2: Create the model
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

# Step 3: Create the RealESRGANer instance
upsampler = RealESRGANer(
    scale=4,
    model_path='/home/martinez/d/weights/RealESRGAN_x4plus.pth',  # ✅ Make sure this .pth file exists in the working dir
    model=model,
    tile=0,  # Set to >0 for tiled inference on large images
    tile_pad=10,
    pre_pad=0,
    half=False  # Set True if using half-precision and CUDA
)

# Step 4: Super-resolve
output, _ = upsampler.enhance(img, outscale=8)
print("Original shape:", img.shape)
print("Upscaled shape:", output.shape)
# Step 5: Save result
cv2.imwrite("/home/martinez/Prioly/eab.jpeg", output)

""" 
# Load image
img = Image.open("/home/martinez/TUS/DISSERT/data/customImages/").convert("RGB")

# Load model
model = RealESRGANer(torch.device('cuda' if torch.cuda.is_available() else 'cpu'), scale=4)
model.load_weights('/home/martinez/d/weights/RealESRGAN_x4plus.pth')  # download this first from the official repo

# Enhance
sr_image = model.predict(img)

# Save
sr_image.save("/tmp/enhanced_output.png")
 """