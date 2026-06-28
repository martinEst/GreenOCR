import cv2
import glob
import sys
import os
for file in glob.glob("/home/martinez/TUS/DISSERT/data/randomSampleImages/*__e.png"):

    # Paths
    input_path = file
    out_image_path = "/home/martinez/GITWORLD/One-DM/data/IAM64-new/train/william/"+os.path.basename(file)
    out_image_path_ref = "/home/martinez/GITWORLD/One-DM/ref/train/william/"+os.path.basename(file)
    out_laplace_path = "/home/martinez/GITWORLD/One-DM/data/IAM64_laplace/train/william/"+os.path.basename(file)



    # Load image (BGR)
    img = cv2.imread(input_path)

    # Resize to match dataset
    target_size = (128, 64)  # (width, height)
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    # Save resized image
    cv2.imwrite(out_image_path, img_resized)
    cv2.imwrite(out_image_path_ref, img_resized)


    # Convert to grayscale for Laplacian
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # Laplacian filter
    laplace = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    laplace = cv2.convertScaleAbs(laplace)

    # Save Laplacian
    cv2.imwrite(out_laplace_path, laplace)

    print("Saved resized image and Laplacian at 128×64")
