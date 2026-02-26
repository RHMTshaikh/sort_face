import cv2 as cv
import os
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener

import paths

# This allows PIL to recognize .heic files
register_heif_opener()

def load_heic_to_cv2(path):
    # Load the image using PIL
    image = Image.open(path)
    
    # Convert PIL image to a NumPy array
    # Note: PIL uses RGB, OpenCV uses BGR
    image_np = np.array(image)
    
    # Convert RGB to BGR for OpenCV compatibility
    image_bgr = cv.cvtColor(image_np, cv.COLOR_RGB2BGR)
    
    return image_bgr


def resize(input_dir, output_dir, f):
    IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.heic')
    
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(IMAGE_EXTENSIONS):
            continue
            
        path = os.path.join(input_dir, filename)
        
        if filename.lower().endswith('.heic'):
            img = load_heic_to_cv2(path)
        else:
            img = cv.imread(path)
        
        if img is None:
            print(f"Failed to load image: {path}")
            continue
            
        H, W, _ = img.shape
        
        img_resize = cv.resize(img, (W//f, H//f))
        
        # cv.imshow('resize', img_resize)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        # break

        # save the resized image in jpg format to the output directory
        filename_without_ext = os.path.splitext(filename)[0]
        output_filename = f"{filename_without_ext}.jpg"
        
        cv.imwrite(os.path.join(output_dir, output_filename), img_resize)
        
INPUT_DIR = paths.INPUT_DIR # Folder containing the original photos
last_folder_name = os.path.basename(INPUT_DIR)
parent_dir = os.path.dirname(INPUT_DIR)
OUTPUT_DIR = os.path.join(parent_dir, f"{last_folder_name}_resized")

f = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)

resize(INPUT_DIR, OUTPUT_DIR, f)