import PIL
import torch
import os
from process import load_seg_model, get_palette, generate_mask



device = 'cuda'

def initialize_and_load_models():
    checkpoint_path = 'model/cloth_segm.pth'
    net = load_seg_model(checkpoint_path, device=device)    
    return net

net = initialize_and_load_models()

def run(img, image_id, output_dir):
    palette = get_palette(4)
    mask0, mask1, mask2, cloth_seg = generate_mask(img, net=net, device=device, image_id=image_id, output_dir=output_dir)
    return mask0, mask1, mask2, cloth_seg

INPUT_PATH = "./input/"
OUTPUT_PATH = "./output/"

import os 
mask0_paths = []
mask1_paths = []
mask2_paths = []
cloth_paths = []

for idx, cur_image in enumerate(os.listdir(INPUT_PATH)):
    img = PIL.Image.open(INPUT_PATH + cur_image)
    mask0, mask1, mask2, cloth_seg = run(img, image_id=idx, output_dir=OUTPUT_PATH)
    # Save masks and cloth_seg with unique names (already saved in generate_mask)
    mask0_path = os.path.join(OUTPUT_PATH, f"{idx}__mask0.png")
    mask1_path = os.path.join(OUTPUT_PATH, f"{idx}__mask1.png")
    mask2_path = os.path.join(OUTPUT_PATH, f"{idx}__mask2.png")
    cloth_path = os.path.join(OUTPUT_PATH, f"{idx}__extracted_garment.png")
    mask0_paths.append(mask0_path)
    mask1_paths.append(mask1_path)
    mask2_paths.append(mask2_path)
    cloth_paths.append(cloth_path)

print("Mask0 batch:", mask0_paths)
print("Mask1 batch:", mask1_paths)
print("Mask2 batch:", mask2_paths)
print("Garment batch:", cloth_paths)
