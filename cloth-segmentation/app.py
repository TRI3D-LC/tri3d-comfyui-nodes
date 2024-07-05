import PIL
import cv2
import torch
import os
from process import load_seg_model, get_palette, generate_mask

device = 'cuda'


def initialize_and_load_models():
    checkpoint_path = 'model/cloth_segm.pth'
    net = load_seg_model(checkpoint_path, device=device)
    return net


net = initialize_and_load_models()


def run(img):
    palette = get_palette(4)
    cloth_seg = generate_mask(img, net=net, device=device)
    return cloth_seg


INPUT_PATH = "./input/"
OUTPUT_PATH = "./output/"

import os
for cur_image in os.listdir(INPUT_PATH):
    img = PIL.Image.open(INPUT_PATH + cur_image)
    cloth_seg = run(img)

    cv2.imwrite(OUTPUT_PATH + cur_image,
                cv2.cvtColor(src=cloth_seg, code=cv2.COLOR_RGB2BGR))
    # cloth_seg.save(OUTPUT_PATH + cur_image, format="PNG")
