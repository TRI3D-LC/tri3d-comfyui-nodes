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

def run(img):
    cloth_seg = generate_mask(img, net=net, device=device)
    return cloth_seg
