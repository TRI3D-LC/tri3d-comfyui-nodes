#!/usr/bin/python3

from PIL import Image, ImageOps, ImageSequence, ImageFile
from PIL.PngImagePlugin import PngInfo
import cv2
import hashlib
import json
import logging
import math
import numpy as np
import os
import random
import safetensors.torch
import sys
import time
import torch
import traceback


def load_image(path):

    return torch.from_numpy(cv2.imread(
        path, cv2.IMREAD_COLOR)).to(dtype=torch.float32) / 255.0


def do_stack(img1, img2):

    dim = max(max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1])

    out = torch.zeros((dim, dim, 3), dtype=img1.dtype, device=img1.device) + 1

    diff1 = (out.shape[0] - img1.shape[0]) // 2
    diff2 = (out.shape[0] - img2.shape[0]) // 2

    part0 = 0
    part1 = img1.shape[1]
    part2 = img2.shape[1] + img1.shape[1]

    out[diff1:diff1 + img1.shape[0], part0:part1, :] = img1
    out[diff2:diff2 + img2.shape[0], part1:part2, :] = img2

    return out


def save_image(image, outpath):

    cv2.imwrite(outpath,
                (image * 255).to(dtype=torch.uint8).detach().cpu().numpy())


class H_Stack_Images:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_L": ("IMAGE", ),
                "image_R": ("IMAGE", ),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "test"
    CATEGORY = "TRI3D"

    def test(self, image_L, image_R):

        return (do_stack(img1=image_L[0], img2=image_R[0]).unsqueeze(0), )


class SaveImage_absolute:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "The images to save."
                }),
                "absolute_filename": ("STRING", {
                    "default":
                    "image.png",
                    "tooltip":
                    "The absolute path to the file to save."
                })
            },
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("text to control order", )
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "image"
    DESCRIPTION = "Saves the input images to an absolute path."

    def save_images(self, images, absolute_filename):
        i = 255.0 * images[0].cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        img.save(absolute_filename)

        return (absolute_filename, )


class SaveText_absolute:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": True,
                    "tooltip": "Text to be saved to the file."
                }),
                "absolute_filename": ("STRING", {
                    "default":
                    "image.txt",
                    "tooltip":
                    "The absolute path to the file to save."
                })
            },
            "optional": {
                "text_opt": ("STRING", {
                    "multiline":
                    True,
                    "dynamicPrompts":
                    True,
                    "tooltip":
                    "Text to provide order when necessary (to create work files after txt files)."
                }),
            }
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("same text as input", )
    FUNCTION = "save_text"

    OUTPUT_NODE = True

    CATEGORY = "text"
    DESCRIPTION = "Saves the input text to an absolute path."

    def save_text(self, text, absolute_filename, text_opt=''):
        open(absolute_filename, "w").write(text)
        return (text, )


class Wait_And_Read_File:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "absolute_filename": ("STRING", {
                    "default":
                    "image.txt",
                    "tooltip":
                    "The absolute path to the file to read."
                })
            },
            "optional": {
                "text": ("STRING", {
                    "multiline":
                    True,
                    "dynamicPrompts":
                    True,
                    "tooltip":
                    "Text to provide order when necessary (to wait on done file)."
                }),
            }
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("text from file", )
    FUNCTION = "read_text"

    OUTPUT_NODE = True

    CATEGORY = "text"
    DESCRIPTION = "Saves the input text to an absolute path."

    def read_text(self, absolute_filename, text=''):
        while not os.path.exists(absolute_filename):
            time.sleep(0.1)

        res = open(absolute_filename, "r").read()
        os.unlink(absolute_filename)

        return (res, )
