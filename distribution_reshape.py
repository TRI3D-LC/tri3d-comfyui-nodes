import cv2
import os
import torch
import numpy as np


def from_torch_image(image):
    image = image.cpu().numpy() * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def to_torch_image(image):
    image = image.astype(dtype=np.float32)
    image /= 255.0
    image = torch.from_numpy(image)
    return image


def get_histogram(array):
    array = array.flatten().astype(dtype=np.float64)
    hist = np.histogram(array, bins=256, range=(0, 256))
    array = hist[0].astype(dtype=np.float64)
    array /= len(array)
    return array


def get_limits(array, threshold_fraction):
    array = get_histogram(array)

    left_sum = 0
    right_sum = 0

    left_start = 0
    right_start = len(array) - 1

    for i in range(len(array)):

        left_index = i
        right_index = len(array) - i - 1

        left_sum += array[left_index]
        right_sum += array[right_index]

        if left_sum < threshold_fraction:
            left_start = left_index

        if right_sum < threshold_fraction:
            right_start = right_index

        if (left_sum > threshold_fraction) and (right_sum
                                                > threshold_fraction):

            return (left_start, right_start)


def do_rescale(x, y1, y2, x1, x2):

    x = x.astype(dtype=np.float64)

    if x1 > x2:
        x1, x2 = x2, x1

    if y1 > y2:
        y1, y2 = y2, y1

    epsilon = 0.0001

    y = (x - x1)
    y /= (x2 - x1 + epsilon)
    y *= (y2 - y1)
    y += y1
    y = np.clip(y, y1, y2)

    for iy in range(y.shape[0]):
        for ix in range(y.shape[1]):
            if y[iy, ix] > 255:
                print(iy, ix)

    y = y.astype(dtype=np.uint8)

    return y


class get_histogram_limits:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "luminosity_as_mask": ("MASK", ),
                "threshold_fraction": ("FLOAT", {
                    "default": 0.001,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.00001,
                    "round": 0.000001,
                    "display": "number"
                }),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("histogram lower limit (x1) as INT",
                    "histogram upper limit (x2) as INT")

    FUNCTION = "test"

    #OUTPUT_NODE = False

    CATEGORY = "TRI3D"

    def test(self, luminosity_as_mask, threshold_fraction):

        luminosity_as_mask = from_torch_image(image=luminosity_as_mask)

        (left_start,
         right_start) = get_limits(array=luminosity_as_mask[0],
                                   threshold_fraction=threshold_fraction)

        return (left_start, right_start)


class simple_rescale_histogram:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "layer_as_mask": ("MASK", ),
                "y1": ("INT", {
                    "default": 100,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "display": "number"
                }),
                "y2": ("INT", {
                    "default": 200,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "display": "number"
                }),
                "x1": ("INT", {
                    "default": 100,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "display": "number"
                }),
                "x2": ("INT", {
                    "default": 200,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "display": "number"
                })
            },
        }

    RETURN_TYPES = ("MASK", )
    RETURN_NAMES = ("rescaled layer as MASK", )
    FUNCTION = "test"
    CATEGORY = "TRI3D"

    def test(self, layer_as_mask, y1, y2, x1, x2):
        layer_as_mask = from_torch_image(image=layer_as_mask[0])
        layer_as_mask = do_rescale(x=layer_as_mask, y1=y1, y2=y2, x1=x1, x2=x2)
        layer_as_mask = to_torch_image(image=layer_as_mask)
        layer_as_mask = layer_as_mask.unsqueeze(0)
        return (layer_as_mask, )


NODE_CLASS_MAPPINGS = {
    "get_histogram_limits": get_histogram_limits,
    'simple_rescale_histogram': simple_rescale_histogram
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "get_histogram_limits": "get_histogram_limits",
    "simple_rescale_histogram": "simple_rescale_histogram"
}
