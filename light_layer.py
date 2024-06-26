import numpy as np
import cv2
import math
import torch


def from_torch_image(image):
    image = image.cpu().numpy() * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def to_torch_image(image):
    image = image.astype(dtype=np.float32)
    image /= 255.0
    image = torch.from_numpy(image)
    return image


def smooth_step_plain(x):

    if x < -1:
        return -1
    elif x <= 1:
        return math.sin(x * np.pi / 2.0)
    else:
        return 1


def smooth_step_np(x):
    truths = np.logical_and(-1 < x, x < 1).astype(np.float32)
    x1 = np.clip(x, -1, 1)
    x2 = np.sin(x * np.pi / 2.0)
    ret = (truths * x2) + ((1 - truths) * x1)
    return ret


def smooth_step_stretch(x, a, b):

    if b < a:
        tmp = b
        b = a
        a = tmp

    if a < 0:
        a = 0

    if b > 1:
        b = 1

    if a == b:
        a = 0
        b = 1

    return smooth_step_np((2 * (x - a) / (b - a)) - 1)


def get_light_layer(image,
                    ref_r=255,
                    ref_g=255,
                    ref_b=255,
                    do_scale=True,
                    scale_a=0.0,
                    scale_b=1.0):

    sqmax = 3 * 255 * 255
    scalemax = math.sqrt(sqmax)

    b = image[:, :, 0].astype(dtype=np.float32)
    g = image[:, :, 1].astype(dtype=np.float32)
    r = image[:, :, 2].astype(dtype=np.float32)

    b2 = b * b
    g2 = g * g
    r2 = r * r

    d2 = np.zeros(b2.shape, dtype=np.float32)
    d2 += sqmax - b2 - g2 - r2
    d = np.sqrt(d2)

    ref_r2 = ref_r * ref_r
    ref_g2 = ref_g * ref_g
    ref_b2 = ref_b * ref_b
    ref_d2 = sqmax - ref_r2 - ref_g2 - ref_b2

    ref_d = math.sqrt(ref_d2)

    dot = (b * ref_b) + (g * ref_g) + (r * ref_r) + (d * ref_d)
    dot /= sqmax

    if do_scale:
        dot = smooth_step_stretch(x=dot, a=scale_a, b=scale_b)

    dot *= 255
    dot = dot.astype(np.uint8)

    return dot


class main_light_layer():

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "ref_r": ("INT", {
                    "default": 255,
                    "min": 0,
                    "max": 255,
                    "step": 1
                }),
                "ref_g": ("INT", {
                    "default": 255,
                    "min": 0,
                    "max": 255,
                    "step": 1
                }),
                "ref_b": ("INT", {
                    "default": 255,
                    "min": 0,
                    "max": 255,
                    "step": 1
                }),
                "do_scale": (["enable", "disable"], ),
                "thresh_low": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "thresh_high": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            },
        }

    FUNCTION = "run"
    RETURN_TYPES = ("MASK", )
    CATEGORY = "HackNode"

    def run(
        self,
        image,
        ref_r,
        ref_g,
        ref_b,
        do_scale,
        thresh_low,
        thresh_high,
    ):

        do_scale = (do_scale == "enable")
        print('do_scale', do_scale)

        image = from_torch_image(image)
        print('image.shape', image.shape)

        batch_size = image.shape[0]
        print('batch_size', batch_size)

        mask = []

        for i in range(batch_size):

            tmp_img = image[i]
            print('tmp_img.shape', tmp_img.shape)

            tmp_mask = get_light_layer(
                tmp_img,
                ref_b,
                ref_g,
                ref_r,
                do_scale,
                scale_a=thresh_low,
                scale_b=thresh_high,
            )
            print('tmp_mask.shape', tmp_mask.shape)

            mask.append(tmp_mask)

        mask = np.array(mask)

        mask = to_torch_image(mask)
        print(mask.shape)

        return (mask, )


NODE_CLASS_MAPPINGS = {
    'main_light_layer': main_light_layer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'main_light_layer': 'main_light_layer',
}
