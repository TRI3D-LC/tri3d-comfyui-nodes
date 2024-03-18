#!/usr/bin/python3
import torch
import cv2
import numpy as np


#!/usr/bin/python3
def from_torch_image(image):
    image = image.cpu().numpy() * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def to_torch_image(image):
    image = image.astype(dtype=np.float32)
    image /= 255.0
    image = torch.from_numpy(image)
    return image


def scaled_paste(
    image_background,
    image_foreground,
    mask_foreground,
    scale_factor,
    height_factor=1.2,
):

    print('DEBUG scaled_paste 0 ', image_background.shape,
          image_foreground.shape, mask_foreground.shape, scale_factor,
          height_factor)

    height = image_foreground.shape[0] * height_factor

    print('DEBUG scaled_paste 1 ', height)

    max_0 = max(image_background.shape[0], height)
    max_1 = max(image_background.shape[1], image_foreground.shape[1])

    print('DEBUG scaled_paste 2 ', max_0, max_1)

    ratio_0 = max_0 / image_background.shape[0]
    ratio_1 = max_1 / image_background.shape[1]
    ratio_max = max(ratio_0, ratio_1) * scale_factor

    print('DEBUG scaled_paste 2 ', ratio_0, ratio_1, ratio_max)

    size_0 = int(image_background.shape[0] * ratio_max) + 1
    size_1 = int(image_background.shape[1] * ratio_max) + 1

    print('DEBUG scaled_paste 3 ', size_0, size_1)

    image_background = cv2.resize(image_background, (size_1, size_0),
                                  cv2.INTER_CUBIC)

    print('DEBUG scaled_paste 4 ', image_background.shape)

    end_0 = int(image_background.shape[0])
    begin_0 = int(end_0 - height)
    end_0 = int(begin_0 + image_foreground.shape[0])

    print('DEBUG scaled_paste 5 ', begin_0, end_0)

    end_1 = image_background.shape[1]
    begin_1 = end_1 - image_foreground.shape[1]
    begin_1 = int(begin_1 / 2)
    end_1 = int(begin_1 + image_foreground.shape[1])

    print('DEBUG scaled_paste 6 ', begin_1, end_1)

    image_reference = image_background[begin_0:end_0, begin_1:end_1, :]

    for i in range(3):
        image_reference[:, :,
                        i] = (mask_foreground * image_foreground[:, :, i]) + (
                            (1 - mask_foreground) * image_reference[:, :, i])

    return image_background


#!/usr/bin/python3
class main_scaled_paste():

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_background": ("IMAGE", ),
                "image_foreground": ("IMAGE", ),
                "mask_foreground": ("MASK", ),
                "scale_factor": ("FLOAT", {
                    "default": 1.2,
                    "min": 1,
                    "max": 10,
                    "step": 0.05
                }),
                "height_factor": ("FLOAT", {
                    "default": 1.01,
                    "min": 1,
                    "max": 8,
                    "step": 0.05
                }),
            },
        }

    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", )
    CATEGORY = "TRI3D"

    def run(
        self,
        image_background,
        image_foreground,
        mask_foreground,
        scale_factor,
        height_factor,
    ):

        print('DEBUG 0 ', image_background.shape, image_foreground.shape,
              mask_foreground.shape)

        image_background = from_torch_image(image_background)
        image_foreground = from_torch_image(image_foreground)
        mask_foreground = mask_foreground.cpu().numpy()

        image_output = scaled_paste(
            image_background[0],
            image_foreground[0],
            mask_foreground[0],
            scale_factor,
            height_factor,
        )

        print('DEBUG 1 ', image_output.shape)

        image_output = to_torch_image(image=image_output)

        print('DEBUG 2 ', image_output.shape)

        image_output = image_output.unsqueeze(0)

        print('DEBUG 3 ', image_output.shape)

        return (image_output, )


#!/usr/bin/python3
# mask = cv2.imread('/home/asd/DATASETS/BG_SWAP_HACK_TEST/FOREGROUND_MASK.png',
#                   cv2.IMREAD_GRAYSCALE)

# mask = mask.astype(dtype=np.float32) / 255.0

# image_background = scaled_paste(
#     image_background=cv2.imread(
#         '/home/asd/DATASETS/BG_SWAP_HACK_TEST/BACKGROUND_DEPTH.png',
#         cv2.IMREAD_COLOR),
#     image_foreground=cv2.imread(
#         '/home/asd/DATASETS/BG_SWAP_HACK_TEST/FOREGROUND_DEPTH.png',
#         cv2.IMREAD_COLOR),
#     mask_foreground=mask,
#     scale_factor=2,
#     height_factor=1.05,
# )

# cv2.imwrite('tmp.png', image_background)

NODE_CLASS_MAPPINGS = {
    'main_scaled_paste': main_scaled_paste,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'main_scaled_paste': 'main_scaled_paste',
}
