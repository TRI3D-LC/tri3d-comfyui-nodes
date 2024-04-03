#!/usr/bin/python3
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt


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
    scale_factor=1.2,
    height_factor=1.05,
):

    height = image_foreground.shape[0] * height_factor

    max_0 = max(image_background.shape[0], height)
    max_1 = max(image_background.shape[1], image_foreground.shape[1])

    ratio_0 = max_0 / image_background.shape[0]
    ratio_1 = max_1 / image_background.shape[1]
    ratio_max = max(ratio_0, ratio_1) * scale_factor

    size_0 = int(image_background.shape[0] * ratio_max) + 1
    size_1 = int(image_background.shape[1] * ratio_max) + 1

    image_background = cv2.resize(image_background, (size_1, size_0),
                                  cv2.INTER_CUBIC)

    end_0 = int(image_background.shape[0])
    begin_0 = int(end_0 - height)
    end_0 = int(begin_0 + image_foreground.shape[0])

    end_1 = image_background.shape[1]
    begin_1 = end_1 - image_foreground.shape[1]
    begin_1 = int(begin_1 / 2)
    end_1 = int(begin_1 + image_foreground.shape[1])

    image_reference = image_background[begin_0:end_0, begin_1:end_1, :]

    for i in range(3):
        image_reference[:, :,
                        i] = (mask_foreground * image_foreground[:, :, i]) + (
                            (1 - mask_foreground) * image_reference[:, :, i])

    return image_background


def do_custom_threshhold(image, value):
    image = image.astype(dtype=np.float64)
    image = 255 * (image - value) / (255 - value)
    image = np.clip(image, 0, 255)
    image = image.astype(dtype=np.uint8)
    return image


def do_bg_swap(
    bkg_image,
    subject_image,
    mask_image,
    threshhold_hist,
    scale_factor=1.2,
    height_factor=1.05,
):

    blank_subject_image = np.zeros(subject_image.shape, dtype=np.uint8)
    blank_subject_image += 255

    blank_background_image = np.zeros(bkg_image.shape, dtype=np.uint8)

    blank_subject_mask = np.zeros(
        (subject_image.shape[0], subject_image.shape[1]), dtype=np.float64)

    blank_subject_mask += 1

    mask_image_3channel = subject_image.copy()
    for i in range(3):
        mask_image_3channel[:, :, i] = mask_image

    mask_image = mask_image.astype(dtype=np.float64)
    mask_image /= 255.0

    result_image = scaled_paste(
        image_background=bkg_image,
        image_foreground=subject_image,
        mask_foreground=mask_image,
        scale_factor=scale_factor,
        height_factor=height_factor,
    )

    luminosity_image = scaled_paste(
        image_background=blank_background_image + 255,
        image_foreground=subject_image,
        mask_foreground=blank_subject_mask,
        scale_factor=scale_factor,
        height_factor=height_factor,
    )

    final_mask = scaled_paste(
        image_background=blank_background_image,
        image_foreground=mask_image_3channel,
        mask_foreground=mask_image,
        scale_factor=scale_factor,
        height_factor=height_factor,
    )

    result_image_lab = cv2.cvtColor(src=result_image, code=cv2.COLOR_BGR2LAB)

    luminosity_image_lab = cv2.cvtColor(src=luminosity_image,
                                        code=cv2.COLOR_BGR2LAB)[:, :, 0]

    luminosity_image_lab_flip = 255 - luminosity_image_lab

    luminosity_image_lab_flip = do_custom_threshhold(
        image=luminosity_image_lab_flip, value=threshhold_hist)

    luminosity_image_lab_flip *= 1 - (final_mask[:, :, 0]
                                      > 127.5).astype(dtype=np.uint8)

    for i in range(3):
        result_image[:, :,
                     i] = (result_image[:, :, i] *
                           (1 - (luminosity_image_lab_flip / 255.0))).astype(
                               dtype=np.uint8)

    return result_image


def find_threshold(image_input, threshold=0.0001):

    image_input_L = cv2.cvtColor(image_input, cv2.COLOR_BGR2LAB)[:, :,
                                                                 0].flatten()
    image_input_L = 255 - image_input_L

    hist = np.histogram(image_input_L, range(0, 256, 1))

    values = hist[0]
    values = values.astype(dtype=np.float64)
    values /= len(image_input_L)

    for i in range(0, values.shape[0], 1):

        lhd = 0
        rhd = 0

        if i > 0:
            lhd = values[i] - values[i - 1]

        if i < values.shape[0] - 1:
            rhd = values[i + 1] - values[i]

        print(lhd, rhd)

        if max(lhd, rhd) > threshold:
            return i


class simple_bg_swap:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "bkg_image": ("IMAGE", ),
                "subject_image": ("IMAGE", ),
                "subject_mask": ("MASK", ),
                "threshhold_hist": (
                    "INT",
                    {
                        "default": 150,
                        "min": 0,  #Minimum value
                        "max": 255,  #Maximum value
                        "step": 1,  #Slider's step
                        "display":
                        "number"  # Cosmetic only: display as "number" or "slider"
                    }),
                "scale_factor": (
                    "FLOAT",
                    {
                        "default": 1.2,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.01,
                        "round":
                        0.001,  #The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                        "display": "number"
                    }),
                "height_factor": (
                    "FLOAT",
                    {
                        "default": 1.05,
                        "min": 1.0,
                        "max": 8.0,
                        "step": 0.01,
                        "round":
                        0.001,  #The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                        "display": "number"
                    }),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("output bg swapped image", )
    FUNCTION = "test"
    CATEGORY = "TRI3D"

    def test(
        self,
        bkg_image,
        subject_image,
        subject_mask,
        threshhold_hist,
        scale_factor,
        height_factor,
    ):

        mask_image = subject_mask

        bkg_image = from_torch_image(image=bkg_image)
        subject_image = from_torch_image(image=subject_image)
        mask_image = from_torch_image(image=mask_image)

        batch_size = bkg_image.shape[0]

        ret = []

        if (subject_image.shape[0] == batch_size) and (mask_image.shape[0]
                                                       == batch_size):

            for i in range(batch_size):

                result = do_bg_swap(
                    bkg_image[i],
                    subject_image[i],
                    mask_image[i],
                    threshhold_hist,
                    scale_factor,
                    height_factor,
                )

                result = to_torch_image(result)
                result = result.unsqueeze(0)
                ret.append(result)

        else:

            print(
                'input format is not correct, got different batch sizes for each input image'
            )

        ret = torch.cat(ret, dim=0)
        return (ret, )


class get_threshold_for_bg_swap:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "subject_image": ("IMAGE", ),
                "gradient_threshold": (
                    "FLOAT",
                    {
                        "default": 0.0001,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.00001,
                        "round":
                        0.000001,  #The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                        "display": "number"
                    }),
            },
        }

    RETURN_TYPES = ("INT", )
    RETURN_NAMES = ("output histogram threshold", )
    FUNCTION = "test"
    CATEGORY = "TRI3D"

    def test(
        self,
        subject_image,
        gradient_threshold,
    ):

        subject_image = from_torch_image(image=subject_image)
        return (find_threshold(subject_image[0],
                               threshold=gradient_threshold), )
