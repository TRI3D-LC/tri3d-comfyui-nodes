#!/usr/bin/python3
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import torch


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


def do_custom_threshhold(image, value):
    image = image.astype(dtype=np.float64)
    image = 255 * (image - value) / (255 - value)
    image = np.clip(image, 0, 255)
    image = image.astype(dtype=np.uint8)
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

    result_image_lab = cv2.cvtColor(src=result_image, code=cv2.COLOR_RGB2LAB)

    luminosity_image_lab = cv2.cvtColor(src=luminosity_image,
                                        code=cv2.COLOR_RGB2LAB)[:, :, 0]

    luminosity_image_lab_flip = 255 - luminosity_image_lab

    luminosity_image_lab_flip = do_custom_threshhold(
        image=luminosity_image_lab_flip, value=threshhold_hist)

    luminosity_image_lab_flip_full = luminosity_image_lab_flip.copy()

    luminosity_image_lab_flip *= 1 - (final_mask[:, :, 0]
                                      > 127.5).astype(dtype=np.uint8)

    for i in range(3):
        result_image[:, :,
                     i] = (result_image[:, :, i] *
                           (1 - (luminosity_image_lab_flip / 255.0))).astype(
                               dtype=np.uint8)

    return (result_image, luminosity_image_lab_flip_full)


def find_threshold(image_input, threshold=0.0001):

    image_input_L = cv2.cvtColor(image_input, cv2.COLOR_RGB2LAB)[:, :,
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


def get_mu_sigma(array_input, mask_input):

    array_input = array_input.astype(dtype=np.float32).flatten()
    mask_input = mask_input.astype(dtype=np.float32).flatten()

    sum = np.sum(mask_input)
    mean = np.sum(array_input * mask_input) / sum

    array_input -= mean
    array_input *= mask_input
    sigma = math.sqrt(np.sum(np.square(array_input)) / sum)

    return mean, sigma


def renormalize_array_main(array_input, mask_input, mu, sigma):
    array_input_original = array_input.copy()
    in_mu, in_sigma = get_mu_sigma(array_input, mask_input)
    array_input = (((array_input - in_mu) / in_sigma) * sigma) + mu

    array_input_original = (array_input_original *
                            (1 - mask_input)) + (array_input * mask_input)

    return array_input_original


#!/usr/bin/python3
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

    RETURN_TYPES = (
        "IMAGE",
        "MASK",
    )

    RETURN_NAMES = (
        "output bg swapped image",
        "shadow layer",
    )

    FUNCTION = "test"

    #OUTPUT_NODE = False

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
        ret_lum = []

        if (subject_image.shape[0] == batch_size) and (mask_image.shape[0]
                                                       == batch_size):

            for i in range(batch_size):

                result, luminosity = do_bg_swap(
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

                luminosity = to_torch_image(luminosity)
                luminosity = luminosity.unsqueeze(0)
                ret_lum.append(luminosity)

        else:

            print(
                'input format is not correct, got different batch sizes for each input image'
            )

        ret = torch.cat(ret, dim=0)
        ret_lum = torch.cat(ret_lum, dim=0)

        return (
            ret,
            ret_lum,
        )


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

    #OUTPUT_NODE = False

    CATEGORY = "TRI3D"

    def test(
        self,
        subject_image,
        gradient_threshold,
    ):

        subject_image = from_torch_image(image=subject_image)
        return (find_threshold(subject_image[0],
                               threshold=gradient_threshold), )


class RGB_2_LAB:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_RGB_image": ("IMAGE", ),
            },
        }

    RETURN_TYPES = ("MASK", "MASK", "MASK")
    RETURN_NAMES = ("L", "A", "B")

    FUNCTION = "test"

    #OUTPUT_NODE = False

    CATEGORY = "TRI3D"

    def test(self, input_RGB_image):
        print('input_RGB_image.shape', input_RGB_image.shape)
        input_RGB_image = from_torch_image(image=input_RGB_image)
        ret_L = []
        ret_A = []
        ret_B = []
        for i in range(input_RGB_image.shape[0]):
            tmp = cv2.cvtColor(input_RGB_image[i], cv2.COLOR_RGB2LAB)
            ret_L.append(to_torch_image(image=tmp[:, :, 0]).unsqueeze(0))
            ret_A.append(to_torch_image(image=tmp[:, :, 1]).unsqueeze(0))
            ret_B.append(to_torch_image(image=tmp[:, :, 2]).unsqueeze(0))

        ret_L = torch.cat(ret_L, dim=0)
        ret_A = torch.cat(ret_A, dim=0)
        ret_B = torch.cat(ret_B, dim=0)

        print(
            'LAB output',
            ret_L.shape,
            ret_A.shape,
            ret_B.shape,
        )

        return (ret_L, ret_A, ret_B)


class LAB_2_RGB:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_L": ("MASK", ),
                "input_A": ("MASK", ),
                "input_B": ("MASK", ),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("Output RGB image", )

    FUNCTION = "test"

    #OUTPUT_NODE = False

    CATEGORY = "TRI3D"

    def test(self, input_L, input_A, input_B):

        batch_size = input_L.shape[0]
        print(input_L.shape, input_A.shape, input_B.shape)

        ret = []

        if (input_A.shape[0] == batch_size) and (input_B.shape[0]
                                                 == batch_size):

            for i in range(batch_size):

                input_L_NP = from_torch_image(image=input_L[i])
                input_A_NP = from_torch_image(image=input_A[i])
                input_B_NP = from_torch_image(image=input_B[i])

                Y_MAX = input_L_NP.shape[0]
                X_MAX = input_L_NP.shape[1]

                if (input_A_NP.shape[0]
                        == Y_MAX) and (input_B_NP.shape[0] == Y_MAX) and (
                            (input_A_NP.shape[1] == X_MAX) and
                            (input_B_NP.shape[1] == X_MAX)):

                    image = np.zeros((Y_MAX, X_MAX, 3), dtype=np.uint8)

                    image[:, :, 0] = input_L_NP
                    image[:, :, 1] = input_A_NP
                    image[:, :, 2] = input_B_NP

                    image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
                    image = to_torch_image(image).unsqueeze(0)
                    print('image.shape')
                    print(image.shape)
                    ret.append(image)

                else:

                    print('Resolution of different layers donot match')

        else:

            print('batch size of different layers donot match')

        ret = torch.cat(ret, dim=0)

        print('ret.shape', ret.shape)

        return (ret, )


class get_mean_and_standard_deviation:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_array": ("MASK", ),
                "input_mask": ("MASK", ),
            },
        }

    RETURN_TYPES = (
        "FLOAT",
        "FLOAT",
    )

    RETURN_NAMES = (
        "Mean",
        "Standard deviation",
    )

    FUNCTION = "test"
    CATEGORY = "TRI3D"

    def test(self, input_array, input_mask):

        input_array = input_array.cpu().numpy()
        input_mask = input_mask.cpu().numpy()

        mean, sigma = get_mu_sigma(array_input=input_array[0],
                                   mask_input=input_mask[0])

        return (
            mean,
            sigma,
        )


class renormalize_array:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_array": ("MASK", ),
                "input_mask": ("MASK", ),
                "input_mean": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.001,
                        "round":
                        0.000001,  #The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                        "display": "number"
                    }),
                "input_standard_deviation": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.001,
                        "round":
                        0.000001,  #The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                        "display": "number"
                    }),
            },
        }

    RETURN_TYPES = ("MASK", )
    RETURN_NAMES = ("Output array as mask", )

    FUNCTION = "test"

    #OUTPUT_NODE = False

    CATEGORY = "TRI3D"

    def test(
        self,
        input_array,
        input_mask,
        input_mean,
        input_standard_deviation,
    ):

        batch_size = input_array.shape[0]

        ret = []

        if input_mask.shape[0] == batch_size:

            for i in range(batch_size):

                tmp = renormalize_array_main(
                    array_input=input_array[i].cpu().numpy(),
                    mask_input=input_mask[i].cpu().numpy(),
                    mu=input_mean,
                    sigma=input_standard_deviation)

                tmp = torch.from_numpy(tmp)
                tmp = tmp.unsqueeze(0)

                ret.append(tmp)

        else:

            print('batch size of different layers donot match')

        ret = torch.cat(ret, dim=0)

        return (ret, )
