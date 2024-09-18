#!/usr/bin/python3
import torch
import cv2


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
