#!/usr/bin/python3
import torch
import facer
import cv2
import einops
import numpy as np


def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, code=cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 255.0
    return image


def do_recolor(vis_seg_probs, n_classes):
    val = int(255 / n_classes)
    vis_seg_probs = vis_seg_probs.cpu().detach().numpy()
    not_visible = (vis_seg_probs == 0).astype(dtype=np.uint8)
    not_visible = 1 - not_visible
    not_visible *= 255
    vis_seg_probs *= val
    ret = np.array((vis_seg_probs, not_visible, not_visible), np.uint8)
    ret = einops.rearrange(ret, 'c h w -> h w c')
    ret = cv2.cvtColor(ret, cv2.COLOR_HSV2BGR_FULL)
    return ret


def detect_face_from_tensor(image):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image *= 255
    image = image.to(dtype=torch.uint8)
    image = facer.hwc2bchw(image).to(device=device)
    face_detector = facer.face_detector('retinaface/mobilenet', device=device)

    with torch.inference_mode():
        faces = face_detector(image)

    face_parser = facer.face_parser(
        'farl/lapa/448', device=device)  # optional "farl/celebm/448"

    with torch.inference_mode():
        faces = face_parser(image, faces)

    seg_logits = faces['seg']['logits']
    seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w
    n_classes = seg_probs.size(1)

    vis_seg_probs = seg_probs.argmax(dim=1)
    vis_seg_probs = einops.einsum(vis_seg_probs, 'b h w -> h w')
    return (vis_seg_probs, n_classes)


def full_work_wrapper(image):
    try:
        res, n_classes = detect_face_from_tensor(image)
    except:
        res = torch.zeros((image.shape[0], image.shape[1]), dtype=torch.int64)
        n_classes = 11
        print('Warning: Failed to find any face in the image...')

    tup = do_recolor(res, n_classes)
    tup = torch.from_numpy(tup).to(device=image.device, dtype=image.dtype)

    return tup

    res, n_classes = detect_face_from_tensor(image)
    tup = do_recolor(res, n_classes)
    # tup = torch.from_numpy(tup).to(device=image.device, dtype=image.dtype)
    return tup


class main_face_segment():

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
            },
        }

    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", )
    CATEGORY = "TRI3D"

    def run(self, image):
        batch_size = image.shape[0]
        ret = []
        for i in range(batch_size):
            ret.append(full_work_wrapper(image[i].clone()))

        ret = np.array(ret)
        ret = torch.from_numpy(ret).to(dtype=image.dtype, device=image.device)
        print(ret.shape)

        return (ret, )
