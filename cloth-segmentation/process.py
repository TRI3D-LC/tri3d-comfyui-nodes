from network import U2NET

import os
from PIL import Image
import cv2
import gdown
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from collections import OrderedDict
from options import opt

import einops

def do_recolor(vis_seg_probs, n_classes):
    val = int(255 / n_classes)
    not_visible = (vis_seg_probs == 0).astype(dtype=np.uint8)
    not_visible = 1 - not_visible
    not_visible *= 255
    vis_seg_probs *= val
    ret = np.array((vis_seg_probs, not_visible, not_visible), np.uint8)
    ret = einops.rearrange(ret, 'c h w -> h w c')
    ret = cv2.cvtColor(ret, cv2.COLOR_HSV2RGB_FULL)
    return ret


def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print("----No checkpoints at given path----")
        return
    model_state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    print("----checkpoints loaded from path: {}----".format(checkpoint_path))
    return model


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


class Normalize_image(object):
    """Normalize given tensor into given mean and standard dev

    Args:
        mean (float): Desired mean to substract from tensors
        std (float): Desired std to divide from tensors
    """

    def __init__(self, mean, std):
        assert isinstance(mean, (float))
        if isinstance(mean, float):
            self.mean = mean

        if isinstance(std, float):
            self.std = std

        self.normalize_1 = transforms.Normalize(self.mean, self.std)
        self.normalize_3 = transforms.Normalize([self.mean] * 3, [self.std] * 3)
        self.normalize_18 = transforms.Normalize([self.mean] * 18, [self.std] * 18)

    def __call__(self, image_tensor):
        if image_tensor.shape[0] == 1:
            return self.normalize_1(image_tensor)

        elif image_tensor.shape[0] == 3:
            return self.normalize_3(image_tensor)

        elif image_tensor.shape[0] == 18:
            return self.normalize_18(image_tensor)

        else:
            assert "Please set proper channels! Normlization implemented only for 1, 3 and 18"




def apply_transform(img):
    transforms_list = []
    transforms_list += [transforms.ToTensor()]
    transforms_list += [Normalize_image(0.5, 0.5)]
    transform_rgb = transforms.Compose(transforms_list)
    return transform_rgb(img)


from PIL import Image


def generate_mask(input_image, net, device='cpu'):
    img = input_image
    img_size = img.size
    # img = img.resize((768, 768), Image.BICUBIC)
    image_tensor = apply_transform(img)
    image_tensor = torch.unsqueeze(image_tensor, 0)

    output_dir = os.path.join(opt.output, 'extracted_garment')
    os.makedirs(output_dir, exist_ok=True)

    print('#### DEBUG START ####')
    with torch.no_grad():
        output_tensor = net(image_tensor.to(device))
        print(output_tensor[0].shape)
        output_tensor = F.log_softmax(output_tensor[0], dim=1)
        output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
        output_tensor = torch.squeeze(output_tensor, dim=0)
        output_arr = output_tensor.cpu().numpy()

    print(output_arr.shape)
    image_tmp = do_recolor(vis_seg_probs = output_arr.squeeze(0), n_classes = 4)
    print(image_tmp.shape)
    print('#### DEBUG STOP ####')

    garment_path = os.path.join(output_dir, 'extracted_garment.png')
    cv2.imwrite(garment_path, cv2.cvtColor(src = image_tmp, code = cv2.COLOR_RGB2BGR))
    return image_tmp

    # # Create a binary mask where selected classes are 1, others are 0
    # binary_mask = np.zeros_like(output_arr, dtype=np.uint8)
    # classes_of_interest = [1, 2, 3]  # Modify this list according to your classes of interest
    # for cls in classes_of_interest:
    #     binary_mask[output_arr == cls] = 255

    # # Ensure binary_mask is 2D
    # if binary_mask.ndim > 2:
    #     binary_mask = binary_mask.squeeze()  # Removes single-dimensional entries from the shape
    # if binary_mask.ndim != 2:
    #     raise ValueError("binary_mask must be a 2-dimensional array")

    # binary_mask_img = Image.fromarray(binary_mask, mode='L').resize(img_size, Image.BICUBIC)

    # # Create an RGBA image for the output
    # extracted_garment = Image.new("RGBA", img_size)
    # original_img = img.resize(img_size)  # Resize the processed image back to original size
    # extracted_garment.paste(original_img, mask=binary_mask_img)

    # # Save the garment image with transparency
    # garment_path = os.path.join(output_dir, 'extracted_garment.png')
    # extracted_garment.save(garment_path, format="PNG")

    # return extracted_garment


# def generate_mask(input_image, net, device='cpu'):
#     img = input_image
#     img_size = img.size
#     img = img.resize((768, 768), Image.BICUBIC)
#     image_tensor = apply_transform(img)
#     image_tensor = torch.unsqueeze(image_tensor, 0)

#     output_dir = os.path.join(opt.output, 'extracted_garment')
#     os.makedirs(output_dir, exist_ok=True)

#     with torch.no_grad():
#         output_tensor = net(image_tensor.to(device))
#         output_tensor = F.log_softmax(output_tensor[0], dim=1)
#         output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
#         output_tensor = torch.squeeze(output_tensor, dim=0)
#         output_arr = output_tensor.cpu().numpy()

#     # Create a binary mask where selected classes are 1, others are 0
#     binary_mask = np.zeros_like(output_arr, dtype=np.uint8)
#     classes_of_interest = [1, 2, 3]  # Modify this list according to your classes of interest
#     for cls in classes_of_interest:
#         binary_mask[output_arr == cls] = 255

#     # Convert binary mask to a 3-channel image to use as a mask

#     # Ensure binary_mask is 2D
#     if binary_mask.ndim > 2:
#         binary_mask = binary_mask.squeeze()  # Removes single-dimensional entries from the shape
#     if binary_mask.ndim != 2:
#         raise ValueError("binary_mask must be a 2-dimensional array")

#     binary_mask_img = Image.fromarray(binary_mask, mode='L').resize(img_size, Image.BICUBIC)
#     binary_mask_3ch = binary_mask_img.convert('RGB')  # Convert to RGB

#     # Apply mask to the original image
#     original_img = img.resize(img_size)  # Resize the processed image back to original size
#     extracted_garment = Image.new("RGB", original_img.size)
#     extracted_garment.paste(original_img, mask=binary_mask_img)

#     # Save the garment image
#     garment_path = os.path.join(output_dir, 'extracted_garment.png')
#     extracted_garment.save(garment_path)

#     return extracted_garment


def check_or_download_model(file_path):
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        url = "https://drive.google.com/uc?export=download&id=1qVv720hAd11JSCuIVJuqfjCGolwb1H8o"
        gdown.download(url, file_path, quiet=False)
        print("Model downloaded successfully.")
    else:
        print("Model already exists.")



def load_seg_model(checkpoint_path, device='cpu'):
    net = U2NET(in_ch=3, out_ch=4)
    check_or_download_model(checkpoint_path)
    net = load_checkpoint(net, checkpoint_path)
    net = net.to(device)
    net = net.eval()

    return net


def main(args):

    device = 'cuda:0' if args.cuda else 'cpu'

    # Create an instance of your model
    model = load_seg_model(args.checkpoint_path, device=device)

    palette = get_palette(4)

    img = Image.open(args.image).convert('RGB')

    cloth_seg = generate_mask(img, net=model, palette=palette, device=device)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Help to set arguments for Cloth Segmentation.')
    parser.add_argument('--image', type=str, help='Path to the input image')
    parser.add_argument('--cuda', action='store_true', help='Enable CUDA (default: False)')
    parser.add_argument('--checkpoint_path', type=str, default='model/cloth_segm.pth', help='Path to the checkpoint file')
    args = parser.parse_args()

    main(args)
