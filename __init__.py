# v1.1.0
import os
import os.path
tri3d_custom_nodes_path = os.path.dirname(os.path.abspath(__file__))
import cv2, json, math, pathlib, requests, io, tempfile, subprocess, sys, wget
import numpy as np
import torch
import torch.nn.functional as F
import hashlib
import comfy.model_management as model_management
import folder_paths
from PIL import Image, ImageOps
sys.path.append(tri3d_custom_nodes_path)
from scaled_paste import main_scaled_paste


def from_torch_image(image):
    image = image.squeeze().cpu().numpy() * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def to_torch_image(image):
    image = image.astype(dtype=np.float32)
    image /= 255.0
    image = torch.from_numpy(image)[
        None,
    ]
    image = image.unsqueeze(0)
    return image


def get_bounding_box(mask_input):
    rows = np.any(mask_input, axis=1)
    cols = np.any(mask_input, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, cmin, rmax, cmax


def extract_box_from_image(image, mask):
    y_min, x_min, y_max, x_max = get_bounding_box(mask_input=mask)
    return image[y_min:y_max + 1, x_min:x_max + 1, :]


def stitch_back_box_to_image(image_original, mask_original, image_patch):
    y_min, x_min, y_max, x_max = get_bounding_box(mask_input=mask_original)
    image_original[y_min:y_max + 1, x_min:x_max + 1] = image_patch
    return image_original


def do_work(image, external_file_name):
    exec(open(external_file_name, 'r').read())
    return image


def do_work_with_mask(image, mask, external_file_name):
    exec(open(external_file_name, 'r').read())
    return image


def mkdir_safe(out_path):
    if type(out_path) == str:
        if len(out_path) > 0:
            if not os.path.exists(out_path):
                os.mkdir(out_path)


def get_path_file_model():
    import folder_paths
    from folder_paths import models_dir

    path_file_model = models_dir
    mkdir_safe(out_path=path_file_model)

    path_file_model = os.path.join(path_file_model, 'inspyrenet')
    mkdir_safe(out_path=path_file_model)

    path_file_model = os.path.join(path_file_model,
                                   'InSPyReNet-SwinB-Plus-Ultra.pth')

    return path_file_model


def download_model_file(path_file_output=None):
    if path_file_output is None:
        path_file_output = get_path_file_model()

    if not os.path.exists(path_file_output):

        file_url = 'https://huggingface.co/hanamizuki-ai/InSPyReNet-SwinB-Plus-Ultra/resolve/main/latest.pth'
        
        import requests

        def download_file(url, destination):
            response = requests.get(url, stream=True)
            with open(destination, 'wb') as file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:  # filter out keep-alive chunks
                        file.write(chunk)

        download_file(file_url, path_file_output)

        # wget.download(file_url, out=path_file_output)

        # r = requests.get(file_url, stream=True)
        # with open(path_file_output, "wb") as f:
        #     for chunk in r.iter_content(chunk_size=1 << 24):
        #         if chunk:
        #             f.write(chunk)


def build_pip_install_cmds(args):

    if "python_embeded" in sys.executable or "python_embedded" in sys.executable:

        return [sys.executable, '-s', '-m', 'pip', 'install'] + args

    else:

        return [sys.executable, '-m', 'pip', 'install'] + args


def ensure_package(path_file_model=None):

    if path_file_model is None:
        path_file_model = get_path_file_model()

    cmds = build_pip_install_cmds(['-r', 'requirements.txt'])
    subprocess.run(cmds, cwd=tri3d_custom_nodes_path)
    download_model_file(path_file_model)


def run_transparent_background(path_dir_input,
                               path_dir_output,
                               path_file_model=None):

    if path_file_model is None:
        path_file_model = get_path_file_model()

    command = [
        'transparent-background', '--source', path_dir_input, '--dest',
        path_dir_output, '--ckpt', path_file_model
    ]

    subprocess.run(command)


def get_transparent_background(images, path_file_model=None):

    if path_file_model is None:
        path_file_model = get_path_file_model()

    batch_size = images.shape[0]

    path_dir_input = tempfile.TemporaryDirectory(
        suffix='.input.dir',
        prefix='transparent_background.',
        dir=None,
        ignore_cleanup_errors=False,
    )

    path_dir_output = tempfile.TemporaryDirectory(
        suffix='.output.dir',
        prefix='transparent_background.',
        dir=None,
        ignore_cleanup_errors=False,
    )

    for i in range(batch_size):
        path_file_output = os.path.join(path_dir_input.name, str(i) + '.png')
        cv2.imwrite(path_file_output, images[i])

    run_transparent_background(path_dir_input=path_dir_input.name,
                               path_dir_output=path_dir_output.name,
                               path_file_model=path_file_model)

    # remove_list = [
    #     os.path.join(path_dir_input, i) for i in os.listdir(path_dir_input)
    # ]
    # for i in remove_list:
    #     os.unlink(i)
    # del remove_list

    images = []
    for i in range(batch_size):
        path_file_input = os.path.join(path_dir_output.name,
                                       str(i) + '_rgba.png')
        tmp_img = cv2.imread(path_file_input, cv2.IMREAD_UNCHANGED)
        images.append(tmp_img)

    images = np.array(images)

    path_dir_output.cleanup()
    path_dir_input.cleanup()

    return images


class TRI3DATRParseBatch:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "main"
    CATEGORY = "TRI3D"

    def main(self, images):
        import cv2
        import numpy as np
        import torch
        import os
        import shutil

        def tensor_to_cv2_img(tensor, remove_alpha=False):
            i = 255. * tensor.cpu().numpy()  # This will give us (H, W, C)
            img = np.clip(i, 0, 255).astype(np.uint8)
            return img

        def cv2_img_to_tensor(img):
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img)[
                None,
            ]
            return img

        ATR_PATH = 'custom_nodes/tri3d-comfyui-nodes/atr_node/'
        ATR_INPUT_PATH = ATR_PATH + 'input/'
        ATR_OUTPUT_PATH = ATR_PATH + 'output/'

        # Create the input directory if it does not exist
        shutil.rmtree(ATR_INPUT_PATH, ignore_errors=True)
        os.makedirs(ATR_INPUT_PATH, exist_ok=True)

        shutil.rmtree(ATR_OUTPUT_PATH, ignore_errors=True)
        os.makedirs(ATR_OUTPUT_PATH, exist_ok=True)

        for i in range(images.shape[0]):
            image = images[i]
            cv2_image = tensor_to_cv2_img(image)
            cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(ATR_INPUT_PATH + f"image{i}.png", cv2_image)

        # Run the ATR model
        cwd = os.getcwd()
        os.chdir(ATR_PATH)
        from dotenv import load_dotenv
        load_dotenv()
        COMFY_PYTHON_PATH = os.getenv('COMFY_PYTHON_PATH','python')
        print("python path ", COMFY_PYTHON_PATH)
        os.system(
            COMFY_PYTHON_PATH + " simple_extractor.py --dataset atr --model-restore checkpoints/atr.pth --input-dir input --output-dir output"
        )
        os.chdir(cwd)

        # Collect and return the results
        batch_results = []
        for i in range(images.shape[0]):
            cv2_segm = cv2.imread(ATR_OUTPUT_PATH + f'image{i}.png')
            cv2_segm = cv2.cvtColor(cv2_segm, cv2.COLOR_BGR2RGB)
            b_tensor_img = cv2_img_to_tensor(cv2_segm)
            batch_results.append(b_tensor_img.squeeze(0))

        batch_results = torch.stack(batch_results)

        return (batch_results, )


class TRI3DExtractPartsBatch:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "batch_images": ("IMAGE", ),
                "batch_segs": ("IMAGE", ),
                "batch_secondaries": ("IMAGE", ),
                "margin": ("INT", {
                    "default": 15,
                    "min": 0
                }),
                "right_leg": ("BOOLEAN", {
                    "default": False
                }),
                "right_hand": ("BOOLEAN", {
                    "default": True
                }),
                "head": ("BOOLEAN", {
                    "default": False
                }),
                "hair": ("BOOLEAN", {
                    "default": False
                }),
                "left_shoe": ("BOOLEAN", {
                    "default": False
                }),
                "bag": ("BOOLEAN", {
                    "default": False
                }),
                "background": ("BOOLEAN", {
                    "default": False
                }),
                "dress": ("BOOLEAN", {
                    "default": False
                }),
                "left_leg": ("BOOLEAN", {
                    "default": False
                }),
                "right_shoe": ("BOOLEAN", {
                    "default": False
                }),
                "left_hand": ("BOOLEAN", {
                    "default": True
                }),
                "upper_garment": ("BOOLEAN", {
                    "default": False
                }),
                "lower_garment": ("BOOLEAN", {
                    "default": False
                }),
                "belt": ("BOOLEAN", {
                    "default": False
                }),
                "skirt": ("BOOLEAN", {
                    "default": False
                }),
                "hat": ("BOOLEAN", {
                    "default": False
                }),
                "sunglasses": ("BOOLEAN", {
                    "default": False
                }),
                "scarf": ("BOOLEAN", {
                    "default": False
                }),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
    )
    FUNCTION = "main"
    CATEGORY = "TRI3D"

    def main(self, batch_images, batch_segs, batch_secondaries, margin,
             right_leg, right_hand, head, hair, left_shoe, bag, background,
             dress, left_leg, right_shoe, left_hand, upper_garment,
             lower_garment, belt, skirt, hat, sunglasses, scarf):
        import cv2
        import numpy as np
        import torch
        from pprint import pprint

        def get_segment_counts(segm):
            # Load the segmentation image

            # Reshape the image array to be 2D
            reshaped = segm.reshape(-1, segm.shape[-1])

            # Find unique vectors and their counts
            unique_vectors, counts = np.unique(reshaped,
                                               axis=0,
                                               return_counts=True)
            segment_counts = list(zip(unique_vectors, counts))
            pprint(segment_counts)
            return segment_counts

        def bounded_image(seg_img, color_code_list, input_img):
            import cv2
            import numpy as np
            # Create a mask for hands
            seg_img = cv2.resize(seg_img,
                                 (input_img.shape[1], input_img.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
            hand_mask = np.zeros_like(seg_img[:, :, 0])
            for color in color_code_list:
                lowerb = np.array(color, dtype=np.uint8)
                upperb = np.array(color, dtype=np.uint8)
                temp_mask = cv2.inRange(seg_img, lowerb, upperb)
                hand_mask = cv2.bitwise_or(hand_mask, temp_mask)

            # Find contours to get the bounding box of the hands
            contours, _ = cv2.findContours(hand_mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            # If no contours were found, just return None
            if not contours:
                return None

            # Combine all contours to find encompassing bounding box
            all_points = np.concatenate(contours, axis=0)
            x, y, w, h = cv2.boundingRect(all_points)

            print(x, y, w, h, "x,y,w,h")
            x = max(x - margin, 0)
            y = max(y - margin, 0)
            # Ensure width does not exceed image boundary
            w = min(w + 2 * margin, input_img.shape[1] - x)
            # Ensure height does not exceed image boundary
            h = min(h + 2 * margin, input_img.shape[0] - y)
            print(x, y, w, h, "x,y,w,h")
            print(input_img.shape, "input_img.shape")
            # Extract the region from the original image that contains both hands
            hand_region = input_img[y:y + h, x:x + w]

            return hand_region

        def tensor_to_cv2_img(tensor, remove_alpha=False):
            # This will give us (H, W, C)
            i = 255. * tensor.squeeze(0).cpu().numpy()
            img = np.clip(i, 0, 255).astype(np.uint8)
            return img

        def cv2_img_to_tensor(img):
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img)[
                None,
            ]
            return img

        batch_results = []
        images = []
        secondaries = []

        for i in range(batch_images.shape[0]):
            image = batch_images[i]
            seg = batch_segs[i]

            cv2_image = tensor_to_cv2_img(image)
            cv2_secondary = tensor_to_cv2_img(batch_secondaries[i])
            cv2_seg = tensor_to_cv2_img(seg)

            color_code_list = []
            ################# ATR MAPPING#################
            if right_leg:
                color_code_list.append([192, 0, 128])
            if right_hand:
                color_code_list.append([192, 128, 128])
            if head:
                color_code_list.append([192, 128, 0])
            if hair:
                color_code_list.append([0, 128, 0])
            if left_shoe:
                color_code_list.append([192, 0, 0])
            if bag:
                color_code_list.append([0, 64, 0])
            if background:
                color_code_list.append([0, 0, 0])
            if dress:
                color_code_list.append([128, 128, 128])
            if left_leg:
                color_code_list.append([64, 0, 128])
            if right_shoe:
                color_code_list.append([64, 128, 0])
            if left_hand:
                color_code_list.append([64, 128, 128])
            if upper_garment:
                color_code_list.append([0, 0, 128])
            if lower_garment:
                color_code_list.append([0, 128, 128])
            if belt:
                color_code_list.append([64, 0, 0])
            if skirt:
                color_code_list.append([128, 0, 128])
            if hat:
                color_code_list.append([128, 0, 0])
            if sunglasses:
                color_code_list.append([128, 128, 0])
            if scarf:
                color_code_list.append([128, 64, 0])

            bimage = bounded_image(cv2_seg, color_code_list, cv2_image)
            bsecondary = bounded_image(cv2_seg, color_code_list, cv2_secondary)

            # Handle case when bimage is None to avoid error during conversion to tensor
            if bimage is not None:
                images.append(bimage)
            else:
                num_channels = cv2_image.shape[2] if len(
                    cv2_image.shape) > 2 else 1
                black_img = np.zeros((10, 10, num_channels),
                                     dtype=cv2_image.dtype)
                images.append(black_img)

            if bsecondary is not None:
                secondaries.append(bsecondary)
            else:
                num_channels = cv2_image.shape[2] if len(
                    cv2_image.shape) > 2 else 1
                black_img = np.zeros((10, 10, num_channels),
                                     dtype=cv2_image.dtype)
                secondaries.append(black_img)

        # Get max height and width
        max_height = max(img.shape[0] for img in images)
        max_width = max(img.shape[1] for img in images)

        batch_results = []
        batch_secondaries = []

        for img in images:
            # Resize the image to max height and width
            resized_img = cv2.resize(img, (max_width, max_height),
                                     interpolation=cv2.INTER_CUBIC)
            tensor_img = cv2_img_to_tensor(resized_img)
            batch_results.append(tensor_img.squeeze(0))

        for sec in secondaries:
            # Resize the image to max height and width
            resized_sec = cv2.resize(sec, (max_width, max_height),
                                     interpolation=cv2.INTER_NEAREST)
            tensor_sec = cv2_img_to_tensor(resized_sec)
            batch_secondaries.append(tensor_sec.squeeze(0))

        batch_results = torch.stack(batch_results)
        batch_secondaries = torch.stack(batch_secondaries)
        print(batch_results.shape, "batch_results.shape")
        return (batch_results, batch_secondaries)


class TRI3DPositionPartsBatch:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "batch_images": ("IMAGE", ),
                "batch_segs": ("IMAGE", ),
                "batch_handimgs": ("IMAGE", ),
                "margin": ("INT", {
                    "default": 15,
                    "min": 0
                }),
                "right_leg": ("BOOLEAN", {
                    "default": False
                }),
                "right_hand": ("BOOLEAN", {
                    "default": True
                }),
                "head": ("BOOLEAN", {
                    "default": False
                }),
                "hair": ("BOOLEAN", {
                    "default": False
                }),
                "left_shoe": ("BOOLEAN", {
                    "default": False
                }),
                "bag": ("BOOLEAN", {
                    "default": False
                }),
                "background": ("BOOLEAN", {
                    "default": False
                }),
                "dress": ("BOOLEAN", {
                    "default": False
                }),
                "left_leg": ("BOOLEAN", {
                    "default": False
                }),
                "right_shoe": ("BOOLEAN", {
                    "default": False
                }),
                "left_hand": ("BOOLEAN", {
                    "default": True
                }),
                "upper_garment": ("BOOLEAN", {
                    "default": False
                }),
                "lower_garment": ("BOOLEAN", {
                    "default": False
                }),
                "belt": ("BOOLEAN", {
                    "default": False
                }),
                "skirt": ("BOOLEAN", {
                    "default": False
                }),
                "hat": ("BOOLEAN", {
                    "default": False
                }),
                "sunglasses": ("BOOLEAN", {
                    "default": False
                }),
                "scarf": ("BOOLEAN", {
                    "default": False
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "main"
    CATEGORY = "TRI3D"

    def main(self, batch_images, batch_segs, batch_handimgs, margin, right_leg,
             right_hand, head, hair, left_shoe, bag, background, dress,
             left_leg, right_shoe, left_hand, upper_garment, lower_garment,
             belt, skirt, hat, sunglasses, scarf):
        import cv2
        import numpy as np
        import torch
        from pprint import pprint

        def bounded_image_points(seg_img, color_code_list, input_img):
            import cv2
            import numpy as np
            # Create a mask for hands
            seg_img = cv2.resize(seg_img,
                                 (input_img.shape[1], input_img.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
            hand_mask = np.zeros_like(seg_img[:, :, 0])
            for color in color_code_list:
                lowerb = np.array(color, dtype=np.uint8)
                upperb = np.array(color, dtype=np.uint8)
                temp_mask = cv2.inRange(seg_img, lowerb, upperb)
                hand_mask = cv2.bitwise_or(hand_mask, temp_mask)

            # Find contours to get the bounding box of the hands
            contours, _ = cv2.findContours(hand_mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            # If no contours were found, just return None
            if not contours:
                return None

            # Combine all contours to find encompassing bounding box
            all_points = np.concatenate(contours, axis=0)
            x, y, w, h = cv2.boundingRect(all_points)
            x = max(x - margin, 0)
            y = max(y - margin, 0)
            # Ensure width does not exceed image boundary
            w = min(w + 2 * margin, input_img.shape[1] - x)
            # Ensure height does not exceed image boundary
            h = min(h + 2 * margin, input_img.shape[0] - y)

            return (x, y, w, h)

        def tensor_to_cv2_img(tensor, remove_alpha=False):
            # This will give us (H, W, C)
            i = 255. * tensor.squeeze(0).cpu().numpy()
            img = np.clip(i, 0, 255).astype(np.uint8)
            return img

        def cv2_img_to_tensor(img):
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img)[
                None,
            ]
            return img
        
        def unsharp_mask(image, sigma=1.0, strength=1.0):
            # Blur the image
            blurred_image = cv2.GaussianBlur(image, (0, 0), sigma)
                                                    
            # Calculate the sharpened image
            sharpened_image = cv2.addWeighted(image, 1.0 + strength, blurred_image, -strength, 0)
            return sharpened_image
        
        batch_results = []

        for i in range(batch_images.shape[0]):
            image = batch_images[i]
            seg = batch_segs[i]
            handimg = batch_handimgs[i]

            cv2_image = tensor_to_cv2_img(image)
            cv2_seg = tensor_to_cv2_img(seg)

            color_code_list = []
            ################# ATR MAPPING#################
            if right_leg:
                color_code_list.append([192, 0, 128])
            if right_hand:
                color_code_list.append([192, 128, 128])
            if head:
                color_code_list.append([192, 128, 0])
            if hair:
                color_code_list.append([0, 128, 0])
            if left_shoe:
                color_code_list.append([192, 0, 0])
            if bag:
                color_code_list.append([0, 64, 0])
            if background:
                color_code_list.append([0, 0, 0])
            if dress:
                color_code_list.append([128, 128, 128])
            if left_leg:
                color_code_list.append([64, 0, 128])
            if right_shoe:
                color_code_list.append([64, 128, 0])
            if left_hand:
                color_code_list.append([64, 128, 128])
            if upper_garment:
                color_code_list.append([0, 0, 128])
            if lower_garment:
                color_code_list.append([0, 128, 128])
            if belt:
                color_code_list.append([64, 0, 0])
            if skirt:
                color_code_list.append([128, 0, 128])
            if hat:
                color_code_list.append([128, 0, 0])
            if sunglasses:
                color_code_list.append([128, 128, 0])
            if scarf:
                color_code_list.append([128, 64, 0])

            positions = bounded_image_points(cv2_seg, color_code_list,
                                             cv2_image)

            try:
                cv2_handimg = tensor_to_cv2_img(handimg)
                cv2_handimg = cv2.resize(cv2_handimg,
                                         (positions[2], positions[3]),
                                         interpolation=cv2.INTER_AREA)

                cv2_handimg = unsharp_mask(cv2_handimg)
                
                cv2_image[positions[1]:positions[1] + positions[3],
                          positions[0]:positions[0] +
                          positions[2]] = cv2_handimg
            except Exception as e:
                print(e)
                pass
            b_tensor_img = cv2_img_to_tensor(cv2_image)
            batch_results.append(b_tensor_img.squeeze(0))

        batch_results = torch.stack(batch_results)

        return (batch_results, )


class TRI3DSwapPixels:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "from_image": ("IMAGE", ),
                # "garment_mask": ("IMAGE",),
                "to_image": ("IMAGE", ),
                "to_mask": ("IMAGE", ),
                "swap_masked": ("BOOLEAN", {
                    "default": False
                })
            },
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "main"
    CATEGORY = "TRI3D"

    def main(self, from_image, to_image, to_mask, swap_masked):
        # og_image = cv2.imread(garment_image)
        # og_mask = cv2.imread(garment_mask)

        # fp_image = cv2.imread(fp_image_pat)
        # fp_mask = cv2.imread(fp_mask_path)
        def tensor_to_cv2_img(tensor, remove_alpha=False):
            i = 255. * tensor.cpu().numpy()  # This will give us (H, W, C)
            img = np.clip(i, 0, 255).astype(np.uint8)
            return img

        def cv2_img_to_tensor(img):
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img)[
                None,
            ]
            return img

        to_image = tensor_to_cv2_img(to_image)[0]
        to_mask = tensor_to_cv2_img(to_mask)[0]
        print(to_mask.shape)
        h, w, _ = to_mask.shape

        from_image = cv2.resize(tensor_to_cv2_img(from_image)[0], (w, h))
        # garment_mask = cv2.resize(tensor_to_cv2_img(garment_mask)[0], (w,h))

        # garment_mask = np.where(garment_mask == 0, 1, 0).astype("bool")
        to_mask = np.where(to_mask == 0, 1, 0).astype('bool')

        a = 1 if swap_masked else 0
        to_idx = np.where(to_mask == a)

        result_image = to_image

        result_image[to_idx] = from_image[to_idx]

        # plt.imshow(result_image)
        # result_image = np.expand_dims(result_image, axis=0)
        # print(result_image.shape)
        result_image = cv2_img_to_tensor(result_image)
        return (result_image, )


class TRI3DExtractPartsBatch2:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "batch_images": ("IMAGE", ),
                "batch_segs": ("IMAGE", ),
                "batch_secondaries": ("IMAGE", ),
                "margin": ("INT", {
                    "default": 15,
                    "min": 0
                }),
                "right_leg": ("BOOLEAN", {
                    "default": False
                }),
                "right_hand": ("BOOLEAN", {
                    "default": True
                }),
                "head": ("BOOLEAN", {
                    "default": False
                }),
                "hair": ("BOOLEAN", {
                    "default": False
                }),
                "left_shoe": ("BOOLEAN", {
                    "default": False
                }),
                "bag": ("BOOLEAN", {
                    "default": False
                }),
                "background": ("BOOLEAN", {
                    "default": False
                }),
                "dress": ("BOOLEAN", {
                    "default": False
                }),
                "left_leg": ("BOOLEAN", {
                    "default": False
                }),
                "right_shoe": ("BOOLEAN", {
                    "default": False
                }),
                "left_hand": ("BOOLEAN", {
                    "default": True
                }),
                "upper_garment": ("BOOLEAN", {
                    "default": False
                }),
                "lower_garment": ("BOOLEAN", {
                    "default": False
                }),
                "belt": ("BOOLEAN", {
                    "default": False
                }),
                "skirt": ("BOOLEAN", {
                    "default": False
                }),
                "hat": ("BOOLEAN", {
                    "default": False
                }),
                "sunglasses": ("BOOLEAN", {
                    "default": False
                }),
                "scarf": ("BOOLEAN", {
                    "default": False
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    FUNCTION = "main"
    CATEGORY = "TRI3D"

    def main(self, batch_images, batch_segs, batch_secondaries, margin,
             right_leg, right_hand, head, hair, left_shoe, bag, background,
             dress, left_leg, right_shoe, left_hand, upper_garment,
             lower_garment, belt, skirt, hat, sunglasses, scarf):
        import cv2
        import numpy as np
        import torch
        from pprint import pprint

        def tensor_to_cv2_img(tensor, remove_alpha=False):
            # This will give us (H, W, C)
            i = 255. * tensor.squeeze(0).cpu().numpy()
            img = np.clip(i, 0, 255).astype(np.uint8)
            return img

        def cv2_img_to_tensor(img):
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img)[
                None,
            ]
            return img

        batch_results = []
        images = []
        secondaries = []

        for i in range(batch_images.shape[0]):
            image = batch_images[i]
            seg = batch_segs[i]

            cv2_image = tensor_to_cv2_img(image)
            cv2_secondary = tensor_to_cv2_img(batch_secondaries[i])
            cv2_seg = tensor_to_cv2_img(seg)

            mask = np.zeros_like(cv2_image)
            color_code_list = []
            ################# ATR MAPPING#################
            if right_leg:
                color_code_list.append([192, 0, 128])
            if right_hand:
                color_code_list.append([192, 128, 128])
            if head:
                color_code_list.append([192, 128, 0])
            if hair:
                color_code_list.append([0, 128, 0])
            if left_shoe:
                color_code_list.append([192, 0, 0])
            if bag:
                color_code_list.append([0, 64, 0])
            if background:
                color_code_list.append([0, 0, 0])
            if dress:
                color_code_list.append([128, 128, 128])
            if left_leg:
                color_code_list.append([64, 0, 128])
            if right_shoe:
                color_code_list.append([64, 128, 0])
            if left_hand:
                color_code_list.append([64, 128, 128])
            if upper_garment:
                color_code_list.append([0, 0, 128])
            if lower_garment:
                color_code_list.append([0, 128, 128])
            if belt:
                color_code_list.append([64, 0, 0])
            if skirt:
                color_code_list.append([128, 0, 128])
            if hat:
                color_code_list.append([128, 0, 0])
            if sunglasses:
                color_code_list.append([128, 128, 0])
            if scarf:
                color_code_list.append([128, 64, 0])

            for color in color_code_list:
                idx = np.where(np.all(cv2_seg == color, axis=-1))
                mask[idx] = 1

            images.append(mask * cv2_image)
            mask = np.where(mask == 0, 255, 0)
            secondaries.append(mask)
            # print(mask.shape)

        # Get max height and width
        max_height = max(img.shape[0] for img in images)
        max_width = max(img.shape[1] for img in images)

        batch_results = []
        batch_secondaries = []

        for img in images:
            # Resize the image to max height and width
            resized_img = cv2.resize(img, (max_width, max_height),
                                     interpolation=cv2.INTER_AREA)
            # print(img.shape, "before tensor_img.shape")
            tensor_img = cv2_img_to_tensor(resized_img)
            # print(tensor_img.shape, "tensor_img.shape")
            batch_results.append(tensor_img.squeeze(0))

        for sec in secondaries:
            # Resize the image to max height and width
            resized_sec = cv2.resize(sec, (max_width, max_height),
                                     interpolation=cv2.INTER_AREA)
            tensor_sec = cv2_img_to_tensor(resized_sec)
            batch_secondaries.append(tensor_sec.squeeze(0))

        batch_results = torch.stack(batch_results)
        batch_secondaries = torch.stack(batch_secondaries)
        print(batch_results.shape, "batch_results.shape")
        return (batch_results, batch_secondaries)


class TRI3DSkinFeatheredPaddedMask:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "garment_masks": ("IMAGE", ),
                "first_pass_masks": ("IMAGE", ),
                "padding_margin": (
                    "INT",
                    {
                        "default": 20,
                        "min": 0
                    },
                )
            },
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "main"
    CATEGORY = "TRI3D"

    def main(self, garment_masks, first_pass_masks, padding_margin):
        import cv2
        import numpy as np
        import torch
        from pprint import pprint

        def tensor_to_cv2_img(tensor, remove_alpha=False):
            # This will give us (H, W, C)
            i = 255. * tensor.squeeze(0).cpu().numpy()
            img = np.clip(i, 0, 255).astype(np.uint8)
            return img

        def cv2_img_to_tensor(img):
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img)[
                None,
            ]
            return img

        results = []
        for i in range(first_pass_masks.shape[0]):

            garment_mask = tensor_to_cv2_img(garment_masks[i])
            # first_pass_image = tensor_to_cv2_img(first_pass_images[i])
            first_pass_mask = tensor_to_cv2_img(first_pass_masks[i])

            h, w, _ = first_pass_mask.shape

            garment_mask = cv2.resize(garment_mask, (w, h))

            garment_mask = np.where(garment_mask == 0, 1, 0).astype("bool")
            first_pass_mask = np.where(first_pass_mask == 0, 1,
                                       0).astype('bool')

            fp_dilate = cv2.dilate(first_pass_mask.astype("uint8"),
                                   np.ones((padding_margin, padding_margin),
                                           np.uint8),
                                   iterations=1)
            og_dilate = cv2.dilate(garment_mask.astype("uint8"),
                                   np.ones((30, 30), np.uint8),
                                   iterations=1)
            fp_erode = cv2.erode(first_pass_mask.astype("uint8"),
                                 np.ones((25, 25), np.uint8),
                                 iterations=1)
            result = (fp_dilate ^ fp_erode) * og_dilate

            result = np.where(result == 0, 0, 255)
            results.append(result)
            # print(mask.shape)

        # Get max height and width
        max_height = max(img.shape[0] for img in results)
        max_width = max(img.shape[1] for img in results)

        batch_results = []

        for img in results:
            # Resize the image to max height and width
            resized_img = cv2.resize(img, (max_width, max_height),
                                     interpolation=cv2.INTER_AREA)
            # print(img.shape, "before tensor_img.shape")
            tensor_img = cv2_img_to_tensor(resized_img)
            # print(tensor_img.shape, "tensor_img.shape")
            batch_results.append(tensor_img.squeeze(0))

        batch_results = torch.stack(batch_results)
        print(batch_results.shape, "batch_results.shape")
        return (batch_results, )


class TRI3DInteractionCanny:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "garment_masks": ("IMAGE", ),
                "first_pass_images": ("IMAGE", ),
                "first_pass_masks": ("IMAGE", ),
                "lower_threshold": (
                    "INT",
                    {
                        "default": 80,
                        "min": 0
                    },
                ),
                "higher_threshold": (
                    "INT",
                    {
                        "default": 240,
                        "min": 0
                    },
                )
            },
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "main"
    CATEGORY = "TRI3D"

    def main(self, garment_masks, first_pass_images, first_pass_masks,
             lower_threshold, higher_threshold):
        import cv2
        import numpy as np
        import torch
        from pprint import pprint

        def tensor_to_cv2_img(tensor, remove_alpha=False):
            # This will give us (H, W, C)
            i = 255. * tensor.squeeze(0).cpu().numpy()
            img = np.clip(i, 0, 255).astype(np.uint8)
            return img

        def cv2_img_to_tensor(img):
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img)[
                None,
            ]
            return img

        results = []
        for i in range(first_pass_masks.shape[0]):

            garment_mask = tensor_to_cv2_img(garment_masks[i])
            first_pass_image = tensor_to_cv2_img(first_pass_images[i])
            first_pass_mask = tensor_to_cv2_img(first_pass_masks[i])

            h, w, _ = first_pass_mask.shape

            garment_mask = cv2.resize(garment_mask, (w, h))

            garment_mask = np.where(garment_mask == 0, 1, 0).astype("bool")
            first_pass_mask = np.where(first_pass_mask == 0, 1,
                                       0).astype('bool')

            canny = cv2.Canny(first_pass_image, lower_threshold,
                              higher_threshold)
            canny = np.dstack((canny, canny, canny))

            result = (garment_mask * first_pass_mask).astype("uint8")

            result = result * canny
            results.append(result)
            # print(mask.shape)

        # Get max height and width
        max_height = max(img.shape[0] for img in results)
        max_width = max(img.shape[1] for img in results)

        batch_results = []

        for img in results:
            # Resize the image to max height and width
            resized_img = cv2.resize(img, (max_width, max_height),
                                     interpolation=cv2.INTER_AREA)
            # print(img.shape, "before tensor_img.shape")
            tensor_img = cv2_img_to_tensor(resized_img)
            # print(tensor_img.shape, "tensor_img.shape")
            batch_results.append(tensor_img.squeeze(0))

        batch_results = torch.stack(batch_results)
        print(batch_results.shape, "batch_results.shape")
        return (batch_results, )


class TRI3DDWPose_Preprocessor:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "detect_hand": (["enable", "disable"], {
                    "default": "enable"
                }),
                "detect_body": (["enable", "disable"], {
                    "default": "enable"
                }),
                "detect_face": (["enable", "disable"], {
                    "default": "enable"
                }),
                "filename_path": ("STRING", {
                    "default": "dwpose/keypoints/input.json"
                })
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    FUNCTION = "estimate_pose"
    CATEGORY = "TRI3D"

    def estimate_pose(self, images, detect_hand, detect_body, detect_face,
                      filename_path):
        from .dwpose import DwposeDetector

        detect_hand = detect_hand == "enable"
        detect_body = detect_body == "enable"
        detect_face = detect_face == "enable"

        DWPOSE_MODEL_NAME = "yzd-v/DWPose"
        annotator_ckpts_path = "dwpose/ckpts"

        model = DwposeDetector.from_pretrained(
            DWPOSE_MODEL_NAME, cache_dir=annotator_ckpts_path).to(
                model_management.get_torch_device())
        # out = common_annotator_call(model, image, include_hand=detect_hand, include_face=detect_face, include_body=detect_body)
        out_image_list = []
        out_dict_list = []
        out_dir_list = []
        for i, image in enumerate(images):
            H, W, C = image.shape
            np_image = np.asarray(image * 255., dtype=np.uint8)
            np_result, pose_dict = model(np_image,
                                         output_type="np",
                                         include_hand=detect_hand,
                                         include_face=detect_face,
                                         include_body=detect_body)
            cur_file_dir = os.path.dirname(os.path.realpath(__file__))
            save_file_path = os.path.join(cur_file_dir,
                                          filename_path)
            json.dump(pose_dict, open(save_file_path, 'w'))
            np_result = cv2.resize(np_result, (W, H),
                                   interpolation=cv2.INTER_AREA)
            out_image_list.append(
                torch.from_numpy(np_result.astype(np.float32) / 255.0))
            out_dict_list.append(pose_dict)
            out_dir_list.append(save_file_path)

        out_image = torch.stack(out_image_list, dim=0)
        del model

        return (out_image, save_file_path)


class TRI3DPosetoImage:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_json_file": ("STRING", {
                    "default": "dwpose/keypoints"
                })
            }
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "main"
    CATEGORY = "TRI3D"

    def main(self, pose_json_file):
        from .dwpose import comfy_utils

        pose = json.load(open(pose_json_file))
        height = pose['height']
        width = pose['width']
        keypoints = pose['keypoints']

        canvas = np.zeros(shape=(height, width, 3), dtype=np.uint8)
        canvas = comfy_utils.draw_bodypose(canvas, keypoints)
        canvas = comfy_utils.draw_handpose(canvas, keypoints[88:109])  #right hand
        canvas = comfy_utils.draw_handpose(canvas, keypoints[109:])  #left hand
        canvas = torch.from_numpy(canvas.astype(np.float32)/255.0)[None,]
        return (canvas, )


class TRI3DPoseAdaption:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_pose_json_file": ("STRING",{"default" : "dwpose/keypoints/input.json"}),
                "ref_pose_json_file": ("STRING",{"default" : "dwpose/keypoints/ref-pose.json"}),
                "image_angle": (["front", "back","back_fixed_kid","back_fixed","back_fixed_left","back_fixed_right"], {"default": "front"}),
                "rotation_threshold": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 15.0,
                    "step": 0.01
                }),
                "garment_category":(["no_sleeve_garment", "half_sleeve_garment", "full_sleeve_garment", \
                                     "shorts", "trouser"], {"default": "no_sleeve_garment"})


            }
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN")
    FUNCTION = "main"
    CATEGORY = "TRI3D"

    def main(self, input_pose_json_file, ref_pose_json_file, image_angle, rotation_threshold, garment_category):

        print(image_angle, "image_angle")
        print(garment_category, "garment_category")
        from .dwpose import comfy_utils

        if image_angle == "front":

            input_pose = json.load(open(input_pose_json_file))
            input_height = input_pose['height']
            input_width = input_pose['width']
            input_keypoints = input_pose['keypoints']
            canvas = np.zeros(shape=(input_height, input_width, 3), dtype=np.uint8)

            ref_pose = json.load(open(ref_pose_json_file))
            ref_height = ref_pose['height']
            ref_width = ref_pose['width']
            ref_keypoints = ref_pose['keypoints']
            similar_torso = None

            #check torso similarity
            if garment_category not in ["shorts", "trouser"]:
                ls_angle_1, rs_angle_1, torso_angle_1 = comfy_utils.get_torso_angles(
                    input_keypoints)
                ls_angle_2, rs_angle_2, torso_angle_2 = comfy_utils.get_torso_angles(
                    ref_keypoints)

                ls_angle_diff = abs(ls_angle_2 - ls_angle_1)
                rs_angle_diff = abs(rs_angle_2 - rs_angle_1)
                torso_angle_diff = abs(torso_angle_2 - torso_angle_1)

                
                similar_torso = False if (ls_angle_diff >= rotation_threshold) | (
                    rs_angle_diff >= rotation_threshold) | (torso_angle_diff >= rotation_threshold) else True

                if similar_torso == False:
                    canvas = torch.from_numpy(canvas.astype(np.float32) / 255.0)[
                        None,
                    ]
                    return (canvas, similar_torso)
            
            #Hands
            if input_keypoints[4] == [-1,-1]: input_keypoints[4] = ref_keypoints[4]
            if input_keypoints[7] == [-1,-1]: input_keypoints[7] = ref_keypoints[7]
            
            input_keypoints[88:] = ref_keypoints[88:]     #replace hands with reference hands

            if garment_category not in ["half_sleeve_garment", "full_sleeve_garment"]:
                input_keypoints = comfy_utils.rotate(ref_keypoints, input_keypoints, 2, 3)      # rotate left elbow
            input_keypoints = comfy_utils.scale(ref_keypoints, input_keypoints, 2, 5, 2, 3) #scaling w.r.t to shoulder to elbow ratio of ref pose

            prev_lw = input_keypoints[4]
            if garment_category != "full_sleeve_garment":
                input_keypoints = comfy_utils.rotate(ref_keypoints, input_keypoints, 3, 4)      #rotate left wrist
            input_keypoints = comfy_utils.scale(ref_keypoints, input_keypoints, 2, 3, 3, 4) #scaling w.r.t to elbow to wrist ratio of ref pose

            #moving to hand ponts to wrist 
            input_keypoints[109:] = comfy_utils.move(input_keypoints[109], input_keypoints[4], input_keypoints[109:]) 
            input_keypoints[109:] = comfy_utils.move(ref_keypoints[4], ref_keypoints[109], input_keypoints[109:])

            # input_keypoints = comfy_utils.rotate_hand(ref_keypoints, input_keypoints, 109)    #rotating left hand
            input_keypoints = comfy_utils.scale_hand(ref_keypoints, input_keypoints, 3, 4, 109)    #scaling left hand w.r.t left wrist

            if garment_category not in ["half_sleeve_garment", "full_sleeve_garment"]:            
                input_keypoints = comfy_utils.rotate(ref_keypoints, input_keypoints, 5, 6)      #rotate right elbow
            input_keypoints = comfy_utils.scale(ref_keypoints, input_keypoints, 2, 5, 5, 6) #scaling w.r.t to shoulder to elbow ratio of ref pose

            prev_rw = input_keypoints[7]
            if garment_category != "full_sleeve_garment":
                input_keypoints = comfy_utils.rotate(ref_keypoints, input_keypoints, 6, 7)      #rotate right wrist
            input_keypoints = comfy_utils.scale(ref_keypoints, input_keypoints, 5, 6, 6, 7)    #scaling w.r.t to elbow to wrist ratio of ref pose

            #moving hand points to wrist
            input_keypoints[88:109] = comfy_utils.move(input_keypoints[88], input_keypoints[7], input_keypoints[88:109])
            input_keypoints[88:109] = comfy_utils.move(ref_keypoints[7], ref_keypoints[88], input_keypoints[88:109])

            # input_keypoints = comfy_utils.rotate_hand(ref_keypoints, input_keypoints, 88)    #rotating right hand
            input_keypoints = comfy_utils.scale_hand(ref_keypoints, input_keypoints, 6, 7, 88)    #scaling right hand w.r.t right wrist


            #legs
            if garment_category not in ["trouser", "shorts"]:
                input_keypoints = comfy_utils.rotate(ref_keypoints, input_keypoints, 8, 9)      #rotate left knee
            input_keypoints = comfy_utils.scale(ref_keypoints, input_keypoints, 1, 8, 8, 9)      #scale left knee
            
            if garment_category != "trouser":
                input_keypoints = comfy_utils.rotate(ref_keypoints, input_keypoints, 9, 10)     #rotate left foot
            input_keypoints = comfy_utils.scale(ref_keypoints, input_keypoints, 8, 9, 9, 10)    #scaling w.r.t to knee to foot ratio of ref pose
            
            if garment_category not in ["trouser", "shorts"]:
                input_keypoints = comfy_utils.rotate(ref_keypoints, input_keypoints, 11, 12)     #rotate right knee
            input_keypoints = comfy_utils.scale(ref_keypoints, input_keypoints, 1, 11, 11, 12)      #scale right knee
            
            if garment_category != "trouser":
                input_keypoints = comfy_utils.rotate(ref_keypoints, input_keypoints, 12, 13)    #rotate right foot
            input_keypoints = comfy_utils.scale(ref_keypoints, input_keypoints, 11, 12, 12, 13)    #scaling w.r.t to knee to foot ratio of ref pose

            #face
            prev_nose = input_keypoints[0]
            input_keypoints = comfy_utils.rotate(ref_keypoints, input_keypoints, 1, 0)      #rotate nose
            input_keypoints = comfy_utils.scale(ref_keypoints, input_keypoints, 2, 5, 1, 0)      #scale neck to nose w.r.t to shoulder neck-nose len ratio of ref pose
            
            #changing face points to w.r.t to new nose point after rotation
            input_keypoints[14:18] = comfy_utils.move(prev_nose, input_keypoints[0], input_keypoints[14:18])

            input_keypoints = comfy_utils.rotate(ref_keypoints, input_keypoints, 0,
                                                14)  #rotate left eye
            input_keypoints = comfy_utils.scale(
                ref_keypoints, input_keypoints, 1, 0, 0,
                14)  #scaling w.r.t to neck len to eye_nose len ratio of ref pose

            input_keypoints = comfy_utils.rotate(ref_keypoints, input_keypoints, 0,
                                                15)  #rotate right eye
            input_keypoints = comfy_utils.scale(
                ref_keypoints, input_keypoints, 1, 0, 0,
                15)  #scaling w.r.t to neck len to eye_nose len ratio of ref pose

            input_keypoints = comfy_utils.rotate(ref_keypoints, input_keypoints,
                                                14, 16)  #rotate left ear
            input_keypoints = comfy_utils.scale(
                ref_keypoints, input_keypoints, 1, 0, 14,
                16)  #scaling w.r.t to neck len to ear_nose len ratio of ref pose

            input_keypoints = comfy_utils.rotate(ref_keypoints, input_keypoints,
                                                15, 17)  #rotate right ear
            input_keypoints = comfy_utils.scale(
                ref_keypoints, input_keypoints, 1, 0, 15,
                17)  #scaling w.r.t to neck len to ear_nose len ratio of ref pose

            canvas = comfy_utils.draw_bodypose(canvas, input_keypoints)
            canvas = comfy_utils.draw_handpose(canvas, input_keypoints[88:109])  #right hand
            canvas = comfy_utils.draw_handpose(canvas, input_keypoints[109:])  #left hand
            canvas = torch.from_numpy(canvas.astype(np.float32)/255.0)[None,]
            return (canvas, similar_torso)
        
        if 'back' in image_angle:

            input_pose = json.load(open(input_pose_json_file))
            input_height = input_pose['height']
            input_width = input_pose['width']
            input_keypoints = input_pose['keypoints']

            input_pose_type = comfy_utils.get_input_pose_type(input_keypoints)
            canvas = np.zeros(shape=(input_height, input_width, 3), dtype=np.uint8)

            if 'back_fixed' in image_angle:
                
                back_pose_dir = pathlib.Path().resolve() / 'custom_nodes/tri3d-comfyui-nodes/samples/back_poses/' 
                back_pose_dictionary = {
                    'back_fixed_kid' : 'backpose_kid.json',
                    'back_fixed' : 'backpose.json',
                    'back_fixed_left' : 'left_backpose.json',
                    'back_fixed_right' : 'right_backpose.json'
                }


                ref_pose_json_file =  back_pose_dir / back_pose_dictionary[image_angle]
            
            # /home/ubuntu/GITHUB/comfyanonymous/ComfyUI/custom_nodes/tri3d-comfyui-nodes
            print(ref_pose_json_file)
            ref_pose = json.load(open(ref_pose_json_file))
            
            ref_keypoints = ref_pose['keypoints']
            similar_torso = None
            
            #check torso similarity
            if garment_category not in ["shorts", "trouser"]:
                ls_angle_1, rs_angle_1, torso_angle_1 = comfy_utils.get_torso_angles(input_keypoints)
                ls_angle_2, rs_angle_2, torso_angle_2 = comfy_utils.get_torso_angles(ref_keypoints)

                ls_angle_diff = abs(ls_angle_2 - ls_angle_1)
                rs_angle_diff = abs(rs_angle_2 - rs_angle_1)
                torso_angle_diff = abs(torso_angle_2 - torso_angle_1)

                similar_torso = False if (ls_angle_diff >= 5) | (rs_angle_diff >= 5) | (torso_angle_diff >= 5) else True

                if similar_torso == False:
                    canvas = torch.from_numpy(canvas.astype(np.float32)/255.0)[None,]
                    return (canvas, similar_torso)

            #Removing the face points if existed
            null_indices = [i for i in range(len(ref_keypoints)) if ref_keypoints[i] == [-1,-1]]    
            for i in null_indices:
                input_keypoints[i] = [-1,-1]
            
            all_x = [i[0] for i in input_keypoints if i[0] != -1]
            min_width = min(all_x)
            max_width = max(all_x)

            if input_pose_type == "front_pose":
                #flip horizontally
                for i in range(len(input_keypoints)):
                    x,y = input_keypoints[i]
                    if input_keypoints[i] == [-1,-1]: continue
                    input_keypoints[i] = [(max_width - x)+min_width, y]
            
            #Hands
            if input_keypoints[4] == [-1,-1]: input_keypoints[4] = ref_keypoints[4]
            if input_keypoints[7] == [-1,-1]: input_keypoints[7] = ref_keypoints[7]
            input_keypoints[88:] = ref_keypoints[88:]     #replace hands with reference hands
            
            if garment_category not in ["half_sleeve_garment", "full_sleeve_garment"]:
                input_keypoints = comfy_utils.rotate(ref_keypoints, input_keypoints, 2, 3)      # rotate left elbow
            input_keypoints = comfy_utils.scale(ref_keypoints, input_keypoints, 2, 5, 2, 3) #scaling w.r.t to shoulder to elbow ratio of ref pose

            prev_lw = input_keypoints[4]
            if garment_category != "full_sleeve_garment":
                input_keypoints = comfy_utils.rotate(ref_keypoints, input_keypoints, 3, 4)      #rotate left wrist
            input_keypoints = comfy_utils.scale(ref_keypoints, input_keypoints, 2, 3, 3, 4) #scaling w.r.t to elbow to wrist ratio of ref pose

            #moving to hand ponts to wrist 
            input_keypoints[109:] = comfy_utils.move(input_keypoints[109], input_keypoints[4], input_keypoints[109:]) 
            input_keypoints[109:] = comfy_utils.move(ref_keypoints[4], ref_keypoints[109], input_keypoints[109:])

            # input_keypoints = comfy_utils.rotate_hand(ref_keypoints, input_keypoints, 109)    #rotating left hand
            input_keypoints = comfy_utils.scale_hand(ref_keypoints, input_keypoints, 3, 4, 109)    #scaling left hand w.r.t left wrist

            if garment_category not in ["half_sleeve_garment", "full_sleeve_garment"]:
                input_keypoints = comfy_utils.rotate(ref_keypoints, input_keypoints, 5, 6)      #rotate right elbow
            input_keypoints = comfy_utils.scale(ref_keypoints, input_keypoints, 2, 5, 5, 6) #scaling w.r.t to shoulder to elbow ratio of ref pose

            prev_rw = input_keypoints[7]
            if garment_category != "full_sleeve_garment":
                input_keypoints = comfy_utils.rotate(ref_keypoints, input_keypoints, 6, 7)      #rotate right wrist
            input_keypoints = comfy_utils.scale(ref_keypoints, input_keypoints, 5, 6, 6, 7)    #scaling w.r.t to elbow to wrist ratio of ref pose

            #moving hand points to wrist
            input_keypoints[88:109] = comfy_utils.move(input_keypoints[88], input_keypoints[7], input_keypoints[88:109])
            input_keypoints[88:109] = comfy_utils.move(ref_keypoints[7], ref_keypoints[88], input_keypoints[88:109])

            # input_keypoints = comfy_utils.rotate_hand(ref_keypoints, input_keypoints, 88)    #rotating right hand
            input_keypoints = comfy_utils.scale_hand(ref_keypoints, input_keypoints, 6, 7, 88)    #scaling right hand w.r.t right wrist

            #legs
            if garment_category not in ["trouser", "shorts"]:
                input_keypoints = comfy_utils.rotate(ref_keypoints, input_keypoints, 8, 9)      #rotate left knee
            input_keypoints = comfy_utils.scale(ref_keypoints, input_keypoints, 1, 8, 8, 9)      #scale left knee

            if garment_category != "trouser":
                input_keypoints = comfy_utils.rotate(ref_keypoints, input_keypoints, 9, 10)     #rotate left foot
            input_keypoints = comfy_utils.scale(ref_keypoints, input_keypoints, 8, 9, 9, 10)    #scaling w.r.t to knee to foot ratio of ref pose

            if garment_category not in ["half_sleeve_garment", "full_sleeve_garment"]:
                input_keypoints = comfy_utils.rotate(ref_keypoints, input_keypoints, 11, 12)     #rotate right knee
            input_keypoints = comfy_utils.scale(ref_keypoints, input_keypoints, 1, 11, 11, 12)      #scale right knee

            if garment_category != "trouser":
                input_keypoints = comfy_utils.rotate(ref_keypoints, input_keypoints, 12, 13)    #rotate right foot
            input_keypoints = comfy_utils.scale(ref_keypoints, input_keypoints, 11, 12, 12, 13)    #scaling w.r.t to knee to foot ratio of ref pose

            #face
            input_keypoints[14:18] = ref_keypoints[14:18]
            input_keypoints[0] = ref_keypoints[0]


        
            if input_keypoints[0] == [-1,-1] and input_keypoints[14] == [-1,-1] and input_keypoints[15] == [-1,-1]:
                input_keypoints = comfy_utils.rotate(ref_keypoints, input_keypoints, 1, 16)        #rotate left ear
                input_keypoints = comfy_utils.scale(ref_keypoints, input_keypoints, 2, 5, 1, 16)        #scaling w.r.t to neck len to ear_nose len ratio of ref pose

                input_keypoints = comfy_utils.rotate(ref_keypoints, input_keypoints, 1,17)        #rotate right ear
                input_keypoints = comfy_utils.scale(ref_keypoints, input_keypoints, 2, 5, 1, 17)        #scaling w.r.t to neck len to ear_nose len ratio of ref pose
            else:
                prev_nose = input_keypoints[0]
                input_keypoints = comfy_utils.rotate(ref_keypoints, input_keypoints, 1, 0)      #rotate nose
                input_keypoints = comfy_utils.scale(ref_keypoints, input_keypoints, 2, 5, 1, 0)      #scale neck to nose w.r.t to shoulder neck-nose len ratio of ref pose
                
                #changing face points to w.r.t to new nose point after rotation
                input_keypoints[14:18] = comfy_utils.move(prev_nose, input_keypoints[0], input_keypoints[14:18])

                if input_keypoints[14] != [-1,-1]: 
                    input_keypoints = comfy_utils.rotate(ref_keypoints, input_keypoints, 0, 14)      #rotate left eye
                    input_keypoints = comfy_utils.scale(ref_keypoints, input_keypoints, 1, 0, 0, 14)        #scaling w.r.t to neck len to eye_nose len ratio of ref pose

                if input_keypoints[15] != [-1,-1]:
                    input_keypoints = comfy_utils.rotate(ref_keypoints, input_keypoints, 0, 15)      #rotate right eye
                    input_keypoints = comfy_utils.scale(ref_keypoints, input_keypoints, 1, 0, 0, 15)        #scaling w.r.t to neck len to eye_nose len ratio of ref pose

                if input_keypoints[16] != [-1,-1]:
                    input_keypoints = comfy_utils.rotate(ref_keypoints, input_keypoints, 14, 16)        #rotate left ear
                    input_keypoints = comfy_utils.scale(ref_keypoints, input_keypoints, 1, 0, 14, 16)        #scaling w.r.t to neck len to ear_nose len ratio of ref pose

                if input_keypoints[17] != [-1,-1]:
                    input_keypoints = comfy_utils.rotate(ref_keypoints, input_keypoints, 15,17)        #rotate right ear
                    input_keypoints = comfy_utils.scale(ref_keypoints, input_keypoints, 1, 0, 15, 17)        #scaling w.r.t to neck len to ear_nose len ratio of ref pose

            canvas = comfy_utils.draw_bodypose(canvas, input_keypoints)
            canvas = comfy_utils.draw_handpose(canvas, input_keypoints[88:109])  #right hand
            canvas = comfy_utils.draw_handpose(canvas, input_keypoints[109:])  #left hand
            canvas = torch.from_numpy(canvas.astype(np.float32)/255.0)[None,]
            return (canvas, similar_torso)


class TRI3DLoadPoseJson:

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [
            f for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]
        return {
            "required": {
                "image": (sorted(files), {
                    "upload": True
                })
            },
        }

    CATEGORY = "TRI3D"

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "load_image"

    def load_image(self, image):
        # image_path = folder_paths.get_annotated_filepath(image)
        # i = Image.open(image_path)
        # i = ImageOps.exif_transpose(i)
        # image = i.convert("RGB")
        if image.endswith('.json'):
            # with open(image) as open:
            pose = json.load(open(image))
            height = pose['height']
            width = pose['width']
            keypoints = pose['keypoints']

            canvas = np.zeros(shape=(height, width, 3), dtype=np.uint8)
            canvas = comfy_utils.draw_bodypose(canvas, keypoints)
            canvas = comfy_utils.draw_handpose(canvas, keypoints[88:109])  #right hand
            canvas = comfy_utils.draw_handpose(canvas, keypoints[109:])  #left hand
            canvas = torch.from_numpy(canvas.astype(np.float32)/255.0)[None,]
        else:
            canvas = np.zeros(shape=(height, width, 3), dtype=np.uint8)
        # if 'A' in i.getbands():
        #     mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
        #     mask = 1. - torch.from_numpy(mask)
        # else:
        #     mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        return (canvas, )


class TRI3DFaceRecognise:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image1": ("IMAGE", ), "image2": ("IMAGE", )}}

    RETURN_TYPES = ("FLOAT", )
    FUNCTION = "encode"
    CATEGORY = "TRI3D"

    def encode(self, image1, image2):
        import cv2
        import numpy as np
        from insightface.app import FaceAnalysis
        import math

        image1 = image1.squeeze().cpu().numpy() * 255.0
        image2 = image2.squeeze().cpu().numpy() * 255.0

        image1 = np.clip(image1, 0, 255).astype(np.uint8)
        image2 = np.clip(image2, 0, 255).astype(np.uint8)

        app = FaceAnalysis(name='buffalo_l')
        app.prepare(ctx_id=0, det_size=(640, 640))

        face1 = app.get(image1)
        face2 = app.get(image2)

        embedding1 = face1[0]['embedding']
        embedding2 = face2[0]['embedding']

        embedding1 /= math.sqrt(embedding1.dot(embedding1))
        embedding2 /= math.sqrt(embedding2.dot(embedding2))

        s = embedding1.dot(embedding2)
        return ({"overlap (float)": s}, )


class FloatToImage:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                })
            }
        }

    CATEGORY = "TRI3D"
    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "load_image"

    def load_image(self, value):

        def render_float(float_input):
            latex_expression = '$' + str(float_input) + '$'
            import matplotlib.pyplot as plt

            fig = plt.figure(
                figsize=(10, 4))  # Dimensions of figsize are in inches

            text = fig.text(
                x=0.5,  # x-coordinate to place the text
                y=0.5,  # y-coordinate to place the text
                s=latex_expression,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=32,
            )

            import tempfile
            path_file_image_output = tempfile.NamedTemporaryFile(
            ).name + '.png'

            plt.savefig(path_file_image_output)

            import cv2
            image = cv2.imread(path_file_image_output, cv2.IMREAD_COLOR)

            import os
            # os.unlink(path_file_image_output)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image

        def to_torch_image(image):

            import numpy as np
            import torch
            image = image.astype(dtype=np.float32)
            image /= 255.0
            image = torch.from_numpy(image)[
                None,
            ]
            image = image.unsqueeze(0)
            return image

        image = render_float(float_input=value)
        image = to_torch_image(image)

        return image



class TRI3D_recolor_LAB_manual:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_input": ("IMAGE", ),
                "mask_input": ("IMAGE", ),
                "factor_mean": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01
                }),
                "factor_sigma": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "recolor"
    CATEGORY = "TRI3D"

    def recolor(self, image_input, mask_input, factor_mean, factor_sigma):

        def get_mu_sigma(array_input, mask_input):

            import numpy as np
            import math

            array_input = array_input.astype(dtype=np.float32).flatten()
            mask_input = mask_input.flatten()

            sum = np.sum(mask_input)
            mean = np.sum(array_input * mask_input) / sum

            array_input -= mean
            array_input *= mask_input
            sigma = math.sqrt(np.sum(np.square(array_input)) / sum)

            return mean, sigma

        def do_recolor(image, mask, mean, sigma):

            import cv2
            import numpy as np
            import math

            image_original = image.copy()

            mask = (mask > 127).astype(dtype=np.uint8)

            sum = np.sum(mask.flatten())

            for i in range(3):
                image[:, :, i] *= mask

            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

            image = image.astype(dtype=np.float32)

            for i in range(1):

                mu_1, sigma_1 = get_mu_sigma(array_input=image[:, :, i],
                                             mask_input=mask)

                image[:, :, i] = (((image[:, :, i] - mu_1) / sigma_1) *
                                  (sigma * sigma_1)) + (mean * mu_1)

            image = np.clip(image, 0, 255)
            image = image.astype(dtype=np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

            for i in range(3):

                image_original[:, :,
                               i] = (image_original[:, :, i] *
                                     (1 - mask)) + (image[:, :, i] * mask)

            return image_original

        def from_torch_image(image):

            image = image.squeeze().cpu().numpy() * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)

            return image

        def to_torch_image(image):

            import numpy as np
            import torch
            image = image.astype(dtype=np.float32)
            image /= 255.0
            image = torch.from_numpy(image)[
                None,
            ]
            image = image.unsqueeze(0)
            return image

        image = from_torch_image(image=image_input)

        mask = from_torch_image(image=mask_input)[:, :, 0]

        image_output = do_recolor(image=image,
                                  mask=mask,
                                  mean=factor_mean,
                                  sigma=factor_sigma)

        image_output = to_torch_image(image=image_output)

        return image_output



class TRI3D_reLUM:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_reference": ("IMAGE", ),
                "image_mask_reference": ("IMAGE", ),
                "image_recolor": ("IMAGE", ),
                "image_mask_recolor": ("IMAGE", ),
                "factor_mean_L": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01
                }),
                "factor_sigma_L": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "recolor"
    CATEGORY = "TRI3D"

    def recolor(
        self,
        image_reference,
        image_mask_reference,
        image_recolor,
        image_mask_recolor,
        factor_mean_L,
        factor_sigma_L,
    ):

        def get_mu_sigma(array_input, mask_input):

            import numpy as np
            import math

            array_input = array_input.astype(dtype=np.float32).flatten()
            mask_input = mask_input.flatten()

            sum = np.sum(mask_input)
            mean = np.sum(array_input * mask_input) / sum

            array_input -= mean
            array_input *= mask_input
            sigma = math.sqrt(np.sum(np.square(array_input)) / sum)

            return mean, sigma

        def do_recolor(image_1, mask_1, image_2, mask_2):

            import cv2
            import numpy as np
            import math

            image_2_original = image_2.copy()

            mask_1 = (mask_1 > 127).astype(dtype=np.uint8)
            mask_2 = (mask_2 > 127).astype(dtype=np.uint8)

            sum_1 = np.sum(mask_1.flatten())
            sum_2 = np.sum(mask_2.flatten())

            for i in range(3):
                image_1[:, :, i] *= mask_1
                image_2[:, :, i] *= mask_2

            image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2LAB)
            image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2LAB)

            image_1 = image_1.astype(dtype=np.float32)
            image_2 = image_2.astype(dtype=np.float32)

            factor_mean = (factor_mean_L, 1, 1)
            factor_sigma = (factor_sigma_L, 1, 1)

            for i in range(1):

                mu_1, sigma_1 = get_mu_sigma(array_input=image_1[:, :, i],
                                             mask_input=mask_1)

                mu_2, sigma_2 = get_mu_sigma(array_input=image_2[:, :, i],
                                             mask_input=mask_2)

                image_2[:, :, i] = (
                    ((image_2[:, :, i] - mu_2) / sigma_2) *
                    (sigma_1 * factor_sigma[i])) + (mu_1 * factor_mean[i])

            image_2 = np.clip(image_2, 0, 255)
            image_2 = image_2.astype(dtype=np.uint8)
            image_2 = cv2.cvtColor(image_2, cv2.COLOR_LAB2BGR)

            for i in range(3):

                image_2_original[:, :,
                                 i] = (image_2_original[:, :, i] *
                                       (1 - mask_2)) + (image_2[:, :, i] *
                                                        mask_2)

            return image_2_original

        def from_torch_image(image):

            image = image.squeeze().cpu().numpy() * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)

            return image

        def to_torch_image(image):

            import numpy as np
            import torch
            image = image.astype(dtype=np.float32)
            image /= 255.0
            image = torch.from_numpy(image)[
                None,
            ]
            image = image.unsqueeze(0)
            return image

        image_reference = from_torch_image(image=image_reference)

        image_mask_reference = from_torch_image(
            image=image_mask_reference)[:, :, 0]

        image_recolor = from_torch_image(image=image_recolor)

        image_mask_recolor = from_torch_image(image=image_mask_recolor)[:, :,
                                                                        0]

        image_output = do_recolor(image_1=image_reference,
                                  mask_1=image_mask_reference,
                                  image_2=image_recolor,
                                  mask_2=image_mask_recolor)

        image_output = to_torch_image(image=image_output)

        return image_output




class TRI3D_recolor_LAB:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_reference": ("IMAGE", ),
                "image_mask_reference": ("IMAGE", ),
                "image_recolor": ("IMAGE", ),
                "image_mask_recolor": ("IMAGE", ),
                "factor_mean_L": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01
                }),
                "factor_sigma_L": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01
                }),
                "factor_mean_A": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01
                }),
                "factor_sigma_A": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01
                }),
                "factor_mean_B": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01
                }),
                "factor_sigma_B": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "recolor"
    CATEGORY = "TRI3D"

    def recolor(
        self,
        image_reference,
        image_mask_reference,
        image_recolor,
        image_mask_recolor,
        factor_mean_L,
        factor_sigma_L,
        factor_mean_A,
        factor_sigma_A,
        factor_mean_B,
        factor_sigma_B,
    ):

        def get_mu_sigma(array_input, mask_input):

            import numpy as np
            import math

            array_input = array_input.astype(dtype=np.float32).flatten()
            mask_input = mask_input.flatten()

            sum = np.sum(mask_input)
            mean = np.sum(array_input * mask_input) / sum

            array_input -= mean
            array_input *= mask_input
            sigma = math.sqrt(np.sum(np.square(array_input)) / sum)

            return mean, sigma

        def do_recolor(image_1, mask_1, image_2, mask_2):

            import cv2
            import numpy as np
            import math

            image_2_original = image_2.copy()

            mask_1 = (mask_1 > 127).astype(dtype=np.uint8)
            mask_2 = (mask_2 > 127).astype(dtype=np.uint8)

            sum_1 = np.sum(mask_1.flatten())
            sum_2 = np.sum(mask_2.flatten())

            for i in range(3):
                image_1[:, :, i] *= mask_1
                image_2[:, :, i] *= mask_2

            image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2LAB)
            image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2LAB)

            image_1 = image_1.astype(dtype=np.float32)
            image_2 = image_2.astype(dtype=np.float32)

            factor_mean = (factor_mean_L, factor_mean_A, factor_mean_B)
            factor_sigma = (factor_sigma_L, factor_sigma_A, factor_sigma_B)

            for i in range(3):

                mu_1, sigma_1 = get_mu_sigma(array_input=image_1[:, :, i],
                                             mask_input=mask_1)

                mu_2, sigma_2 = get_mu_sigma(array_input=image_2[:, :, i],
                                             mask_input=mask_2)

                image_2[:, :, i] = (
                    ((image_2[:, :, i] - mu_2) / sigma_2) *
                    (sigma_1 * factor_sigma[i])) + (mu_1 * factor_mean[i])

            image_2 = np.clip(image_2, 0, 255)
            image_2 = image_2.astype(dtype=np.uint8)
            image_2 = cv2.cvtColor(image_2, cv2.COLOR_LAB2BGR)

            for i in range(3):

                image_2_original[:, :,
                                 i] = (image_2_original[:, :, i] *
                                       (1 - mask_2)) + (image_2[:, :, i] *
                                                        mask_2)

            return image_2_original

        def from_torch_image(image):

            image = image.squeeze().cpu().numpy() * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)

            return image

        def to_torch_image(image):

            import numpy as np
            import torch
            image = image.astype(dtype=np.float32)
            image /= 255.0
            image = torch.from_numpy(image)[
                None,
            ]
            image = image.unsqueeze(0)
            return image

        image_reference = from_torch_image(image=image_reference)

        image_mask_reference = from_torch_image(
            image=image_mask_reference)[:, :, 0]

        image_recolor = from_torch_image(image=image_recolor)

        image_mask_recolor = from_torch_image(image=image_mask_recolor)[:, :,
                                                                        0]

        image_output = do_recolor(image_1=image_reference,
                                  mask_1=image_mask_reference,
                                  image_2=image_recolor,
                                  mask_2=image_mask_recolor)

        image_output = to_torch_image(image=image_output)

        return image_output


class TRI3D_recolor_RGB:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_reference": ("IMAGE", ),
                "image_mask_reference": ("IMAGE", ),
                "image_recolor": ("IMAGE", ),
                "image_mask_recolor": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "recolor"
    CATEGORY = "TRI3D"

    def recolor(self, image_reference, image_mask_reference, image_recolor,
                image_mask_recolor):

        def get_mu_sigma(array_input, mask_input):

            import numpy as np
            import math

            array_input = array_input.astype(dtype=np.float32).flatten()
            mask_input = mask_input.flatten()

            sum = np.sum(mask_input)
            mean = np.sum(array_input * mask_input) / sum

            array_input -= mean
            array_input *= mask_input
            sigma = math.sqrt(np.sum(np.square(array_input)) / sum)

            return mean, sigma

        def do_recolor(image_1, mask_1, image_2, mask_2):

            import cv2
            import numpy as np
            import math

            image_2_original = image_2.copy()

            mask_1 = (mask_1 > 127).astype(dtype=np.uint8)
            mask_2 = (mask_2 > 127).astype(dtype=np.uint8)

            sum_1 = np.sum(mask_1.flatten())
            sum_2 = np.sum(mask_2.flatten())

            for i in range(3):
                image_1[:, :, i] *= mask_1
                image_2[:, :, i] *= mask_2

            image_1 = image_1.astype(dtype=np.float32)
            image_2 = image_2.astype(dtype=np.float32)

            for i in range(3):

                mu_1, sigma_1 = get_mu_sigma(array_input=image_1[:, :, i],
                                             mask_input=mask_1)

                mu_2, sigma_2 = get_mu_sigma(array_input=image_2[:, :, i],
                                             mask_input=mask_2)

                image_2[:, :, i] = ((
                    (image_2[:, :, i] - mu_2) / sigma_2) * sigma_1) + mu_1

            image_2 = np.clip(image_2, 0, 255)
            image_2 = image_2.astype(dtype=np.uint8)

            for i in range(3):

                image_2_original[:, :,
                                 i] = (image_2_original[:, :, i] *
                                       (1 - mask_2)) + (image_2[:, :, i] *
                                                        mask_2)

            return image_2_original

        def from_torch_image(image):

            image = image.squeeze().cpu().numpy() * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)

            return image

        def to_torch_image(image):

            import numpy as np
            import torch
            image = image.astype(dtype=np.float32)
            image /= 255.0
            image = torch.from_numpy(image)[
                None,
            ]
            image = image.unsqueeze(0)
            return image

        image_reference = from_torch_image(image=image_reference)

        image_mask_reference = from_torch_image(
            image=image_mask_reference)[:, :, 0]

        image_recolor = from_torch_image(image=image_recolor)

        image_mask_recolor = from_torch_image(image=image_mask_recolor)[:, :,
                                                                        0]

        image_output = do_recolor(image_1=image_reference,
                                  mask_1=image_mask_reference,
                                  image_2=image_recolor,
                                  mask_2=image_mask_recolor)

        image_output = to_torch_image(image=image_output)

        return image_output


class TRI3D_recolor:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_reference": ("IMAGE", ),
                "image_mask_reference": ("IMAGE", ),
                "image_recolor": ("IMAGE", ),
                "image_mask_recolor": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "recolor"
    CATEGORY = "TRI3D"

    def recolor(self, image_reference, image_mask_reference, image_recolor,
                image_mask_recolor):

        def get_mu_sigma(array_input, mask_input):

            import numpy as np
            import math

            array_input = array_input.astype(dtype=np.float32).flatten()
            mask_input = mask_input.flatten()

            sum = np.sum(mask_input)
            mean = np.sum(array_input * mask_input) / sum

            array_input -= mean
            array_input *= mask_input
            sigma = math.sqrt(np.sum(np.square(array_input)) / sum)

            return mean, sigma

        def do_recolor(image_1, mask_1, image_2, mask_2):

            import cv2
            import numpy as np
            import math

            image_2_original = image_2.copy()

            mask_1 = (mask_1 > 127).astype(dtype=np.uint8)
            mask_2 = (mask_2 > 127).astype(dtype=np.uint8)

            sum_1 = np.sum(mask_1.flatten())
            sum_2 = np.sum(mask_2.flatten())

            for i in range(3):
                image_1[:, :, i] *= mask_1
                image_2[:, :, i] *= mask_2

            image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2HSV_FULL)
            image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2HSV_FULL)

            image_1 = image_1.astype(dtype=np.float32)
            image_2 = image_2.astype(dtype=np.float32)

            for i in range(3):

                mu_1, sigma_1 = get_mu_sigma(array_input=image_1[:, :, i],
                                             mask_input=mask_1)

                mu_2, sigma_2 = get_mu_sigma(array_input=image_2[:, :, i],
                                             mask_input=mask_2)

                image_2[:, :, i] = ((
                    (image_2[:, :, i] - mu_2) / sigma_2) * sigma_1) + mu_1

            image_2 = image_2.astype(dtype=np.int16)
            image_2[:, :, 0] %= 255
            image_2 = np.clip(image_2, 0, 255)
            image_2 = image_2.astype(dtype=np.uint8)
            image_2 = cv2.cvtColor(image_2, cv2.COLOR_HSV2BGR_FULL)

            for i in range(3):

                image_2_original[:, :,
                                 i] = (image_2_original[:, :, i] *
                                       (1 - mask_2)) + (image_2[:, :, i] *
                                                        mask_2)

            return image_2_original

        def from_torch_image(image):

            image = image.squeeze().cpu().numpy() * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)

            return image

        def to_torch_image(image):

            import numpy as np
            import torch
            image = image.astype(dtype=np.float32)
            image /= 255.0
            image = torch.from_numpy(image)[
                None,
            ]
            image = image.unsqueeze(0)
            return image

        image_reference = from_torch_image(image=image_reference)

        image_mask_reference = from_torch_image(
            image=image_mask_reference)[:, :, 0]

        image_recolor = from_torch_image(image=image_recolor)

        image_mask_recolor = from_torch_image(image=image_mask_recolor)[:, :,
                                                                        0]

        image_output = do_recolor(image_1=image_reference,
                                  mask_1=image_mask_reference,
                                  image_2=image_recolor,
                                  mask_2=image_mask_recolor)

        image_output = to_torch_image(image=image_output)

        return image_output




class TRI3D_image_mask_2_box:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "mask": ("MASK", ),
            },
        }

    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", )
    CATEGORY = "TRI3D"

    def run(self, image, mask):
        image = from_torch_image(image)
        mask = from_torch_image(mask)
        image = extract_box_from_image(image, mask)
        image = to_torch_image(image)
        return image


class TRI3D_image_mask_box_2_image:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "mask": ("MASK", ),
                "box": ("IMAGE", ),
            },
        }

    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", )
    CATEGORY = "TRI3D"

    def run(self, image, mask, box):
        image = from_torch_image(image)
        mask = from_torch_image(mask)
        box = from_torch_image(box)
        image = stitch_back_box_to_image(image, mask, box)
        image = to_torch_image(image)
        return image

class TRI3D_clipdrop_bgremove_api:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", )
            },
        }

    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", )
    CATEGORY = "TRI3D"

    def run(self, image):
        image = from_torch_image(image)
        print(image.shape)
        _, enc_image = cv2.imencode('.jpg', image)
        from dotenv import load_dotenv
        load_dotenv()
        CLIPDROP_API_KEY = os.getenv('CLIPDROP_API_KEY')

        # CLIPDROP_API_KEY = os.environ.get('CLIPDROP_API_KEY')
        
        r = requests.post('https://clipdrop-api.co/remove-background/v1',
        files = {
            'image_file': ("mannequin.jpg", enc_image.tobytes(), 'image/jpeg'),
            },
        headers = { 'x-api-key': CLIPDROP_API_KEY}
        )
        if (r.ok):
            pass
        else:
            r.raise_for_status()
        output = np.array(Image.open(io.BytesIO(r.content)))
        output = cv2.cvtColor(output, cv2.COLOR_BGRA2RGBA)
        # print("decoded output",output.shape)
        # mask = output[:,:,3]
        # output = output[:,:,0:3]
        output = torch.from_numpy(output.astype(np.float32)/255.0)[None,]
        # print("converted image to torch")
        # print(output.shape)
        # mask = torch.from_numpy(mask.astype(np.float32)/255.0)[None,]
        # print("converted mask to torch")
        # print(mask.shape)
        return output,

class TRI3DAdjustNeck:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "posemap_json_file_path": ("STRING", {"default": "dwpose/keypoints/input.json"}),
                # "age_group" :(["10-12 yrs"],{"default":"10-12 yrs"}),
                "neck_shoulder_ratio": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "save_json_file_path": ("STRING", {"default": "dwpose/keypoints/output.json"})
            },
        }

    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "STRING")
    CATEGORY = "TRI3D"


    def run(self, posemap_json_file_path, neck_shoulder_ratio, save_json_file_path):
        from .dwpose import comfy_utils

        # age_to_ratio = {"10-12 yrs": 0.65}
        
        input_pose = json.load(open(posemap_json_file_path))
        input_height = input_pose['height']
        input_width = input_pose['width']
        input_keypoints = input_pose['keypoints']

        ref_x1, ref_y1 = input_keypoints[2]                   #left shoulder
        ref_x2, ref_y2 = input_keypoints[5]                   #right_shoulder

        x1,y1 = input_keypoints[1]                       #neck
        x2,y2 = input_keypoints[0]                       #nose
        prev_nose = input_keypoints[0]

        ref_len = np.linalg.norm(np.array([ref_x1, ref_y1]) - np.array([ref_x2, ref_y2]))             #ref body part length i.e. shoulder length
        targ_len = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))             #targ body part length i.e. neck length - neck to nose length

        input_targ_ref_len = targ_len / ref_len               #neck to shoulder ratio
        print("neck to shoulder ratio found", input_targ_ref_len)
        print("neck to shoulder ratio target", neck_shoulder_ratio)
        #scale the coords
        # scale = age_to_ratio[age_group] / input_targ_ref_len                
        scale = neck_shoulder_ratio / input_targ_ref_len

        x2_scaled = x1 + (x2-x1) * scale
        y2_scaled = y1 + (y2-y1) * scale
        input_keypoints[0] = [x2_scaled, y2_scaled]
        
        #changing face points to w.r.t to new nose point after rotation
        input_keypoints[14:18] = comfy_utils.move(prev_nose, input_keypoints[0], input_keypoints[14:18])

        canvas = np.zeros(shape=(input_height, input_width, 3), dtype=np.uint8)
        canvas = comfy_utils.draw_bodypose(canvas, input_keypoints)
        canvas = comfy_utils.draw_handpose(canvas, input_keypoints[88:109])  #right hand
        canvas = comfy_utils.draw_handpose(canvas, input_keypoints[109:])  #left hand
        canvas = torch.from_numpy(canvas.astype(np.float32)/255.0)[None,]

        output_pose = {"height":input_height, "width":input_width, "keypoints":input_keypoints}
        cur_file_dir = os.path.dirname(os.path.realpath(__file__))
        save_json_file_path = os.path.join(cur_file_dir,
                                        save_json_file_path)
        json.dump(output_pose, open(save_json_file_path, 'w'))
        return (canvas, save_json_file_path)



class HistogramEqualization:
    """
    This node provides a simple interface to equalize the histogram of the output image.
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        """
        Input Types
        """
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_filter"

    CATEGORY = "Image Processing"

    def apply_filter(self, image, strength):

        # Convert the input image tensor to a PIL Image
        i = 255. * image.cpu().numpy().squeeze()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        # Equalize the histogram of the image
        equalized_img = ImageOps.equalize(img)

        # Blend the original image with the equalized image based on the strength
        blended_img = Image.blend(img, equalized_img, alpha=strength)

        # Convert the blended PIL Image back to a tensor
        blended_image_np = np.array(blended_img).astype(np.float32) / 255.0
        blended_image_tensor = torch.from_numpy(blended_image_np).unsqueeze(0)

        return (blended_image_tensor,)

class TRI3DCompositeImageSplitter:
    """
    This node splits composite (horizontally concatenated) image into half.
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        """
        Input Types
        """
        return {
            "required": {
                "images": ("IMAGE",)
            },
        }

    RETURN_TYPES = ("IMAGE","IMAGE")
    RETURN_NAMES = ("image1","image2")
    FUNCTION = "main"

    CATEGORY = "TRI3D"

    def main(self, images):
        
        def tensor_to_cv2_img(tensor, remove_alpha=False):
            i = 255. * tensor.cpu().numpy()  # This will give us (H, W, C)
            img = np.clip(i, 0, 255).astype(np.uint8)
            return img

        def cv2_img_to_tensor(img):
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img)[
                None,
            ]
            return img
        images1 = []
        images2 = []
        for image in images:
            image = tensor_to_cv2_img(image)
            h,w,_ = image.shape
            image1 = image[:,:w//2,:]
            image2 = image[:,w//2:,:]
            images1.append(cv2_img_to_tensor(image1).squeeze(0))
            images2.append(cv2_img_to_tensor(image2).squeeze(0))

        return (torch.stack(images1), torch.stack(images2))
    

class main_transparent_background():

    def from_torch_image(self, image):
        image = image.cpu().numpy() * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
        return image


    def to_torch_image(self, image):
        image = image.astype(dtype=np.float32)
        image /= 255.0
        image = torch.from_numpy(image)
        return image

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
    RETURN_TYPES = ("IMAGE", "MASK")
    CATEGORY = "TRI3D"

    def run(self, image):
        image = self.from_torch_image(image)
        image = get_transparent_background(image)
        mask = image[:, :, :, 3]
        image = image[:, :, :, 0:3]
        image = self.to_torch_image(image)
        mask = self.to_torch_image(mask)
        print('DEBUG', image.shape, mask.shape)
        return (image, mask)


ensure_package()


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "tri3d-atr-parse-batch": TRI3DATRParseBatch,
    'tri3d-extract-parts-batch': TRI3DExtractPartsBatch,
    "tri3d-extract-parts-batch2": TRI3DExtractPartsBatch2,
    "tri3d-position-parts-batch": TRI3DPositionPartsBatch,
    "tri3d-swap-pixels": TRI3DSwapPixels,
    "tri3d-skin-feathered-padded-mask": TRI3DSkinFeatheredPaddedMask,
    "tri3d-interaction-canny": TRI3DInteractionCanny,
    "tri3d-dwpose": TRI3DDWPose_Preprocessor,
    "tri3d-pose-to-image": TRI3DPosetoImage,
    "tri3d-pose-adaption": TRI3DPoseAdaption,
    "tri3d-load-pose-json": TRI3DLoadPoseJson,
    "tri3d-face-recognise": TRI3DFaceRecognise,
    "tri3d-float-to-image": FloatToImage,
    "tri3d-recolor-mask": TRI3D_recolor,
    "tri3d-recolor-mask-LAB_space_manual": TRI3D_recolor_LAB_manual,
    "tri3d-recolor-mask-LAB_space": TRI3D_recolor_LAB,
    "tri3d-recolor-mask-RGB_space": TRI3D_recolor_RGB,
    "tri3d-image-mask-2-box": TRI3D_image_mask_2_box,
    "tri3d-image-mask-box-2-image": TRI3D_image_mask_box_2_image,
    "tri3d-clipdrop-bgremove-api": TRI3D_clipdrop_bgremove_api,
    "tri3d-adjust-neck": TRI3DAdjustNeck,
    "tri3d-HistogramEqualization": HistogramEqualization,
    "tri3d-composite-image-splitter": TRI3DCompositeImageSplitter,
    'tri3d-main_transparent_background': main_transparent_background,
    'tri3d-scaled-paste': main_scaled_paste,
    'tri3d-luminosity-match': TRI3D_reLUM,
}

VERSION = "2.10.0"
# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "tri3d-atr-parse-batch": "ATR Parse Batch" + " v" + VERSION,
    'tri3d-extract-parts-batch': 'Extract Parts Batch' + " v" + VERSION,
    'tri3d-extract-parts-batch2': 'Extract Parts Batch 2' + " v" + VERSION,
    "tri3d-position-parts-batch": "Position Parts Batch" + " v" + VERSION,
    "tri3d-swap-pixels": "Swap Pixels by Mask" + " v" + VERSION,
    "tri3d-skin-feathered-padded-mask":
    "Skin Feathered Padded Mask" + " v" + VERSION,
    "tri3d-interaction-canny":
    "Garment Skin Interaction Canny" + " v" + VERSION,
    "tri3d-dwpose": "DWPose" + " v" + VERSION,
    "tri3d-pose-to-image": "Pose to Image" + " v" + VERSION,
    "tri3d-pose-adaption": "Pose Adaption" + " v" + VERSION,
    "tri3d-load-pose-json": "Load Pose Json" + " v" + VERSION,
    "tri3d-face-recognise": "Recognise face" + " v" + VERSION,
    "tri3d-float-to-image": "Render float" + " v" + VERSION,
    "tri3d-recolor-mask": "Recolor mask HSV space" + " v" + VERSION,
    "tri3d-recolor-mask-LAB_space_manual": "Recolor mask LAB space manual" + " v" + VERSION,
    "tri3d-recolor-mask-LAB_space": "Recolor mask LAB space" + " v" + VERSION,
    "tri3d-recolor-mask-RGB_space": "Recolor mask RGB space" + " v" + VERSION,
    "tri3d--image-mask-2-box": "Extract box from image" + " v" + VERSION,
    "tri3d-image-mask-box-2-image": "Stitch box to image" + " v" + VERSION,
    "tri3d-clipdrop-bgremove-api": "RemBG ClipDrop" + " v" + VERSION,
    "tri3d-adjust-neck": "Adjust Neck" + " v" + VERSION,
    "tri3d-HistogramEqualization": "Adjust Neck" + " v" + VERSION,
    "tri3d-composite-image-splitter": "Composite Image Splitter" + " v" + VERSION,
    'tri3d-main_transparent_background': 'Transparent Background' + " v" + VERSION,
    'tri3d-scaled-paste': 'Scaled paste' + " v" + VERSION,
    'tri3d-luminosity-match': 'Luminosity match' + " v" + VERSION,
}
