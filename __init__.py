# v1.1.0
import cv2
import numpy as np
import torch


class TRI3DATRParseBatch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
            },
        }
    RETURN_TYPES = ("IMAGE",)
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
            img = torch.from_numpy(img)[None,]
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
        os.system(
            "python simple_extractor.py --dataset atr --model-restore checkpoints/atr.pth --input-dir input --output-dir output")
        os.chdir(cwd)

        # Collect and return the results
        batch_results = []
        for i in range(images.shape[0]):
            cv2_segm = cv2.imread(ATR_OUTPUT_PATH + f'image{i}.png')
            cv2_segm = cv2.cvtColor(cv2_segm, cv2.COLOR_BGR2RGB)
            b_tensor_img = cv2_img_to_tensor(cv2_segm)
            batch_results.append(b_tensor_img.squeeze(0))

        batch_results = torch.stack(batch_results)

        return (batch_results,)


class TRI3DExtractPartsBatch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "batch_images": ("IMAGE",),
                "batch_segs": ("IMAGE",),
                "batch_secondaries": ("IMAGE",),
                "margin": ("INT", {"default": 15, "min": 0}),
                "right_leg": ("BOOLEAN", {"default": False}),
                "right_hand": ("BOOLEAN", {"default": True}),
                "head": ("BOOLEAN", {"default": False}),
                "hair": ("BOOLEAN", {"default": False}),
                "left_shoe": ("BOOLEAN", {"default": False}),
                "bag": ("BOOLEAN", {"default": False}),
                "background": ("BOOLEAN", {"default": False}),
                "dress": ("BOOLEAN", {"default": False}),
                "left_leg": ("BOOLEAN", {"default": False}),
                "right_shoe": ("BOOLEAN", {"default": False}),
                "left_hand": ("BOOLEAN", {"default": True}),
                "upper_garment": ("BOOLEAN", {"default": False}),
                "lower_garment": ("BOOLEAN", {"default": False}),
                "belt": ("BOOLEAN", {"default": False}),
                "skirt": ("BOOLEAN", {"default": False}),
                "hat": ("BOOLEAN", {"default": False}),
                "sunglasses": ("BOOLEAN", {"default": False}),
                "scarf": ("BOOLEAN", {"default": False}),
            },
        }
    RETURN_TYPES = ("IMAGE", "IMAGE",)
    FUNCTION = "main"
    CATEGORY = "TRI3D"

    def main(self, batch_images, batch_segs, batch_secondaries, margin, right_leg, right_hand, head, hair, left_shoe, bag, background, dress, left_leg, right_shoe, left_hand, upper_garment, lower_garment, belt, skirt, hat, sunglasses, scarf):
        import cv2
        import numpy as np
        import torch
        from pprint import pprint

        def get_segment_counts(segm):
            # Load the segmentation image

            # Reshape the image array to be 2D
            reshaped = segm.reshape(-1, segm.shape[-1])

            # Find unique vectors and their counts
            unique_vectors, counts = np.unique(
                reshaped, axis=0, return_counts=True)
            segment_counts = list(zip(unique_vectors, counts))
            pprint(segment_counts)
            return segment_counts

        def bounded_image(seg_img, color_code_list, input_img):
            import cv2
            import numpy as np
            # Create a mask for hands
            seg_img = cv2.resize(
                seg_img, (input_img.shape[1], input_img.shape[0]), interpolation=cv2.INTER_NEAREST)
            hand_mask = np.zeros_like(seg_img[:, :, 0])
            for color in color_code_list:
                lowerb = np.array(color, dtype=np.uint8)
                upperb = np.array(color, dtype=np.uint8)
                temp_mask = cv2.inRange(seg_img, lowerb, upperb)
                hand_mask = cv2.bitwise_or(hand_mask, temp_mask)

            # Find contours to get the bounding box of the hands
            contours, _ = cv2.findContours(
                hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
            w = min(w + 2*margin, input_img.shape[1] - x)
            # Ensure height does not exceed image boundary
            h = min(h + 2*margin, input_img.shape[0] - y)
            print(x, y, w, h, "x,y,w,h")
            print(input_img.shape, "input_img.shape")
            # Extract the region from the original image that contains both hands
            hand_region = input_img[y:y+h, x:x+w]

            return hand_region

        def tensor_to_cv2_img(tensor, remove_alpha=False):
            # This will give us (H, W, C)
            i = 255. * tensor.squeeze(0).cpu().numpy()
            img = np.clip(i, 0, 255).astype(np.uint8)
            return img

        def cv2_img_to_tensor(img):
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img)[None,]
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
                black_img = np.zeros(
                    (10, 10, num_channels), dtype=cv2_image.dtype)
                images.append(black_img)

            if bsecondary is not None:
                secondaries.append(bsecondary)
            else:
                num_channels = cv2_image.shape[2] if len(
                    cv2_image.shape) > 2 else 1
                black_img = np.zeros(
                    (10, 10, num_channels), dtype=cv2_image.dtype)
                secondaries.append(black_img)

        # Get max height and width
        max_height = max(img.shape[0] for img in images)
        max_width = max(img.shape[1] for img in images)

        batch_results = []
        batch_secondaries = []

        for img in images:
            # Resize the image to max height and width
            resized_img = cv2.resize(
                img, (max_width, max_height), interpolation=cv2.INTER_AREA)
            tensor_img = cv2_img_to_tensor(resized_img)
            batch_results.append(tensor_img.squeeze(0))

        for sec in secondaries:
            # Resize the image to max height and width
            resized_sec = cv2.resize(
                sec, (max_width, max_height), interpolation=cv2.INTER_AREA)
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
                "batch_images": ("IMAGE",),
                "batch_segs": ("IMAGE",),
                "batch_handimgs": ("IMAGE",),
                "margin": ("INT", {"default": 15, "min": 0}),
                "right_leg": ("BOOLEAN", {"default": False}),
                "right_hand": ("BOOLEAN", {"default": True}),
                "head": ("BOOLEAN", {"default": False}),
                "hair": ("BOOLEAN", {"default": False}),
                "left_shoe": ("BOOLEAN", {"default": False}),
                "bag": ("BOOLEAN", {"default": False}),
                "background": ("BOOLEAN", {"default": False}),
                "dress": ("BOOLEAN", {"default": False}),
                "left_leg": ("BOOLEAN", {"default": False}),
                "right_shoe": ("BOOLEAN", {"default": False}),
                "left_hand": ("BOOLEAN", {"default": True}),
                "upper_garment": ("BOOLEAN", {"default": False}),
                "lower_garment": ("BOOLEAN", {"default": False}),
                "belt": ("BOOLEAN", {"default": False}),
                "skirt": ("BOOLEAN", {"default": False}),
                "hat": ("BOOLEAN", {"default": False}),
                "sunglasses": ("BOOLEAN", {"default": False}),
                "scarf": ("BOOLEAN", {"default": False}),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "main"
    CATEGORY = "TRI3D"

    def main(self, batch_images, batch_segs, batch_handimgs, margin, right_leg, right_hand, head, hair, left_shoe, bag, background, dress, left_leg, right_shoe, left_hand, upper_garment, lower_garment, belt, skirt, hat, sunglasses, scarf):
        import cv2
        import numpy as np
        import torch
        from pprint import pprint

        def bounded_image_points(seg_img, color_code_list, input_img):
            import cv2
            import numpy as np
            # Create a mask for hands
            seg_img = cv2.resize(
                seg_img, (input_img.shape[1], input_img.shape[0]), interpolation=cv2.INTER_NEAREST)
            hand_mask = np.zeros_like(seg_img[:, :, 0])
            for color in color_code_list:
                lowerb = np.array(color, dtype=np.uint8)
                upperb = np.array(color, dtype=np.uint8)
                temp_mask = cv2.inRange(seg_img, lowerb, upperb)
                hand_mask = cv2.bitwise_or(hand_mask, temp_mask)

            # Find contours to get the bounding box of the hands
            contours, _ = cv2.findContours(
                hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # If no contours were found, just return None
            if not contours:
                return None

            # Combine all contours to find encompassing bounding box
            all_points = np.concatenate(contours, axis=0)
            x, y, w, h = cv2.boundingRect(all_points)
            x = max(x - margin, 0)
            y = max(y - margin, 0)
            # Ensure width does not exceed image boundary
            w = min(w + 2*margin, input_img.shape[1] - x)
            # Ensure height does not exceed image boundary
            h = min(h + 2*margin, input_img.shape[0] - y)

            return (x, y, w, h)

        def tensor_to_cv2_img(tensor, remove_alpha=False):
            # This will give us (H, W, C)
            i = 255. * tensor.squeeze(0).cpu().numpy()
            img = np.clip(i, 0, 255).astype(np.uint8)
            return img

        def cv2_img_to_tensor(img):
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img)[None,]
            return img

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

            positions = bounded_image_points(
                cv2_seg, color_code_list, cv2_image)

            try:
                cv2_handimg = tensor_to_cv2_img(handimg)
                cv2_handimg = cv2.resize(
                    cv2_handimg, (positions[2], positions[3]), interpolation=cv2.INTER_NEAREST)

                cv2_image[positions[1]:positions[1]+positions[3],
                          positions[0]:positions[0]+positions[2]] = cv2_handimg
            except Exception as e:
                print(e)
                pass
            b_tensor_img = cv2_img_to_tensor(cv2_image)
            batch_results.append(b_tensor_img.squeeze(0))

        batch_results = torch.stack(batch_results)

        return (batch_results,)


class TRI3DSwapPixels:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "from_image": ("IMAGE",),
                # "garment_mask": ("IMAGE",),
                "to_image": ("IMAGE",),
                "to_mask": ("IMAGE",),
                "swap_masked": ("BOOLEAN", {"default": False})
            },
        }
    RETURN_TYPES = ("IMAGE",)
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
            img = torch.from_numpy(img)[None,]
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
        return (result_image,)


class TRI3DExtractPartsBatch2:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "batch_images": ("IMAGE",),
                "batch_segs": ("IMAGE",),
                "batch_secondaries": ("IMAGE",),
                "margin": ("INT", {"default": 15, "min": 0}),
                "right_leg": ("BOOLEAN", {"default": False}),
                "right_hand": ("BOOLEAN", {"default": True}),
                "head": ("BOOLEAN", {"default": False}),
                "hair": ("BOOLEAN", {"default": False}),
                "left_shoe": ("BOOLEAN", {"default": False}),
                "bag": ("BOOLEAN", {"default": False}),
                "background": ("BOOLEAN", {"default": False}),
                "dress": ("BOOLEAN", {"default": False}),
                "left_leg": ("BOOLEAN", {"default": False}),
                "right_shoe": ("BOOLEAN", {"default": False}),
                "left_hand": ("BOOLEAN", {"default": True}),
                "upper_garment": ("BOOLEAN", {"default": False}),
                "lower_garment": ("BOOLEAN", {"default": False}),
                "belt": ("BOOLEAN", {"default": False}),
                "skirt": ("BOOLEAN", {"default": False}),
                "hat": ("BOOLEAN", {"default": False}),
                "sunglasses": ("BOOLEAN", {"default": False}),
                "scarf": ("BOOLEAN", {"default": False}),
            },
        }
    RETURN_TYPES = ("IMAGE", "IMAGE")
    FUNCTION = "main"
    CATEGORY = "TRI3D"

    def main(self, batch_images, batch_segs, batch_secondaries, margin, right_leg, right_hand, head, hair, left_shoe, bag, background, dress, left_leg, right_shoe, left_hand, upper_garment, lower_garment, belt, skirt, hat, sunglasses, scarf):
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
            img = torch.from_numpy(img)[None,]
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

            images.append(mask*cv2_image)
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
            resized_img = cv2.resize(
                img, (max_width, max_height), interpolation=cv2.INTER_AREA)
            # print(img.shape, "before tensor_img.shape")
            tensor_img = cv2_img_to_tensor(resized_img)
            # print(tensor_img.shape, "tensor_img.shape")
            batch_results.append(tensor_img.squeeze(0))

        for sec in secondaries:
            # Resize the image to max height and width
            resized_sec = cv2.resize(
                sec, (max_width, max_height), interpolation=cv2.INTER_AREA)
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
                "garment_masks": ("IMAGE",),
                "first_pass_masks": ("IMAGE",),
                "padding_margin": ("INT", {"default": 20, "min": 0},)
            },
        }
    RETURN_TYPES = ("IMAGE",)
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
            img = torch.from_numpy(img)[None,]
            return img

        results = []
        for i in range(first_pass_masks.shape[0]):

            garment_mask = tensor_to_cv2_img(garment_masks[i])
            # first_pass_image = tensor_to_cv2_img(first_pass_images[i])
            first_pass_mask = tensor_to_cv2_img(first_pass_masks[i])

            h, w, _ = first_pass_mask.shape

            garment_mask = cv2.resize(garment_mask, (w, h))

            garment_mask = np.where(garment_mask == 0, 1, 0).astype("bool")
            first_pass_mask = np.where(
                first_pass_mask == 0, 1, 0).astype('bool')

            fp_dilate = cv2.dilate(first_pass_mask.astype("uint8"), np.ones(
                (padding_margin, padding_margin), np.uint8), iterations=1)
            og_dilate = cv2.dilate(garment_mask.astype(
                "uint8"), np.ones((30, 30), np.uint8), iterations=1)
            fp_erode = cv2.erode(first_pass_mask.astype(
                "uint8"), np.ones((25, 25), np.uint8), iterations=1)
            result = (fp_dilate ^ fp_erode)*og_dilate

            result = np.where(result == 0, 0, 255)
            results.append(result)
            # print(mask.shape)

        # Get max height and width
        max_height = max(img.shape[0] for img in results)
        max_width = max(img.shape[1] for img in results)

        batch_results = []

        for img in results:
            # Resize the image to max height and width
            resized_img = cv2.resize(
                img, (max_width, max_height), interpolation=cv2.INTER_AREA)
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
                "garment_masks": ("IMAGE",),
                "first_pass_images": ("IMAGE",),
                "first_pass_masks": ("IMAGE",),
                "lower_threshold": ("INT", {"default": 80, "min": 0},),
                "higher_threshold": ("INT", {"default": 240, "min": 0},)
            },
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "main"
    CATEGORY = "TRI3D"

    def main(self, garment_masks, first_pass_images, first_pass_masks, lower_threshold, higher_threshold):
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
            img = torch.from_numpy(img)[None,]
            return img

        results = []
        for i in range(first_pass_masks.shape[0]):

            garment_mask = tensor_to_cv2_img(garment_masks[i])
            first_pass_image = tensor_to_cv2_img(first_pass_images[i])
            first_pass_mask = tensor_to_cv2_img(first_pass_masks[i])

            h, w, _ = first_pass_mask.shape

            garment_mask = cv2.resize(garment_mask, (w, h))

            garment_mask = np.where(garment_mask == 0, 1, 0).astype("bool")
            first_pass_mask = np.where(
                first_pass_mask == 0, 1, 0).astype('bool')

            canny = cv2.Canny(first_pass_image,
                              lower_threshold, higher_threshold)
            canny = np.dstack((canny, canny, canny))

            result = (garment_mask*first_pass_mask).astype("uint8")

            result = result*canny
            results.append(result)
            # print(mask.shape)

        # Get max height and width
        max_height = max(img.shape[0] for img in results)
        max_width = max(img.shape[1] for img in results)

        batch_results = []

        for img in results:
            # Resize the image to max height and width
            resized_img = cv2.resize(
                img, (max_width, max_height), interpolation=cv2.INTER_AREA)
            # print(img.shape, "before tensor_img.shape")
            tensor_img = cv2_img_to_tensor(resized_img)
            # print(tensor_img.shape, "tensor_img.shape")
            batch_results.append(tensor_img.squeeze(0))

        batch_results = torch.stack(batch_results)
        print(batch_results.shape, "batch_results.shape")
        return (batch_results, )


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "tri3d-atr-parse-batch": TRI3DATRParseBatch,
    'tri3d-extract-parts-batch': TRI3DExtractPartsBatch,
    "tri3d-extract-parts-batch2": TRI3DExtractPartsBatch2,
    "tri3d-position-parts-batch": TRI3DPositionPartsBatch,
    "tri3d-swap-pixels": TRI3DSwapPixels,
    "tri3d-skin-feathered-padded-mask": TRI3DSkinFeatheredPaddedMask,
    "tri3d-interaction-canny": TRI3DInteractionCanny

}

VERSION = "1.2.0"
# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "tri3d-atr-parse-batch": "ATR Parse Batch" + " v" + VERSION,
    'tri3d-extract-parts-batch': 'Extract Parts Batch' + " v" + VERSION,
    'tri3d-extract-parts-batch2': 'Extract Parts Batch 2' + " v" + VERSION,
    "tri3d-position-parts-batch": "Position Parts Batch" + " v" + VERSION,
    "tri3d-swap-pixels": "Swap Pixels by Mask" + " v" + VERSION,
    "tri3d-skin-feathered-padded-mask": "Skin Feathered Padded Mask" + " v" + VERSION,
    "tri3d-interaction-canny": "Garment Skin Interaction Canny" + " v" + VERSION
}
