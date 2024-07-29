import torch, cv2, json
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

class TRI3D_clean_mask():

    """For the given mask and threshold area, remove all patches in the mask with area smaller than threshold"""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "masks": ("MASK", ),
                "threshold":("FLOAT",{"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01})
            }
        }

    FUNCTION = "run"
    RETURN_TYPES = ("MASK", )
    CATEGORY = "TRI3D"

    def run(self, masks, threshold):
        batch_results = []
        for mask in masks:
            mask = from_torch_image(mask)
            mask = np.where(mask < 127, 0, 255).astype(np.uint8)
            h,w = mask.shape[:2]

            total_area = h*w
            num_labels, labels = cv2.connectedComponents(mask)
            region_mask = np.zeros_like(mask)

            for label in range(1, num_labels):  
                area_percent = (np.sum(labels == label)/ total_area) * 100
                if area_percent < threshold:
                    continue
                region_mask[labels == label] = 255

            region_mask = to_torch_image(region_mask)
            batch_results.append(region_mask.squeeze(0))

        batch_results = torch.stack(batch_results)
        print(batch_results.shape)
        return batch_results
        

class TRI3D_extract_pose_part():
    """
        For the given pose, extract region around body parts, region can be defined by % of image size  
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "pose_json": ("STRING",{"default" : "dwpose/keypoints/input.json"}),
                "width_pad": ("FLOAT",{"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "height_pad": ("FLOAT",{"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "shoulders":("BOOLEAN", {
                    "default": False
                })
            }
        }


    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", )
    CATEGORY = "TRI3D"

    def get_frame_coords(self,point1, point2):
        x1, y1 = point1
        x2, y2 = point2

        xmin, xmax, ymin, ymax = min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)

        return [xmin, xmax, ymin, ymax]

    def run(self, image, pose_json, width_pad, height_pad, shoulders):
        """
        image : input image
        width_pad: % of image width you want to apply on both size of pose body part
        height_pad: % of image width you want to apply on both size of pose body part
        rest of them are body parts
        """

        image = from_torch_image(image[0])
        batch_result = []
        input_pose = json.load(open(pose_json))
        keypoints = input_pose['keypoints']

        print("Input image shape:", image.shape)

        h,w = image.shape[:2]

        width_offset = int(w * (width_pad) / 100)
        height_offset = int(h * (height_pad) / 100)

        xmin, xmax, ymin, ymax = [0, image.shape[1], 0, image.shape[0]]

        part_to_coords = {
            "shoulders":self.get_frame_coords(keypoints[2], keypoints[5])
        }
        
        if shoulders:
            new_xmin, new_xmax, new_ymin, new_ymax = part_to_coords["shoulders"]

            xmin, xmax, ymin, ymax = min(xmin, new_xmin), max(xmax, new_xmax), min(ymin, new_ymin), max(ymax, new_ymax)
        
        print("extracted coords", xmin, xmax, ymin, ymax)
        xmin = max(0, xmin - width_offset)
        xmax = min(w, xmax + width_offset)
        ymin = max(0, ymin - height_offset)
        ymax = min(h, ymax + height_offset)

        image = to_torch_image(image[ymin:ymax, xmin:xmax,:]).unsqueeze(0)
        batch_result.append(image)
        batch_result = torch.stack(batch_result)
        # print("final_coords", xmin, xmax, ymin, ymax)
        return batch_result

