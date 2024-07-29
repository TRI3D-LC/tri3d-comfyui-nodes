import torch, cv2
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
        return batch_results
        