
def tensor_to_cv2_img(tensor):
    # Transfer tensor to CPU and convert to numpy array
    np_img = tensor.cpu().numpy()
    
    # If tensor has 3 dimensions (C, H, W), transpose to (H, W, C)
    if len(np_img.shape) == 3:
        np_img = np.transpose(np_img, (1, 2, 0))
    
    # Scale to [0, 255] and convert type to uint8
    np_img = (np_img * 255).astype(np.uint8)
    
    return np_img

class Example:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "seg": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "test"
    CATEGORY = "Example"
    import cv2
    import numpy as np
    import torch

    def tensor_to_cv2_img(tensor):
        np_img = tensor.cpu().numpy()
        if len(np_img.shape) == 3:
            np_img = np.transpose(np_img, (1, 2, 0))
        np_img = (np_img * 255).astype(np.uint8)
        return np_img

    def cv2_img_to_tensor(cv2_img):
        # Convert image values to [0, 1]
        cv2_img = cv2_img.astype(np.float32) / 255.0
        
        # Convert the numpy array to a torch tensor
        tensor = torch.from_numpy(cv2_img)
        
        # If the tensor has 3 dimensions (H, W, C), change to (C, H, W)
        if len(tensor.shape) == 3:
            tensor = tensor.permute(2, 0, 1)
            
        return tensor

    def test(self, image, seg):
        cv2_image = tensor_to_cv2_img(image)
        cv2_seg = tensor_to_cv2_img(seg)
        
        desired_height, desired_width = cv2_image.shape[:2]
        cv2_seg_resized = cv2.resize(cv2_seg, (desired_width, desired_height))
        added_image = cv2.add(cv2_image, cv2_seg_resized)
        
        # Convert the added image back to a torch tensor
        result_tensor = cv2_img_to_tensor(added_image)
        
        return result_tensor



# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "tri3d-extract-hand": Example
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "tri3d-extract-hand": "Extract hand region using segmentation map"
}
