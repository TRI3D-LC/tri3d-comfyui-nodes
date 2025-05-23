import os
import cv2
import numpy as np
import torch

class TRI3D_MaskAreaPercentage:
    """
    ComfyUI node that calculates the percentage of white pixels in an image
    relative to the total image area.
    """
    
    def from_torch_image(self, image):
        """Convert a torch tensor image to numpy array for OpenCV processing"""
        image = image.cpu().numpy() * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("INT", {"default": 127, "min": 0, "max": 255, "step": 1}),
            },
        }

    FUNCTION = "run"
    RETURN_TYPES = ("FLOAT", "INT", "INT",)
    RETURN_NAMES = ("percentage", "white_pixels", "total_pixels",)
    CATEGORY = "TRI3D"

    def run(self, image, threshold=127):
        # Convert Torch image to OpenCV format
        cv_image = self.from_torch_image(image)
        
        # Remove batch dimension if present
        if len(cv_image.shape) == 4:
            cv_image = cv_image[0]
        
        # Convert to grayscale if it's a color image
        if len(cv_image.shape) == 3 and cv_image.shape[2] > 1:
            gray_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = cv_image[:, :, 0]
        
        # Calculate total number of pixels
        total_pixels = gray_image.shape[0] * gray_image.shape[1]
        
        # Count white pixels (pixels with values above threshold)
        _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
        white_pixels = cv2.countNonZero(binary_image)
        
        # Calculate percentage of white pixels
        percentage = (white_pixels / total_pixels) * 100.0
        
        return (percentage, white_pixels, total_pixels,)

# # Node registration for ComfyUI
# NODE_CLASS_MAPPINGS = {
#     "TRI3D_MaskAreaPercentage": TRI3D_MaskAreaPercentage
# }

# NODE_DISPLAY_NAME_MAPPINGS = {
#     "TRI3D_MaskAreaPercentage": "TRI3D Mask Area Percentage"
# } 