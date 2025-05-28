import os
import cv2
import numpy as np
import torch

class TRI3D_RemoveSmallMaskIslands:
    """
    ComfyUI node that removes small islands of white pixels from a mask image
    based on a specified area threshold.
    """
    
    def from_torch_image(self, image):
        """Convert a torch tensor image to numpy array for OpenCV processing"""
        image = image.cpu().numpy() * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def to_torch_image(self, image):
        """Convert numpy array back to torch tensor format"""
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
                "min_island_area": ("INT", {"default": 100, "min": 1, "max": 10000, "step": 10}),
                "invert": ("BOOLEAN", {"default": False}),
            },
        }

    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", )
    CATEGORY = "TRI3D"

    def run(self, image, min_island_area, invert):
        # Convert Torch image to OpenCV format
        cv_image = self.from_torch_image(image)
        
        # Remove batch dimension if present
        if len(cv_image.shape) == 4:
            cv_image = cv_image[0]
        
        # Make a copy to work with
        result_image = cv_image.copy()
        
        # Process each channel (if grayscale, it will just be one iteration)
        height, width = cv_image.shape[:2]
        
        # If the image has 3 channels (RGB), convert to grayscale for contour detection
        if len(cv_image.shape) == 3 and cv_image.shape[2] == 3:
            # Convert to grayscale for processing
            gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        else:
            # Use the first channel if it's already grayscale or has alpha
            gray = cv_image[:, :, 0]
        
        # Invert if needed (to work with black islands instead of white)
        if invert:
            gray = 255 - gray
            
        # Create binary image
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours in the binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a blank mask for the cleaned image
        clean_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Draw only contours with area greater than the threshold
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_island_area:
                cv2.drawContours(clean_mask, [contour], 0, 255, -1)
        
        # Invert back if needed
        if invert:
            clean_mask = 255 - clean_mask
        
        # Apply the clean mask to each channel of the original image
        if len(cv_image.shape) == 3 and cv_image.shape[2] == 3:
            # RGB image
            for i in range(3):
                result_image[:, :, i] = cv2.bitwise_and(cv_image[:, :, i], clean_mask)
        elif len(cv_image.shape) == 3 and cv_image.shape[2] == 4:
            # RGBA image
            for i in range(4):
                result_image[:, :, i] = cv2.bitwise_and(cv_image[:, :, i], clean_mask)
        else:
            # Single channel image
            result_image = cv2.bitwise_and(cv_image, clean_mask)
            # Reshape to match expected dimensions
            result_image = result_image.reshape(height, width, 1)
        
        # Convert back to torch format
        torch_image = self.to_torch_image(result_image)
        
        # Add batch dimension back
        torch_image = torch_image.unsqueeze(0)
        
        return (torch_image,)

# # Node registration for ComfyUI
# NODE_CLASS_MAPPINGS = {
#     "TRI3D_RemoveSmallMaskIslands": TRI3D_RemoveSmallMaskIslands
# }

# NODE_DISPLAY_NAME_MAPPINGS = {
#     "TRI3D_RemoveSmallMaskIslands": "TRI3D Remove Small Mask Islands"
# }