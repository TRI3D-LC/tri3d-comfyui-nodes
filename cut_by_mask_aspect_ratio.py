import os
import cv2
import numpy as np
import torch

class TRI3D_CutByMaskAspectRatio:
    """
    ComfyUI node that crops an image based on a mask's bounding box,
    adjusts the aspect ratio, and resizes to specified dimensions.
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
                "image": ("IMAGE",),
                "mask": ("IMAGE",),
                "margin": ("INT", {"default": 10, "min": 0, "max": 100, "step": 1}),
                "target_width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "target_height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "padding_color": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
            },
        }

    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "TRI3D"

    def run(self, image, mask, margin, target_width, target_height, padding_color=255):
        # Convert Torch images to OpenCV format
        cv_image = self.from_torch_image(image)
        cv_mask = self.from_torch_image(mask)
        
        # Remove batch dimension if present
        if len(cv_image.shape) == 4:
            cv_image = cv_image[0]
        if len(cv_mask.shape) == 4:
            cv_mask = cv_mask[0]
        
        # Convert mask to grayscale if it's not already
        if len(cv_mask.shape) == 3 and cv_mask.shape[2] > 1:
            mask_gray = cv2.cvtColor(cv_mask, cv2.COLOR_RGB2GRAY)
        else:
            mask_gray = cv_mask[:, :, 0]
        
        # Create binary mask
        _, binary_mask = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours in the binary mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # If no contours found, return the original image
            print("No contours found in mask. Returning original image.")
            return (image,)
        
        # Find bounding box around all contours
        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = 0, 0
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
        
        # Add margin to bounding box
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(cv_image.shape[1], x_max + margin)
        y_max = min(cv_image.shape[0], y_max + margin)
        
        # Current dimensions of the bounding box
        height = y_max - y_min
        width = x_max - x_min
        
        # Calculate the target aspect ratio (width/height)
        target_aspect_ratio = target_width / target_height
        
        # Calculate current aspect ratio
        current_aspect_ratio = width / height
        
        # Crop the image to the original bounding box
        cropped_image = cv_image[y_min:y_max, x_min:x_max]
        
        # Adjust width to match the target aspect ratio while keeping height constant
        if current_aspect_ratio < target_aspect_ratio:
            # Current width is too narrow, add white padding
            # Calculate the intermediate dimensions for resizing
            intermediate_height = height
            intermediate_width = width  # Keep original width
            
            # Calculate the final canvas size with padding
            canvas_height = intermediate_height
            canvas_width = int(canvas_height * target_aspect_ratio)
            
            # Calculate padding on each side
            padding_x = (canvas_width - intermediate_width) // 2
            
            # Create canvas with padding color
            num_channels = cropped_image.shape[2] if len(cropped_image.shape) == 3 else 1
            if num_channels == 1:
                canvas = np.full((canvas_height, canvas_width), padding_color, dtype=np.uint8)
            else:
                canvas = np.full((canvas_height, canvas_width, num_channels), padding_color, dtype=np.uint8)
            
            # Place the cropped image on the canvas
            if num_channels == 1:
                canvas[:, padding_x:padding_x+intermediate_width] = cropped_image
            else:
                canvas[:, padding_x:padding_x+intermediate_width, :] = cropped_image
            
            # Set the final image to be the padded canvas
            cropped_image = canvas
            
        elif current_aspect_ratio > target_aspect_ratio:
            # Current width is too wide, crop it
            new_width = int(height * target_aspect_ratio)
            width_difference = width - new_width
            
            # Crop equally from both sides if possible
            left_crop = width_difference // 2
            right_crop = width_difference - left_crop
            
            # Apply the crop to the cropped image
            cropped_image = cropped_image[:, left_crop:width-right_crop]
        
        # Resize the cropped/padded image to the target dimensions using Lanczos interpolation
        resized_image = cv2.resize(cropped_image, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Convert back to torch format
        torch_image = self.to_torch_image(resized_image)
        
        # Add batch dimension back
        torch_image = torch_image.unsqueeze(0)
        
        return (torch_image,)

# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "TRI3D_CutByMaskAspectRatio": TRI3D_CutByMaskAspectRatio
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TRI3D_CutByMaskAspectRatio": "TRI3D Cut By Mask Aspect Ratio"
}
