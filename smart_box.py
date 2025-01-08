import numpy as np
import torch
import json

import cv2

class TRI3D_SmartBox:
    
    
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
                "keypoints_json": ("STRING", {"multiline": True}),
            },
        }

    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", )
    CATEGORY = "TRI3D"

    def run(self, image, keypoints_json):
        from .dwpose import comfy_utils
        kp_data = json.loads(open(keypoints_json, 'r').read())
        original_height, original_width = kp_data['height'], kp_data['width']
        torso_keypoints = comfy_utils.extract_torso_keypoints(kp_data['keypoints'])

        # Convert Torch image to OpenCV format
        cv_image = self.from_torch_image(image)
        
        # Remove the batch dimension if present
        if len(cv_image.shape) == 4:
            cv_image = cv_image[0]

        # Adjust keypoints to match the image dimensions
        adjusted_keypoints = self.adjust_keypoints(torso_keypoints, cv_image.shape, original_height, original_width)
        

        # Fill the area below the hip line
        filled_image = self.fill_below_hip(cv_image, adjusted_keypoints)
        

        # Convert back to Torch format
        torch_image = self.to_torch_image(filled_image)

        # Add the batch dimension back
        torch_image = torch_image.unsqueeze(0)
        

        return (torch_image,)

    def adjust_keypoints(self, keypoints, image_shape, original_height, original_width):
        image_height, image_width = image_shape[:2]
        scale_x = image_width / original_width
        scale_y = image_height / original_height

        adjusted_keypoints = [
            (int(x * scale_x), int(y * scale_y)) for x, y in keypoints
        ]
        return adjusted_keypoints

    def fill_below_hip(self, image, keypoints):
        # Correct the indices for hip keypoints
        # Assuming indices 8 and 11 are for left and right hips
        hip_y = (keypoints[0][1] + keypoints[1][1]) // 2

        # Find the bounding box of the mask below the hip line
        mask = image[:, :, 0]  # Assuming single-channel mask
        below_hip = mask[hip_y:, :]
        contours, _ = cv2.findContours(below_hip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y + hip_y), (x + w, y + h + hip_y), (255, 255, 255), -1)

        return image

    
