import numpy as np
import torch
import json

import cv2

#  {0,  "Nose"},
# //     {1,  "Neck"},
# //     {2,  "RShoulder"},
# //     {3,  "RElbow"},
# //     {4,  "RWrist"},
# //     {5,  "LShoulder"},
# //     {6,  "LElbow"},
# //     {7,  "LWrist"},
# //     {8,  "MidHip"},
# //     {9,  "RHip"},
# //     {10, "RKnee"},
# //     {11, "RAnkle"},
# //     {12, "LHip"},
# //     {13, "LKnee"},
# //     {14, "LAnkle"},
# //     {15, "REye"},
# //     {16, "LEye"},
# //     {17, "REar"},
# //     {18, "LEar"},
# //     {19, "LBigToe"},
# //     {20, "LSmallToe"},
# //     {21, "LHeel"},
# //     {22, "RBigToe"},
# //     {23, "RSmallToe"},
# //     {24, "RHeel"},
# //     {25, "Background"}


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

    def extract_torso_keypoints(self, keypoints):
        # Indices for torso-related keypoints
        torso_indices = [8, 9, 10, 11, 12, 13]
        return [keypoints[i] for i in torso_indices]


    def run(self, image, keypoints_json):
        
        kp_data = json.loads(open(keypoints_json, 'r').read())
        original_height, original_width = kp_data['height'], kp_data['width']
        torso_keypoints = self.extract_torso_keypoints(kp_data['keypoints'])

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
        # print(keypoints,"hip keypoints")
        try:
            valid_y_coords = [kp[1] for kp in keypoints if kp[1] >= 0]
            hip_y = min(valid_y_coords) if valid_y_coords else 0
        except:
            hip_y = 0
        
        if hip_y == 0:
            return image

        # Find the bounding box of the mask below the hip line
        mask = image[:, :, 0]  # Assuming single-channel mask
        below_hip = mask[hip_y:, :]
        contours, _ = cv2.findContours(below_hip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = 0
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # print(cnt, x,y,w,h, cv2.contourArea(contour), "cnt,x,y,w,h,area")
            cnt+=1
        #     cv2.rectangle(image, (x, y + hip_y), (x + w, y + h + hip_y), (255, 255, 255), -1)
        contours = [contour for contour in contours if cv2.contourArea(contour) > 20]

        if len(contours) == 0:
            return image
        # Combine all contours into one
        all_contours = np.vstack(contours)

        # Calculate a single bounding rectangle for all contours
        x, y, w, h = cv2.boundingRect(all_contours)
        # print(x,y,w,h, "x,y,w,h")
        cv2.rectangle(image, (x, y + hip_y), (x + w, y + h + hip_y), (255, 255, 255), -1)

        return image

    
class TRI3D_Skip_HeadMask:
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
                "head_mask": ("IMAGE", ),
            },
        }

    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", )
    CATEGORY = "TRI3D"

    def run(self, image, head_mask):
        # Convert Torch images to OpenCV format
        cv_image = self.from_torch_image(image)
        cv_head_mask = self.from_torch_image(head_mask)

        # Remove the batch dimension if present
        if len(cv_image.shape) == 4:
            cv_image = cv_image[0]
        if len(cv_head_mask.shape) == 4:
            cv_head_mask = cv_head_mask[0]

        # Find the lowest point in the head mask
        mask = cv_head_mask[:, :, 0]  # Assuming single-channel mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lowest_y = 0
        for contour in contours:
            for point in contour:
                x, y = point[0]
                if y > lowest_y:
                    lowest_y = y

        # Black out everything above the lowest point
        cv_image[:lowest_y, :] = 0

        # Convert back to Torch format
        torch_image = self.to_torch_image(cv_image)

        # Add the batch dimension back
        torch_image = torch_image.unsqueeze(0)

        return (torch_image,)


    
class TRI3D_Skip_HeadMask_AddNeck:

    def adjust_keypoints(self, keypoints, image_shape, original_height, original_width):
        image_height, image_width = image_shape[:2]
        scale_x = image_width / original_width
        scale_y = image_height / original_height

        adjusted_keypoints = [
            (int(x * scale_x), int(y * scale_y)) for x, y in keypoints
        ]
        return adjusted_keypoints

    def from_torch_image(self, image):
        image = image.cpu().numpy() * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def to_torch_image(self, image):
        image = image.astype(dtype=np.float32)
        image /= 255.0
        image = torch.from_numpy(image)
        return image


    def extract_neck_keypoint(self, keypoints):
        # Indices for torso-related keypoints
        neck_indices = [1]
        return [keypoints[i] for i in neck_indices]


    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "head_mask": ("IMAGE", ),
                "keypoints_json": ("STRING", {"multiline": True}),
                "ratio_aggression": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
            },
        }

    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", )
    CATEGORY = "TRI3D"

    def run(self, image, head_mask, keypoints_json,ratio_aggression):
        # Convert Torch images to OpenCV format
        cv_image = self.from_torch_image(image)
        cv_head_mask = self.from_torch_image(head_mask)

        # Remove the batch dimension if present
        if len(cv_image.shape) == 4:
            cv_image = cv_image[0]
        if len(cv_head_mask.shape) == 4:
            cv_head_mask = cv_head_mask[0]

        kp_data = json.loads(open(keypoints_json, 'r').read())
        original_height, original_width = kp_data['height'], kp_data['width']
        neck_keypoints = self.extract_neck_keypoint(kp_data['keypoints'])
        from pprint import pprint
        pprint(neck_keypoints)
        adjusted_neck_keypoints = self.adjust_keypoints(neck_keypoints, cv_image.shape, original_height, original_width)
        # Find the lowest point in the head mask
        mask = cv_head_mask[:, :, 0]  # Assuming single-channel mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lowest_y = 0
        for contour in contours:
            for point in contour:
                x, y = point[0]
                if y > lowest_y:
                    lowest_y = y
        
        #take average of neck keypoint y and lowest_y in head mask 
        neck_y = adjusted_neck_keypoints[0][1]
        if(neck_y <= 0):
            neck_y = lowest_y
        average_y = int((neck_y*ratio_aggression + lowest_y*(1-ratio_aggression)))
        
        print(neck_y, lowest_y, "neck_y, lowest_y")
        print(average_y, "average_y")

        # Black out everything above the lowest point
        cv_image[:average_y, :] = 0

        # Convert back to Torch format
        torch_image = self.to_torch_image(cv_image)

        # Add the batch dimension back
        torch_image = torch_image.unsqueeze(0)

        return (torch_image,)


class TRI3D_Image_extend:
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
                "face_mask": ("IMAGE", ),
                "image": ("IMAGE", ),
                "ratio": ("FLOAT", {"default": 1.5, "min": 1.2, "max": 2, "step": 0.01}),
            },
        }
    
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "IMAGE", )
    RETURN_NAMES = ("image", "mask_image", )
    CATEGORY = "TRI3D"

    def run(self, face_mask, image, ratio):
        cv_face_mask = self.from_torch_image(face_mask)
        cv_image = self.from_torch_image(image)

        # Remove the batch dimension if present
        if len(cv_image.shape) == 4:
            cv_image = cv_image[0]
        if len(cv_face_mask.shape) == 4:
            cv_face_mask = cv_face_mask[0]
        mask = cv_face_mask[:, :, 0]  # Assuming single-channel mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lowest_y = 0
        highest_y = cv_image.shape[0]
        for contour in contours:
            for point in contour:
                x, y = point[0]
                if y > lowest_y:
                    lowest_y = y
                if y < highest_y:
                    highest_y = y

        y_below_face = cv_image.shape[0] - lowest_y
        y_face = lowest_y-highest_y

        # Only extend if the space below face is less than 1.5 times face height
        target_below_face = int(y_face * ratio)
        # print("y_face", y_face)
        # print("lowest_y", lowest_y)
        # print("highest_y", highest_y)
        # print("target_below_face", target_below_face)
        # print("y_below_face", y_below_face)

        original_height = cv_image.shape[0]
        original_width = cv_image.shape[1]
        
        if y_below_face < target_below_face:
            y_extend = target_below_face - y_below_face
            
            # Calculate how much to extend horizontally to maintain aspect ratio
            new_height = original_height + y_extend
            new_width = int(original_width * (new_height / original_height))
            x_extend = new_width - original_width
            x_extend_left = x_extend // 2
            x_extend_right = x_extend - x_extend_left
            
            # Extend the image in all necessary directions
            cv_image = cv2.copyMakeBorder(
                cv_image, 
                0, y_extend,                    # top, bottom
                x_extend_left, x_extend_right,  # left, right
                cv2.BORDER_CONSTANT, 
                value=[0, 0, 0]
            )
            
            # Create extension mask
            extension_mask = np.zeros_like(cv_image)
            # Make extended portions white
            extension_mask[original_height:, :] = 255  # bottom extension
            extension_mask[:, :x_extend_left] = 255    # left extension
            extension_mask[:, -x_extend_right:] = 255  # right extension

        else:
            extension_mask = np.zeros_like(cv_image)

        # Convert both images back to torch format
        torch_image = self.to_torch_image(cv_image)
        torch_mask = self.to_torch_image(extension_mask)
        
        # Add batch dimension to both
        torch_image = torch_image.unsqueeze(0)
        torch_mask = torch_mask.unsqueeze(0)
        
        return (torch_image, torch_mask)
