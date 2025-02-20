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


class TRI3D_Smart_Depth:
    
    
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
        contours = [contour for contour in contours if cv2.contourArea(contour) > 0]

        if len(contours) == 0:
            return image
        # Combine all contours into one
        all_contours = np.vstack(contours)

        # Calculate a single bounding rectangle for all contours
        x, y, w, h = cv2.boundingRect(all_contours)
        # print(x,y,w,h, "x,y,w,h")
        cv2.rectangle(image, (x, y + hip_y), (x + w, y + h + hip_y), (0, 0, 0), -1)

        return image


class TRI3D_NarrowfyImage:
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
                "mask": ("IMAGE", ),
                "aspect_ratio": ("FLOAT", {"default": 0.33, "min": 0.25, "max": 1, "step": 0.01}),
            },
        }
    
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT",)
    RETURN_NAMES = ("cropped_image", "cropped_mask", "cropped_width", "cropped_height",)
    CATEGORY = "TRI3D"

    def run(self, image, mask, aspect_ratio):
        # Convert to CV format and remove batch dimension
        cv_image = self.from_torch_image(image)[0]
        cv_mask = self.from_torch_image(mask)[0]
        
        # Find bounding box of the mask
        mask_channel = cv_mask[:, :, 0]
        contours, _ = cv2.findContours(mask_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image, mask, aspect_ratio
            
        # Filter contours by area
        significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
        
        if not significant_contours:
            return image, mask, aspect_ratio
            
        # Get combined bounding box for all significant contours
        x_min = float('inf')
        y_min = float('inf')
        x_max = 0
        y_max = 0
        
        for contour in significant_contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
        
        # Calculate final width and height with margin
        margin = 15
        x = max(0, x_min - margin)  # Ensure we don't go below 0
        y = max(0, y_min - margin)
        w = min(cv_image.shape[1] - x, (x_max - x_min) + 2 * margin)  # Ensure we don't exceed image width
        h = min(cv_image.shape[0] - y, (y_max - y_min) + 2 * margin)  # Ensure we don't exceed image height
        
        # Crop both image and mask to bounding box
        cropped_image = cv_image[y:y+h, x:x+w]
        cropped_mask = cv_mask[y:y+h, x:x+w]
        
        # Calculate required height for aspect ratio 1/3
        min_height = w * 1/aspect_ratio
        if h < min_height:
            height_extend = min_height - h
            
            # Extend image with black pixels
            extended_image = cv2.copyMakeBorder(
                cropped_image,
                0, int(height_extend),  # top, bottom
                0, 0,                   # left, right
                cv2.BORDER_CONSTANT,
                value=[0, 0, 0]
            )
            
            # Create mask with white pixels only in extended region
            extended_mask = cv2.copyMakeBorder(
                np.zeros_like(cropped_mask),  # Start with black base
                0, int(height_extend),  # top, bottom
                0, 0,                   # left, right
                cv2.BORDER_CONSTANT,
                value=[255, 255, 255]   # White extension
            )
            
            cropped_image = extended_image
            cropped_mask = extended_mask
        
        # Convert back to torch format and add batch dimension
        torch_image = self.to_torch_image(cropped_image).unsqueeze(0)
        torch_mask = self.to_torch_image(cropped_mask).unsqueeze(0)
        
        return (torch_image, torch_mask,w,h)




class TRI3D_CropAndExtend:
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
                "garment_image": ("IMAGE",),
                "garment_mask": ("IMAGE",),
                "human_image": ("IMAGE",),
                "human_mask": ("IMAGE",),
                "margin": ("INT", {"default": 10, "min": 0, "max": 50}),
            },
        }

    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "INT", "INT",)
    RETURN_NAMES = ("cropped_garment", "cropped_garment_mask", "cropped_human", "cropped_human_mask", "cropped_width", "cropped_height",)
    
    def run(self, garment_image, garment_mask, human_image, human_mask, margin):
        # Convert to CV format and remove batch dimension
        cv_garment = self.from_torch_image(garment_image)[0]
        cv_garment_mask = self.from_torch_image(garment_mask)[0]
        cv_human = self.from_torch_image(human_image)[0]
        cv_human_mask = self.from_torch_image(human_mask)[0]
        
        # Process garment
        mask_channel = cv_garment_mask[:, :, 0]
        contours, _ = cv2.findContours(mask_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return garment_image, garment_mask, human_image, human_mask, cv_garment.shape[1], cv_garment.shape[0]
            
        # Get bounding box with margin
        x, y, w, h = cv2.boundingRect(contours[0])
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(cv_garment.shape[1] - x, w + 2 * margin)
        h = min(cv_garment.shape[0] - y, h + 2 * margin)
        
        # Store the cropped dimensions before extension
        cropped_width = w
        cropped_height = h
        
        # Crop garment and its mask
        cropped_garment = cv_garment[y:y+h, x:x+w]
        cropped_garment_mask = cv_garment_mask[y:y+h, x:x+w]
        
        # Calculate required height for aspect ratio 1/3
        min_height = w * 3
        if h < min_height:
            height_extend = min_height - h
            
            # Extend garment image and mask
            extended_garment = cv2.copyMakeBorder(
                cropped_garment,
                0, int(height_extend),
                0, 0,
                cv2.BORDER_CONSTANT,
                value=[0, 0, 0]
            )
            
            extended_garment_mask = cv2.copyMakeBorder(
                cropped_garment_mask,
                0, int(height_extend),
                0, 0,
                cv2.BORDER_CONSTANT,
                value=[255, 255, 255]
            )
            
            cropped_garment = extended_garment
            cropped_garment_mask = extended_garment_mask
        
        # Process human image similarly
        mask_channel = cv_human_mask[:, :, 0]
        contours, _ = cv2.findContours(mask_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(cv_human.shape[1] - x, w + 2 * margin)
            h = min(cv_human.shape[0] - y, h + 2 * margin)
            
            cropped_human = cv_human[y:y+h, x:x+w]
            cropped_human_mask = cv_human_mask[y:y+h, x:x+w]
            
            min_height = w * 3
            if h < min_height:
                height_extend = min_height - h
                
                extended_human = cv2.copyMakeBorder(
                    cropped_human,
                    0, int(height_extend),
                    0, 0,
                    cv2.BORDER_CONSTANT,
                    value=[0, 0, 0]
                )
                
                extended_human_mask = cv2.copyMakeBorder(
                    cropped_human_mask,
                    0, int(height_extend),
                    0, 0,
                    cv2.BORDER_CONSTANT,
                    value=[255, 255, 255]
                )
                
                cropped_human = extended_human
                cropped_human_mask = extended_human_mask
        
        # Convert back to torch format and add batch dimension
        torch_garment = self.to_torch_image(cropped_garment).unsqueeze(0)
        torch_garment_mask = self.to_torch_image(cropped_garment_mask).unsqueeze(0)
        torch_human = self.to_torch_image(cropped_human).unsqueeze(0)
        torch_human_mask = self.to_torch_image(cropped_human_mask).unsqueeze(0)
        
        return (torch_garment, torch_garment_mask, torch_human, torch_human_mask, cropped_width, cropped_height)
