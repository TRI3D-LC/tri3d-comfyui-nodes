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
    RETURN_TYPES = ("MASK", "BOOL")
    RETURN_NAMES = ("mask", "cleaned")
    CATEGORY = "TRI3D"

    def run(self, masks, threshold):
        batch_results = []
        for mask in masks:
            mask = from_torch_image(mask)
            mask = np.where(mask < 127, 0, 255).astype(np.uint8)
            h,w = mask.shape[:2]

            total_area = h*w
            # num_labels, labels = cv2.connectedComponents(mask)
            region_mask = np.zeros_like(mask)

            # for label in range(1, num_labels):  
            #     area_percent = (np.sum(labels == label)/ total_area) * 100
            #     if area_percent < threshold:
            #         continue
            #     region_mask[labels == label] = 255
            less_than_threshold = True
            area_percent = (np.sum(mask == 255)/ total_area) * 100
            if area_percent > threshold:
                region_mask[mask == 255] = 255
                less_than_threshold = False
            region_mask = to_torch_image(region_mask)
            batch_results.append(region_mask.squeeze(0))

        batch_results = torch.stack(batch_results)
        return (batch_results, less_than_threshold)
        

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
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "coords")
    CATEGORY = "TRI3D"

    def get_frame_coords(self,point1, point2):
        x1, y1 = point1
        x2, y2 = point2

        xmin, xmax, ymin, ymax = min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)

        for i in [xmin, xmax, ymin, ymax]:
            if i < 0: 
                return None

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

        og_h, og_w = image.shape[:2]
        ph, pw = [input_pose['height'], input_pose['width']]

        for i,point in enumerate(keypoints):
            x,y = point
            y = int((y/ph)*og_h)
            x = int((x/pw)*og_w)
            keypoints[i] = [x, y]

        width_offset = int(og_w * (width_pad) / 100)
        height_offset = int(og_h * (height_pad) / 100)

        xmin, xmax, ymin, ymax = [0, og_w, 0, og_h]

        part_to_coords = {
            "shoulders":self.get_frame_coords(keypoints[2], keypoints[5])
        }

        if shoulders:
            print(part_to_coords["shoulders"])
            if part_to_coords["shoulders"] != None:
                new_xmin, new_xmax, new_ymin, new_ymax = part_to_coords["shoulders"]

                xmin, xmax, ymin, ymax = new_xmin, new_xmax, new_ymin, new_ymax
        
        xmin = max(0, xmin - width_offset)
        xmax = min(og_w, xmax + width_offset)
        ymin = max(0, ymin - height_offset)
        ymax = min(og_h, ymax + height_offset)

        image = image[ymin:ymax, xmin:xmax, :].astype(np.uint8)
        image = to_torch_image(image)
        batch_result.append(image)
        batch_result = torch.stack(batch_result)
        print("final_coords", xmin, xmax, ymin, ymax)
        coords = ",".join([str(xmin), str(xmax), str(ymin), str(ymax)])
        
        return batch_result, coords

class TRI3D_position_pose_part():
    """
        put back extracted parts on OG image 
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "og_image": ("IMAGE", ),
                "extracted_image": ("IMAGE", ),
                "coords": ("STRING",{"default" : "xmin, xmax, ymin, ymax"}),
            }
        }


    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    CATEGORY = "TRI3D"

    def run(self, og_image, extracted_image, coords):

        batch_result = []
        og_image = from_torch_image(og_image[0])
        extracted_image = from_torch_image(extracted_image[0])
        
        xmin, xmax, ymin, ymax = [int(i) for i in coords.split(",")]

        og_image[ymin:ymax, xmin:xmax, :] = extracted_image

        og_image = to_torch_image(og_image).unsqueeze(0)
        batch_result.append(og_image)
        batch_result = torch.stack(batch_result)
        return batch_result


class TRI3D_fill_mask():
    """
        fill mask with the neighbouring pixels
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "mask": ("MASK", ),
                "negative_mask": ("MASK", ),
                "offset":("FLOAT",{"default": 1, "min": 0.0, "max": 100.0, "step": 0.01})
            }
        }

    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    CATEGORY = "TRI3D"

    def run(self, image, mask, negative_mask, offset):
        image = from_torch_image(image[0])

        mask = mask[0].cpu().numpy()
        mask = np.expand_dims(mask, -1)
        mh, mw, _ = mask.shape

        inverse_mask = np.ones_like(mask) - mask
        
        negative_mask = negative_mask[0].cpu().numpy()
        indices = np.where(mask > 0)

        offset = offset / 100
        
        source = image.copy()

        for y,x in zip(indices[0],indices[1]):
            x_off = min(mw-1, int(x + offset * mw))
            if negative_mask[y][x_off] == 0:         #check if pixles on right are outside body
                source[y][x] = image[y][x_off]
                
            else:
                x_off = max(0, int(x - offset * mw))       #check if pixles on left are outside body
                if negative_mask[y][x_off] == 0:
                    source[y][x] = image[y][x_off]
                else:
                    y_off = max(0, int(y - offset * mh))
                    if negative_mask[y_off][x] == 0:            #check if pixles on top are outside body
                        source[y][x] = image[y_off][x]

                    else:
                        y_off = min(mh-1, int(y + offset * mh))
                        if negative_mask[y_off][x] == 0:            #check if pixles on bottom are outside body
                            source[y][x] = image[y_off][x]

        image = mask * source + inverse_mask * image
        image = to_torch_image(image).unsqueeze(0)

        return (image,)

class TRI3D_is_only_trouser:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_json_file": ("STRING", {
                    "default": "dwpose/keypoints"
                })
            }
        }

    RETURN_TYPES = ("BOOLEAN", )
    FUNCTION = "main"
    CATEGORY = "TRI3D"

    def main(self, pose_json_file):
        pose = json.load(open(pose_json_file))
        height = pose['height']
        width = pose['width']
        keypoints = pose['keypoints']

        points = [0,14,15,16,17,2,1,5]
        point_to_part = {0:'nose',14:"left eye",15:"right eye",16:"left ear",17:"right ear",2:"left shoulder",1:"neck",5:"right shoulder"}
        all_negative = True          #if all face and shoulder points are negative means it is a bottom shot
        for point in points:
            x,y = keypoints[point]
            if x > 0 and y > 0:
                all_negative = False    
                print(f"{point_to_part[point]} exist")
        return (all_negative,)

class TRI3D_extract_facer_mask:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "background": ("BOOLEAN", {
                    "default": False
                }),
                'hair':("BOOLEAN", {
                    "default": False
                }),
                'lower_lip':("BOOLEAN", {
                    "default": False
                }),
                'inner_mouth':("BOOLEAN", {
                    "default": False
                }),
                'upper_lip':("BOOLEAN", {
                    "default": False
                }),
                'nose':("BOOLEAN", {
                    "default": False
                }),
                'left_eyebrow':("BOOLEAN", {
                    "default": False
                }),
                'right_eyebrow':("BOOLEAN", {
                    "default": False
                }),
                'left_eye':("BOOLEAN", {
                    "default": False
                }),
                'right_eye':("BOOLEAN", {
                    "default": False
                }),
                'face':("BOOLEAN", {
                    "default": False
                })
            }
        }

    RETURN_TYPES = ("MASK", )
    FUNCTION = "main"
    CATEGORY = "TRI3D"

    def main(self, image, background, hair, lower_lip, inner_mouth, upper_lip, nose, left_eyebrow, right_eyebrow, left_eye, right_eye, face):

        image = from_torch_image(image[0])
        h,w,_ = image.shape
        
        mask = np.zeros_like(image)
        
        label_to_rgb = {'background':[0,0,0], 'face':[0,138,255], 'right_eye':[180, 255, 0], 'left_eye':[42, 255, 0], 'right_eyebrow':[0, 255, 96],
                'left_eyebrow':[0,255,234], 'nose':[255, 192, 0],  'upper_lip':[255, 54, 0], 'inner_mouth':[255, 0, 84], 'lower_lip':[255, 0, 222], 
                'hair':[150,0,255]}

        if background:
            temp = np.all(image == label_to_rgb['background'], axis=-1)
            idcs = np.where(temp==True)
            mask[idcs] = 255
        
        if face:
            temp = np.all(image == label_to_rgb['face'], axis=-1)
            idcs = np.where(temp==True)
            mask[idcs] = 255

        if right_eye:
            temp = np.all(image == label_to_rgb['right_eye'], axis=-1)
            idcs = np.where(temp==True)
            mask[idcs] = 255

        if left_eye:
            temp = np.all(image == label_to_rgb['left_eye'], axis=-1)
            idcs = np.where(temp==True)
            mask[idcs] = 255

        if right_eyebrow:
            temp = np.all(image == label_to_rgb['right_eyebrow'], axis=-1)
            idcs = np.where(temp==True)
            mask[idcs] = 255

        if left_eyebrow:
            temp = np.all(image == label_to_rgb['left_eyebrow'], axis=-1)
            idcs = np.where(temp==True)
            mask[idcs] = 255

        if nose:
            temp = np.all(image == label_to_rgb['nose'], axis=-1)
            idcs = np.where(temp==True)
            mask[idcs] = 255

        if upper_lip:
            temp = np.all(image == label_to_rgb['upper_lip'], axis=-1)
            idcs = np.where(temp==True)
            mask[idcs] = 255

        if inner_mouth:
            temp = np.all(image == label_to_rgb['inner_mouth'], axis=-1)
            idcs = np.where(temp==True)
            mask[idcs] = 255

        if lower_lip:
            temp = np.all(image == label_to_rgb['lower_lip'], axis=-1)
            idcs = np.where(temp==True)
            mask[idcs] = 255

        if hair:
            temp = np.all(image == label_to_rgb['hair'], axis=-1)
            idcs = np.where(temp==True)
            mask[idcs] = 255

        mask = to_torch_image(mask[:,:,0]).unsqueeze(0)   
        return (mask,)