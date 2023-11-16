#v0.0.1
class TRI3DATRParseBatch:
    def __init__(self):
        pass    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "main"
    CATEGORY = "TRI3D"
    
    def main(self, images):
        import cv2
        import numpy as np
        import torch
        import os
        import shutil

        def tensor_to_cv2_img(tensor, remove_alpha=False):
            i = 255. * tensor.cpu().numpy()  # This will give us (H, W, C)
            img = np.clip(i, 0, 255).astype(np.uint8)
            return img

        def cv2_img_to_tensor(img):
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img)[None,]
            return img

        ATR_PATH = 'custom_nodes/tri3d-comfyui-nodes/atr_node/'
        ATR_INPUT_PATH = ATR_PATH + 'input/'
        ATR_OUTPUT_PATH = ATR_PATH + 'output/'

        # Create the input directory if it does not exist
        shutil.rmtree(ATR_INPUT_PATH, ignore_errors=True)
        os.makedirs(ATR_INPUT_PATH, exist_ok=True)

        shutil.rmtree(ATR_OUTPUT_PATH, ignore_errors=True)
        os.makedirs(ATR_OUTPUT_PATH, exist_ok=True)

        for i in range(images.shape[0]):
            image = images[i]
            cv2_image = tensor_to_cv2_img(image)    
            cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(ATR_INPUT_PATH + f"image{i}.png", cv2_image)

        # Run the ATR model
        cwd = os.getcwd()
        os.chdir(ATR_PATH)
        os.system("python simple_extractor.py --dataset atr --model-restore 'checkpoints/atr.pth' --input-dir input --output-dir output")
        os.chdir(cwd)

        # Collect and return the results
        batch_results = []
        for i in range(images.shape[0]):
            cv2_segm = cv2.imread(ATR_OUTPUT_PATH + f'image{i}.png')
            cv2_segm = cv2.cvtColor(cv2_segm, cv2.COLOR_BGR2RGB)
            b_tensor_img = cv2_img_to_tensor(cv2_segm)
            batch_results.append(b_tensor_img.squeeze(0))

        batch_results = torch.stack(batch_results)

        return (batch_results,)

class TRI3DExtractPartsBatch:
    def __init__(self):
        pass    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "batch_images": ("IMAGE",),
                "batch_segs" : ("IMAGE",),
                "batch_secondaries" : ("IMAGE",),
                "margin" : ("INT", {"default": 15, "min": 0 }),
                "right_leg": ("BOOLEAN", {"default": False}),
                "right_hand": ("BOOLEAN", {"default": True}),
                "head": ("BOOLEAN", {"default": False}),
                "hair": ("BOOLEAN", {"default": False}),
                "left_shoe" : ("BOOLEAN", {"default": False}),
                "bag" : ("BOOLEAN", {"default": False}),
                "background" : ("BOOLEAN", {"default": False}),
                "dress" : ("BOOLEAN", {"default": False}),
                "left_leg": ("BOOLEAN", {"default": False}),
                "right_shoe" : ("BOOLEAN", {"default": False}),
                "left_hand": ("BOOLEAN", {"default": True}),
                "upper_garment" : ("BOOLEAN", {"default": False}),
                "lower_garment" : ("BOOLEAN", {"default": False}),
                "belt" : ("BOOLEAN", {"default": False}),
                "skirt" : ("BOOLEAN", {"default": False}),
                "hat" : ("BOOLEAN", {"default": False}),
                "sunglasses" : ("BOOLEAN", {"default": False}),
                "scarf" : ("BOOLEAN", {"default": False}),
            },
        }
    RETURN_TYPES = ("IMAGE","IMAGE",)
    FUNCTION = "main"
    CATEGORY = "TRI3D"

    def main(self, batch_images, batch_segs, batch_secondaries, margin, right_leg,right_hand, head, hair, left_shoe,bag,background,dress,left_leg,right_shoe,left_hand, upper_garment,lower_garment,belt,skirt,hat,sunglasses,scarf):
        import cv2
        import numpy as np
        import torch
        from pprint import pprint

        
        def get_segment_counts(segm):
            # Load the segmentation image

            # Reshape the image array to be 2D
            reshaped = segm.reshape(-1, segm.shape[-1])

            # Find unique vectors and their counts
            unique_vectors, counts = np.unique(reshaped, axis=0, return_counts=True)
            segment_counts = list(zip(unique_vectors, counts))
            pprint(segment_counts)
            return segment_counts
        
        def bounded_image(seg_img, color_code_list, input_img):
            import cv2
            import numpy as np
            # Create a mask for hands
            seg_img = cv2.resize(seg_img,(input_img.shape[1],input_img.shape[0]),interpolation=cv2.INTER_NEAREST)
            hand_mask = np.zeros_like(seg_img[:,:,0])
            for color in color_code_list:
                lowerb = np.array(color, dtype=np.uint8)
                upperb = np.array(color, dtype=np.uint8)
                temp_mask = cv2.inRange(seg_img, lowerb, upperb)
                hand_mask = cv2.bitwise_or(hand_mask, temp_mask)

            # Find contours to get the bounding box of the hands
            contours, _ = cv2.findContours(hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # If no contours were found, just return None
            if not contours:
                return None

            # Combine all contours to find encompassing bounding box
            all_points = np.concatenate(contours, axis=0)
            x, y, w, h = cv2.boundingRect(all_points)

            print(x,y,w,h,"x,y,w,h")
            x = max(x - margin, 0)
            y = max(y - margin, 0)
            w = min(w + 2*margin, input_img.shape[1] - x)  # Ensure width does not exceed image boundary
            h = min(h + 2*margin, input_img.shape[0] - y)  # Ensure height does not exceed image boundary
            print(x,y,w,h,"x,y,w,h")
            print(input_img.shape,"input_img.shape")
            # Extract the region from the original image that contains both hands
            hand_region = input_img[y:y+h, x:x+w]

            return hand_region

        def tensor_to_cv2_img(tensor, remove_alpha=False):
            i = 255. * tensor.squeeze(0).cpu().numpy()  # This will give us (H, W, C)
            img = np.clip(i, 0, 255).astype(np.uint8)
            return img

        def cv2_img_to_tensor(img):
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img)[None,]
            return img
        


        batch_results = []
        images = []
        secondaries = []

        for i in range(batch_images.shape[0]):
            image = batch_images[i]
            seg = batch_segs[i]
            
            cv2_image = tensor_to_cv2_img(image)
            cv2_secondary = tensor_to_cv2_img(batch_secondaries[i])
            cv2_seg = tensor_to_cv2_img(seg)

            color_code_list = []
            #################ATR MAPPING#################
            if right_leg:
                color_code_list.append([192,0,128])
            if right_hand:
                color_code_list.append([192,128,128])
            if head:
                color_code_list.append([192,128,0])
            if hair:
                color_code_list.append([0,128,0])
            if left_shoe:
                color_code_list.append([192,0,0])
            if bag:
                color_code_list.append([0,64,0])
            if background:
                color_code_list.append([0,0,0])
            if dress:
                color_code_list.append([128,128,128])
            if left_leg:
                color_code_list.append([64,0,128])
            if right_shoe:
                color_code_list.append([64,128,0])
            if left_hand:
                color_code_list.append([64,128,128])
            if upper_garment:
                color_code_list.append([0,0,128])
            if lower_garment:
                color_code_list.append([0,128,128])
            if belt:
                color_code_list.append([64,0,0])
            if skirt:
                color_code_list.append([128,0,128])
            if hat:
                color_code_list.append([128,0,0])
            if sunglasses:
                color_code_list.append([128,128,0])
            if scarf:
                color_code_list.append([128,64,0])


            bimage = bounded_image(cv2_seg, color_code_list, cv2_image)
            bsecondary = bounded_image(cv2_seg, color_code_list, cv2_secondary)
            
            # Handle case when bimage is None to avoid error during conversion to tensor
            if bimage is not None:
                images.append(bimage)
            else:
                black_img = np.zeros_like(cv2_image)
                images.append(black_img)

            if bsecondary is not None:
                secondaries.append(bsecondary)
            else:
                black_img = np.zeros_like(cv2_secondary)
                secondaries.append(black_img)

        # Get max height and width
        max_height = max(img.shape[0] for img in images)
        max_width = max(img.shape[1] for img in images)

        batch_results = []
        batch_secondaries = []

        for img in images:
            # Resize the image to max height and width
            resized_img = cv2.resize(img, (max_width, max_height), interpolation=cv2.INTER_AREA)
            tensor_img = cv2_img_to_tensor(resized_img)
            batch_results.append(tensor_img.squeeze(0))
        
        for sec in secondaries:
            # Resize the image to max height and width
            resized_sec = cv2.resize(sec, (max_width, max_height), interpolation=cv2.INTER_AREA)
            tensor_sec = cv2_img_to_tensor(resized_sec)
            batch_secondaries.append(tensor_sec.squeeze(0))

        batch_results = torch.stack(batch_results)
        batch_secondaries = torch.stack(batch_secondaries)
        print(batch_results.shape,"batch_results.shape")
        return (batch_results,batch_secondaries)

class TRI3DPositionPartsBatch:
    def __init__(self):
        pass    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "batch_images": ("IMAGE",),
                "batch_segs" : ("IMAGE",),
                "batch_handimgs" : ("IMAGE",),
                "margin" : ("INT", {"default": 15, "min": 0 }),
                 "right_leg": ("BOOLEAN", {"default": False}),
                "right_hand": ("BOOLEAN", {"default": True}),
                "head": ("BOOLEAN", {"default": False}),
                "hair": ("BOOLEAN", {"default": False}),
                "left_shoe" : ("BOOLEAN", {"default": False}),
                "bag" : ("BOOLEAN", {"default": False}),
                "background" : ("BOOLEAN", {"default": False}),
                "dress" : ("BOOLEAN", {"default": False}),
                "left_leg": ("BOOLEAN", {"default": False}),
                "right_shoe" : ("BOOLEAN", {"default": False}),
                "left_hand": ("BOOLEAN", {"default": True}),
                "upper_garment" : ("BOOLEAN", {"default": False}),
                "lower_garment" : ("BOOLEAN", {"default": False}),
                "belt" : ("BOOLEAN", {"default": False}),
                "skirt" : ("BOOLEAN", {"default": False}),
                "hat" : ("BOOLEAN", {"default": False}),
                "sunglasses" : ("BOOLEAN", {"default": False}),
                "scarf" : ("BOOLEAN", {"default": False}),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "main"
    CATEGORY = "TRI3D"

    def main(self, batch_images, batch_segs, batch_handimgs, margin,right_leg,right_hand, head, hair, left_shoe,bag,background,dress,left_leg,right_shoe,left_hand, upper_garment,lower_garment,belt,skirt,hat,sunglasses,scarf):
        import cv2
        import numpy as np
        import torch
        from pprint import pprint

        
        def bounded_image_points(seg_img, color_code_list, input_img):
            import cv2
            import numpy as np
            # Create a mask for hands
            seg_img = cv2.resize(seg_img,(input_img.shape[1],input_img.shape[0]),interpolation=cv2.INTER_NEAREST)
            hand_mask = np.zeros_like(seg_img[:,:,0])
            for color in color_code_list:
                lowerb = np.array(color, dtype=np.uint8)
                upperb = np.array(color, dtype=np.uint8)
                temp_mask = cv2.inRange(seg_img, lowerb, upperb)
                hand_mask = cv2.bitwise_or(hand_mask, temp_mask)

            # Find contours to get the bounding box of the hands
            contours, _ = cv2.findContours(hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # If no contours were found, just return None
            if not contours:
                return None

            # Combine all contours to find encompassing bounding box
            all_points = np.concatenate(contours, axis=0)
            x, y, w, h = cv2.boundingRect(all_points)
            x = max(x - margin, 0)
            y = max(y - margin, 0)
            w = min(w + 2*margin, input_img.shape[1] - x)  # Ensure width does not exceed image boundary
            h = min(h + 2*margin, input_img.shape[0] - y)  # Ensure height does not exceed image boundary

            return (x,y,w,h)

        def tensor_to_cv2_img(tensor, remove_alpha=False):
            i = 255. * tensor.squeeze(0).cpu().numpy()  # This will give us (H, W, C)
            img = np.clip(i, 0, 255).astype(np.uint8)
            return img

        def cv2_img_to_tensor(img):
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img)[None,]
            return img


        batch_results = []

        for i in range(batch_images.shape[0]):
            image = batch_images[i]
            seg = batch_segs[i]
            handimg = batch_handimgs[i]
            
            cv2_image = tensor_to_cv2_img(image)    
            cv2_seg = tensor_to_cv2_img(seg)

            color_code_list = []
            #################ATR MAPPING#################
            if right_leg:
                color_code_list.append([192,0,128])
            if right_hand:
                color_code_list.append([192,128,128])
            if head:
                color_code_list.append([192,128,0])
            if hair:
                color_code_list.append([0,128,0])
            if left_shoe:
                color_code_list.append([192,0,0])
            if bag:
                color_code_list.append([0,64,0])
            if background:
                color_code_list.append([0,0,0])
            if dress:
                color_code_list.append([128,128,128])
            if left_leg:
                color_code_list.append([64,0,128])
            if right_shoe:
                color_code_list.append([64,128,0])
            if left_hand:
                color_code_list.append([64,128,128])
            if upper_garment:
                color_code_list.append([0,0,128])
            if lower_garment:
                color_code_list.append([0,128,128])
            if belt:
                color_code_list.append([64,0,0])
            if skirt:
                color_code_list.append([128,0,128])
            if hat:
                color_code_list.append([128,0,0])
            if sunglasses:
                color_code_list.append([128,128,0])
            if scarf:
                color_code_list.append([128,64,0])


            positions = bounded_image_points(cv2_seg, color_code_list, cv2_image)

            try:
                cv2_handimg = tensor_to_cv2_img(handimg)
                cv2_handimg = cv2.resize(cv2_handimg, (positions[2], positions[3]), interpolation=cv2.INTER_NEAREST)

                cv2_image[positions[1]:positions[1]+positions[3], positions[0]:positions[0]+positions[2]] = cv2_handimg
            except Exception as e:
                print(e)
                pass
            b_tensor_img = cv2_img_to_tensor(cv2_image)
            batch_results.append(b_tensor_img.squeeze(0))

        batch_results = torch.stack(batch_results)
        
        return (batch_results,)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "tri3d-atr-parse-batch": TRI3DATRParseBatch,
    'tri3d-extract-parts-batch': TRI3DExtractPartsBatch,
    "tri3d-position-parts-batch": TRI3DPositionPartsBatch,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "tri3d-atr-parse-batch" : "ATR Parse Batch",
    'tri3d-extract-parts-batch': 'Extract Parts Batch',
    "tri3d-position-parts-batch" : "Position Parts Batch",
}
