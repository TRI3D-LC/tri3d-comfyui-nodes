class TRI3DExtractHand:
    def __init__(self):
        pass    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "seg" : ("IMAGE",),
                "margin" : ("INT", {"default": 15, "min": 0 }),
                "left_hand" : ("BOOLEAN", {"default": True}),
                "right_hand" : ("BOOLEAN", {"default": True}),
                "head" : ("BOOLEAN", {"default": False}),
                "hair" : ("BOOLEAN", {"default": False}),
                "left_leg" : ("BOOLEAN", {"default": False}),
                "right_leg" : ("BOOLEAN", {"default": False}),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "main"
    CATEGORY = "TRI3D"
    def main(self, image,seg,margin,left_hand,right_hand,head,hair,left_leg,right_leg):
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

        cv2_image = tensor_to_cv2_img(image)    
        cv2_seg = tensor_to_cv2_img(seg)
        # cv2_seg = cv2.resize(cv2_seg,(cv2_image.shape[1],cv2_image.shape[0]),interpolation=cv2.INTER_NEAREST)

        # 128 128 64 / 128 128 192
        # color_code_list = [[128,128,64], [128,128,192]]
        # (array([  0, 128,   0], dtype=uint8), 9393), #hair
            # (array([  0, 128, 128], dtype=uint8), 50277), #lower garment
            # (array([  0,   0, 128], dtype=uint8), 291418), #upper garment

            # (array([ 64, 128, 128], dtype=uint8), 14548), #left hand
            # (array([192, 128,   0], dtype=uint8), 33325), #face
            # (array([192, 128, 128], dtype=uint8), 14855)] #right hand
            # [(array([0, 0, 0], dtype=uint8), 638434), #background
            # (array([  0,   0, 128], dtype=uint8), 77453),
            # (array([ 64, 128,   0], dtype=uint8), 5640),
            # (array([192,   0,   0], dtype=uint8), 5409),
        get_segment_counts(cv2_seg)
        color_code_list = []
        if left_hand:
            color_code_list.append([64,128,128])
        if right_hand:
            color_code_list.append([192,128,128])
        if head:
            color_code_list.append([192,128,0])
        if hair:
            color_code_list.append([0,128,0])
        if left_leg:
            color_code_list.append([192,0,0])
        if right_leg:
            color_code_list.append([64,128,0])

        # color_code_list = [[64,128,128], [192,128,128]]
        bimage = bounded_image(cv2_seg,color_code_list,cv2_image)
        b_tensor_img = cv2_img_to_tensor(bimage)
        
        return (b_tensor_img,)

class TRI3DFuzzification:
    def __init__(self):
        pass    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("IMAGE",),
                "inputseg" : ("IMAGE",),
                "controlnetoutput": ("IMAGE",),
                "controlnetoutputseg" : ("IMAGE",),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "main"
    CATEGORY = "TRI3D"
    def main(self, input,inputseg,controlnetoutput,controlnetoutputseg):
        import cv2
        import numpy as np
        import torch
        from pprint import pprint
        from scipy.spatial import distance


        def combined_image(img1, img2, mask):
            if img1.shape[2] == 4:
            # Convert it from four channels to three channels
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGRA2BGR)
            mask_inv = cv2.bitwise_not(mask)

            # Normalize the masks to the range [0, 1]
            mask = cv2.normalize(mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            mask_inv = cv2.normalize(mask_inv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            # Convert images and masks to the same data type
            img1 = img1.astype(np.float32)
            img2 = img2.astype(np.float32)
            mask = mask.astype(np.float32)
            mask_inv = mask_inv.astype(np.float32)

            img1 = cv2.resize(img1, (mask.shape[1], mask.shape[0]),interpolation=cv2.INTER_NEAREST)
            img2 = cv2.resize(img2, (mask.shape[1], mask.shape[0]),interpolation=cv2.INTER_NEAREST)

            # Check if img1 (and hence img2) have more than one channel (e.g., RGB images)
            if len(img1.shape) > 2:
                # Convert mask and mask_inv to the same number of channels as img1
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                mask_inv = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)

            # Use the masks to get the weighted regions of each image
            img1_masked = cv2.multiply(img1, mask_inv)
            img2_masked = cv2.multiply(img2, mask)

            # Combine the two images
            combined = cv2.add(img1_masked, img2_masked).astype(np.uint8)
            return combined


        def fuzzify(img):
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Threshold the image
            _, thresholded = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)

            # Find contours in the thresholded image
            contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # # Approximate contours to reduce number of points
            # epsilon_factor = 0.02  # can be adjusted, higher values mean more simplification
            # simplified_contours = [cv2.approxPolyDP(cnt, epsilon_factor * cv2.arcLength(cnt, True), True) for cnt in contours]

            # Define brush size
            brush_radius = int(15*img.shape[0]/1024.0)
            mask = np.zeros_like(img)

            for contour in contours:
                # print("contour shape - ",contour.shape)
                contour_points = contour.squeeze(1)
                
                # Determine the bounding box around the contour and expand it by the brush radius
                x_min, y_min = np.min(contour_points, axis=0) - brush_radius
                x_max, y_max = np.max(contour_points, axis=0) + brush_radius

                # Clip the coordinates to the image boundaries
                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(img.shape[1]-1, x_max), min(img.shape[0]-1, y_max)

                # # Create a grid of coordinates within this bounding box
                # ys, xs = np.ogrid[y_min:y_max+1, x_min:x_max+1]
                # grid_coords = np.column_stack((xs.ravel(), ys.ravel()))

                ys, xs = np.mgrid[y_min:y_max+1, x_min:x_max+1]
                grid_coords = np.column_stack((xs.flatten(), ys.flatten()))

                # Compute distances for pixels inside the bounding box
                distances = distance.cdist(grid_coords, contour_points, 'euclidean')
                min_distances = distances.min(axis=1)
                brush_effect = np.clip((1 - min_distances / brush_radius) * 255, 0, 255)

                mask[grid_coords[:, 1], grid_coords[:, 0]] = np.maximum(mask[grid_coords[:, 1], grid_coords[:, 0]], brush_effect)

            # Save the result
            final_mask = np.maximum(img, mask.astype(img.dtype))
            return final_mask

        def get_segment_counts(segm):
            # Load the segmentation image

            # Reshape the image array to be 2D
            reshaped = segm.reshape(-1, segm.shape[-1])

            # Find unique vectors and their counts
            unique_vectors, counts = np.unique(reshaped, axis=0, return_counts=True)
            segment_counts = list(zip(unique_vectors, counts))
            pprint(segment_counts)

            # array([0, 0, 0], dtype=uint8), 421768), #background
            # (array([  0,   0, 128], dtype=uint8), 291418), #upper garment
            # (array([  0, 128,   0], dtype=uint8), 9393), #hair
            # (array([  0, 128, 128], dtype=uint8), 50277), #lower garment
            # (array([ 64, 128, 128], dtype=uint8), 14548), #left hand
            # (array([192, 128,   0], dtype=uint8), 33325), #face
            # (array([192, 128, 128], dtype=uint8), 14855)] #right hand

            # (array([ 0,  0, 64], dtype=uint8), 20918),
            # (array([  0,   0, 192], dtype=uint8), 20264), #left shoe
            # (array([  0, 128,   0], dtype=uint8), 64359),
            # (array([  0, 128,  64], dtype=uint8), 21031), #right shoe
            # (array([  0, 128, 192], dtype=uint8), 76005),
            # (array([128,   0,   0], dtype=uint8), 102761),
            # (array([128,   0,  64], dtype=uint8), 89881),
            # (array([128,   0, 192], dtype=uint8), 93931),
            # (array([128, 128,   0], dtype=uint8), 39445),
            # (array([128, 128,  64], dtype=uint8), 44930),
            # (array([128, 128, 192], dtype=uint8), 59772)]

            # color_code_list = [[64,128,128], [192,128,128]] #left and right hands
            # color_code_list = [[192,128,0]] #face 
            # color_code_list = [[0,128,0]] #hair
            # color_code_list = [[0,128,128]] #lower garment
            # color_code_list = [[0,0,128]] #upper garment
            # color_code_list = [[0,0,0]] #background

            return segment_counts
 
        def blend_images(cv2_input, cv2_inputseg, cv2_controlnetoutput, cv2_controlnetoutputseg, color_code_dict):

            # Helper function to create masks
            def get_mask_from_colors(image, color_list):
                mask = np.zeros_like(image[:,:,0])
                for color in color_list:
                    lowerb = np.array(color, dtype=np.uint8)
                    upperb = np.array(color, dtype=np.uint8)
                    temp_mask = cv2.inRange(image, lowerb, upperb)
                    mask = cv2.bitwise_or(mask, temp_mask)
                return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            input_facehair = get_mask_from_colors(cv2_inputseg, color_code_dict['face_hair'])
            input_background = get_mask_from_colors(cv2_inputseg, color_code_dict['background'])
            input_rest = cv2.bitwise_not(cv2.add(input_facehair, input_background))

            controlnet_facehair = get_mask_from_colors(cv2_controlnetoutputseg, color_code_dict['face_hair'])

            # Initial Image: Set it to input_rest
            blended_image = np.copy(cv2_input)
            blended_image = cv2.bitwise_and(blended_image, input_rest)

            # blended_image = combined_image(cv2_input, np.zeros_like(cv2_input), fuzzify(input_rest))


            # Stage 2: Overlay face and hair pixels from controlnet_facehair onto blended_image
            face_hair_region = cv2.bitwise_and(cv2_controlnetoutput, controlnet_facehair)
            inverse_face_hair_mask = cv2.bitwise_not(controlnet_facehair)
            blended_without_facehair = cv2.bitwise_and(blended_image, inverse_face_hair_mask)
            blended_image = cv2.add(blended_without_facehair, face_hair_region)

            # blended_facehair = combined_image(blended_image, cv2_controlnetoutput, fuzzify(controlnet_facehair))


            # Stage 3: Overlay the remaining pixels with white
            remaining_mask = cv2.bitwise_not(cv2.add(input_rest, controlnet_facehair))
            inverse_remaining_mask = cv2.bitwise_not(remaining_mask)
            
            white_fill = np.ones_like(cv2_input) * 255  # Create an image filled with white
            white_region = cv2.bitwise_and(white_fill, remaining_mask)
            
            blended_without_remaining = cv2.bitwise_and(blended_image, inverse_remaining_mask)
            blended_image = cv2.add(blended_without_remaining, white_region)

            # remaining_mask = cv2.bitwise_not(cv2.add(input_rest, controlnet_facehair))
            # blended_image = combined_image(blended_facehair, cv2_input, fuzzify(remaining_mask))



            return blended_image

        def tensor_to_cv2_img(tensor, remove_alpha=False):
            i = 255. * tensor.squeeze(0).cpu().numpy()  # This will give us (H, W, C)
            img = np.clip(i, 0, 255).astype(np.uint8)
            return img

        def cv2_img_to_tensor(img):
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img)[None,]
            return img

        cv2_input = tensor_to_cv2_img(input)    
        cv2_inputseg = tensor_to_cv2_img(inputseg)
        cv2_controlnetoutput = tensor_to_cv2_img(controlnetoutput)
        cv2_controlnetoutputseg = tensor_to_cv2_img(controlnetoutputseg)

        cv2_inputseg = cv2.resize(cv2_inputseg,(cv2_input.shape[1],cv2_input.shape[0]),interpolation=cv2.INTER_NEAREST)
        cv2_controlnetoutput = cv2.resize(cv2_controlnetoutput,(cv2_input.shape[1],cv2_input.shape[0]),interpolation=cv2.INTER_NEAREST)
        cv2_controlnetoutputseg = cv2.resize(cv2_controlnetoutputseg,(cv2_input.shape[1],cv2_input.shape[0]),interpolation=cv2.INTER_NEAREST)

        
        # # cv2_seg = cv2.resize(cv2_seg,(cv2_image.shape[1],cv2_image.shape[0]),interpolation=cv2.INTER_NEAREST)

        # color_code_list = [[192,128,0],[0,128,0]] #face and hair 
        color_code_dict = {
            'face_hair' : [[192,128,0],[0,128,0]],
            'background' : [[0,0,0]],
        }
        bimage = blend_images(cv2_input,cv2_inputseg,cv2_controlnetoutput,cv2_controlnetoutputseg,color_code_dict)
        

        # bimage = bounded_image(cv2_inputseg,color_code_list,cv2_input)



        output_img = cv2_img_to_tensor(bimage)
        
        return (output_img,)


class TRI3DPositiontHands:
    def __init__(self):
        pass    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "seg" : ("IMAGE",),
                "handimg" : ("IMAGE",),
                "margin" : ("INT", {"default": 15, "min": 0 }),
                "left_hand" : ("BOOLEAN", {"default": True}),
                "right_hand" : ("BOOLEAN", {"default": True}),
                "head" : ("BOOLEAN", {"default": False}),
                "hair" : ("BOOLEAN", {"default": False}),
                "left_leg" : ("BOOLEAN", {"default": False}),
                "right_leg" : ("BOOLEAN", {"default": False}),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "main"
    CATEGORY = "TRI3D"
    def main(self, image,seg,handimg,margin,left_hand,right_hand,head,hair,left_leg,right_leg):
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

        cv2_image = tensor_to_cv2_img(image)    
        cv2_seg = tensor_to_cv2_img(seg)
        # cv2_seg = cv2.resize(cv2_seg,(cv2_image.shape[1],cv2_image.shape[0]),interpolation=cv2.INTER_NEAREST)

        # 128 128 64 / 128 128 192
        # color_code_list = [[128,128,64], [128,128,192]]
        # color_code_list = [[64,128,128], [192,128,128]]
        color_code_list = []
        if left_hand:
            color_code_list.append([64,128,128])
        if right_hand:
            color_code_list.append([192,128,128])
        if head:
            color_code_list.append([192,128,0])
        if hair:
            color_code_list.append([0,128,0])
        if left_leg:
            color_code_list.append([192,0,0])
        if right_leg:
            color_code_list.append([64,128,0])

        positions = bounded_image_points(cv2_seg,color_code_list,cv2_image)


        
        cv2_handimg = tensor_to_cv2_img(handimg)
        #Resize cv2_handimg to positions

        print("before resizing ",cv2_handimg.shape,"handimg.shape")
        cv2_handimg = cv2.resize(cv2_handimg,(positions[2],positions[3]),interpolation=cv2.INTER_NEAREST)


        print(positions,"positions")
        print(cv2_image.shape,"cv2img.shape")
        print(cv2_handimg.shape,"handimg.shape")

        #position cv2_handimg in cv2_image
        cv2_image[positions[1]:positions[1]+positions[3],positions[0]:positions[0]+positions[2]] = cv2_handimg

        b_tensor_img = cv2_img_to_tensor(cv2_image)
        
        return (b_tensor_img,)

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
        from pprint import pprint

        def tensor_to_cv2_img(tensor, remove_alpha=False):
            i = 255. * tensor.cpu().numpy()  # This will give us (H, W, C)
            img = np.clip(i, 0, 255).astype(np.uint8)
            return img

        def cv2_img_to_tensor(img):
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img)[None,]
            return img

        batch_results = []

        for i in range(images.shape[0]):
            image = images[i]
            cv2_image = tensor_to_cv2_img(image)    
            cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

            ATR_PATH = 'custom_nodes/tri3d-comfyui-nodes/atr_node/'
            ATR_INPUT_PATH = ATR_PATH + 'input/'
            ATR_OUTPUT_PATH = ATR_PATH + 'output/'

            # Create the input directory if it does not exist
            shutil.rmtree(ATR_INPUT_PATH, ignore_errors=True)
            os.makedirs(ATR_INPUT_PATH, exist_ok=True)

            shutil.rmtree(ATR_OUTPUT_PATH, ignore_errors=True)
            os.makedirs(ATR_OUTPUT_PATH, exist_ok=True)

            cv2.imwrite(ATR_INPUT_PATH + "image.png", cv2_image)

            # Run the ATR model
            cwd = os.getcwd()
            os.chdir(ATR_PATH)
            os.system("python simple_extractor.py --dataset atr --model-restore 'checkpoints/atr.pth' --input-dir input --output-dir output")

            # Load the segmentation image
            os.chdir(cwd)
            cv2_segm = cv2.imread(ATR_OUTPUT_PATH + 'image.png')
            cv2_segm = cv2.cvtColor(cv2_segm, cv2.COLOR_BGR2RGB)

            b_tensor_img = cv2_img_to_tensor(cv2_segm)
            batch_results.append(b_tensor_img.squeeze(0))

        batch_results = torch.stack(batch_results)

        return (batch_results,)


class TRI3DATRParse:
    def __init__(self):
        pass    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "main"
    CATEGORY = "TRI3D"
    def main(self, image):
        import cv2
        import numpy as np
        import torch
        import os 
        import shutil
        from pprint import pprint

        def tensor_to_cv2_img(tensor, remove_alpha=False):
            i = 255. * tensor.squeeze(0).cpu().numpy()  # This will give us (H, W, C)
            img = np.clip(i, 0, 255).astype(np.uint8)
            return img

        def cv2_img_to_tensor(img):
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img)[None,]
            return img

        cv2_image = tensor_to_cv2_img(image)    
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)


        ATR_PATH = 'custom_nodes/tri3d-comfyui-nodes/atr_node/'
        ATR_INPUT_PATH = ATR_PATH + 'input/'
        ATR_OUTPUT_PATH = ATR_PATH + 'output/'

        # Create the input directory if it does not exist
        shutil.rmtree(ATR_INPUT_PATH, ignore_errors=True)
        os.makedirs(ATR_INPUT_PATH, exist_ok=True)

        shutil.rmtree(ATR_OUTPUT_PATH, ignore_errors=True)
        os.makedirs(ATR_OUTPUT_PATH, exist_ok=True)

        cv2.imwrite(ATR_INPUT_PATH + "image.png",cv2_image)

        # Run the ATR model
        cwd = os.getcwd()
        os.chdir(ATR_PATH)
        os.system("python simple_extractor.py --dataset atr --model-restore 'checkpoints/atr.pth' --input-dir input --output-dir output")

        # Load the segmentation image

        os.chdir(cwd)
        cv2_segm = cv2.imread(ATR_OUTPUT_PATH + 'image.png')
        cv2_segm = cv2.cvtColor(cv2_segm, cv2.COLOR_BGR2RGB)

        b_tensor_img = cv2_img_to_tensor(cv2_segm)
        
        return (b_tensor_img,)



# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "tri3d-extract-hand": TRI3DExtractHand,
    "tri3d-fuzzification": TRI3DFuzzification,
    "tri3d-position-hands": TRI3DPositiontHands,
    "tri3d-atr-parse": TRI3DATRParse,
    "tri3d-atr-parse-batch": TRI3DATRParseBatch,

}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "tri3d-extract-hand": "Extract Hand",
    "tri3d-fuzzification" : "Fuzzification",
    "tri3d-position-hands" : "Position Hands",
    "tri3d-atr-parse" : "ATR Parse",
    "tri3d-atr-parse-batch" : "ATR Parse Batch",
}
