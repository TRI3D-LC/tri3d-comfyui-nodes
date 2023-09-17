class TRI3DExtractHand:
    def __init__(self):
        pass    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "seg" : ("IMAGE",),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "main"
    CATEGORY = "TRI3D"
    def main(self, image,seg):
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
            margin = 25
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
        color_code_list = [[64,128,128], [192,128,128]]
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


            # # Stage 2: Fill face and hair pixels using controlnet_facehair
            face_hair_region = cv2.bitwise_and(cv2_controlnetoutput, controlnet_facehair)
            blended_image = cv2.add(blended_image, face_hair_region)

            # Stage 3: Fill the background pixels using input_background
            inverse_face_hair_mask = cv2.bitwise_not(controlnet_facehair)
            remaining_mask = cv2.bitwise_and(input_background, inverse_face_hair_mask)
            background_region = cv2.bitwise_and(cv2_input, remaining_mask)
            blended_image = cv2.add(blended_image, background_region)

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

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "tri3d-extract-hand": TRI3DExtractHand,
    "tri3d-fuzzification": TRI3DFuzzification
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "tri3d-extract-hand": "Extract Hand",
    "tri3d-fuzzification" : "Fuzzification"
}
