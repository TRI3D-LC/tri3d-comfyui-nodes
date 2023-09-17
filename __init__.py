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

        cv2_input = tensor_to_cv2_img(input)    
        cv2_inputseg = tensor_to_cv2_img(inputseg)
        cv2_output = tensor_to_cv2_img(controlnetoutput)
        cv2_outputseg = tensor_to_cv2_img(controlnetoutputseg)

        # # cv2_seg = cv2.resize(cv2_seg,(cv2_image.shape[1],cv2_image.shape[0]),interpolation=cv2.INTER_NEAREST)

        # # 128 128 64 / 128 128 192
        # # color_code_list = [[128,128,64], [128,128,192]]
        # color_code_list = [[64,128,128], [192,128,128]]
        # bimage = bounded_image(cv2_seg,color_code_list,cv2_image)



        output_img = cv2_img_to_tensor(cv2_inputseg)
        
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
