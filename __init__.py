class Example:
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
    FUNCTION = "test"
    CATEGORY = "ALPHA"
    def test(self, image,seg):
        import cv2
        import numpy as np
        import torch

        
        def bounded_image(seg_img, color_code_list, input_img):
            import cv2
            import numpy as np
            print(seg_img.shape, seg_img.dtype,"seg_img")
            
            # Create a mask for hands
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
            margin = 10
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

        # 128 128 64 / 128 128 192
        color_code_list = [[128,128,64], [128,128,192]]
        print("Hello")
        bimage = bounded_image(cv2_seg,color_code_list,cv2_image)
        print(bimage.shape, bimage.dtype,"bimage")
        b_tensor_img = cv2_img_to_tensor(bimage)
        
        return (b_tensor_img,)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "tri3d-extract-hand": Example
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "tri3d-extract-hand": "Extract Hand"
}
