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

        
        def bounded_image(img,color_code_list, canny_img):
            import cv2
            import numpy as np
            input_img = canny_img
            seg_img = img

            # Create a mask for hands
            hand_mask = np.zeros_like(seg_img[:,:,0])
            for color in color_code_list:
                hand_mask += cv2.inRange(seg_img, color, color)

            # Find contours to get the bounding box of the hands
            contours, _ = cv2.findContours(hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Extract the region from the original image
                hand_region = input_img[y:y+h, x:x+w]
            
            return hand_region
        

        def get_segment_counts(segm):
            # Load the segmentation image

            # Reshape the image array to be 2D
            reshaped = segm.reshape(-1, segm.shape[-1])

            # Find unique vectors and their counts
            unique_vectors, counts = np.unique(reshaped, axis=0, return_counts=True)
            segment_counts = list(zip(unique_vectors, counts))
            # pprint(segment_counts)
            return segment_counts

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
        print("color_code_list",color_code_list)
        # segs = get_segment_counts(cv2_seg)
        bimage = bounded_image(cv2_seg,color_code_list,cv2_image)

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
