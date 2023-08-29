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
            mask = np.zeros_like(canny_img)

            for color_code in color_code_list:
                lower = np.array(color_code, dtype = "uint8")
                upper = np.array(color_code, dtype = "uint8")
                color_mask = cv2.inRange(img, lower, upper)
                contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for cnt in contours:
                    hull = cv2.convexHull(cnt)
                    cv2.drawContours(mask, [hull], 0, (255), thickness=cv2.FILLED)

            kernel = np.ones((10,10),np.uint8)  # Kernel size determines the size of the dilation (margin size)
            mask = cv2.dilate(mask, kernel, iterations = 1)
            bounded_canny_img = cv2.bitwise_and(canny_img, mask)
            return bounded_canny_img
        

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
        print("segs",segs)
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
