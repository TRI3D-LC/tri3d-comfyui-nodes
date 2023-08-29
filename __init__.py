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

                # Generate a mask for the current color code
                lower = np.array(color_code, dtype = "uint8")
                upper = np.array(color_code, dtype = "uint8")
                color_mask = cv2.inRange(img, lower, upper)
                # cv2.imwrite(output_dir + str(color_code) + ".jpg",color_mask)
                # Find contours in the mask
                contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for cnt in contours:

                    # x, y, w, h = cv2.boundingRect(cnt)

                    # # Display the region within the bounding box on the mask
                    # mask[y:y+h, x:x+w] = 255

                    # # print("contour found",cnt)
                    # Get the convex hull
                    hull = cv2.convexHull(cnt)
                    # Draw the convex hull onto the mask
                    cv2.drawContours(mask, [hull], 0, (255), thickness=cv2.FILLED)

            # cv2.imwrite(output_dir + "mask_original.jpg",mask)
            # Add a margin around the hand
            kernel = np.ones((10,10),np.uint8)  # Kernel size determines the size of the dilation (margin size)
            mask = cv2.dilate(mask, kernel, iterations = 1)

            # Apply the mask to the canny image

            # cv2.imwrite(output_dir + "mask.jpg",mask)
            bounded_canny_img = cv2.bitwise_and(canny_img, mask)
            # print(bounded_canny_img.shape)
            #convert to gray scale
            bounded_canny_img = cv2.cvtColor(bounded_canny_img, cv2.COLOR_BGR2GRAY)
            return bounded_canny_img
        
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
