
class Example:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "seg": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE","IMAGE")
    FUNCTION = "test"
    CATEGORY = "Example"
    

    

    def test(self, image, seg):
        import cv2
        import numpy as np
        import torch
        from collections import Counter


       

        def print_rgb_histogram(tensor):
            # Convert tensor to numpy array
            np_arr = tensor.cpu().numpy()

            # Ensure the tensor has the shape [B, H, W, C] or [H, W, C]
            if len(np_arr.shape) == 4:
                np_arr = np_arr[0]

            # Convert the array to a list of tuples (R, G, B)
            pixels = [tuple(pixel) for row in np_arr for pixel in row]

            # Get the most common RGB pixel values and their counts
            counter = Counter(pixels)
            common_pixels = counter.most_common(5)

            print("RGB Pixel Value : Frequency")
            for pixel, count in common_pixels:
                print(f"{pixel} : {count}")

                def tensor_to_cv2_img(tensor, remove_alpha=False):
                    np_img = tensor.cpu().numpy()
                    if len(np_img.shape) == 3:
                        np_img = np.transpose(np_img, (1, 2, 0))
                    np_img = (np_img * 255).astype(np.uint8)

                    # Check for alpha channel in image and optionally remove it
                    if remove_alpha and np_img.shape[-1] == 4:
                        np_img = np_img[:, :, :3]

                    return np_img
        print("image", image.shape)
        print("seg", seg.shape)
        print_rgb_histogram(image)
        print_rgb_histogram(seg)
       
        def tensor_to_cv2_img(tensor, remove_alpha=False):
            np_img = tensor.cpu().numpy()
            if len(np_img.shape) == 3:
                np_img = np.transpose(np_img, (1, 2, 0))
            np_img = (np_img * 255).astype(np.uint8)

            # Check for alpha channel in image and optionally remove it
            if remove_alpha and np_img.shape[-1] == 4:
                np_img = np_img[:, :, :3]

            return np_img

        def cv2_img_to_tensor(cv2_img):
            # Convert image values to [0, 1]
            cv2_img = cv2_img.astype(np.float32) / 255.0
            
            # Convert the numpy array to a torch tensor
            tensor = torch.from_numpy(cv2_img)
            
            # If the tensor has 3 dimensions (H, W, C), change to (C, H, W)
            if len(tensor.shape) == 3:
                tensor = tensor.permute(2, 0, 1)
                
            return tensor
        cv2_image = tensor_to_cv2_img(image)
        cv2_seg = tensor_to_cv2_img(seg, remove_alpha=True)

        print(cv2_image.shape)
        print(cv2_seg.shape)
                
        # Convert the added image back to a torch tensor
        result_tensor_seg = cv2_img_to_tensor(cv2_seg)
        result_tensor_img = cv2_img_to_tensor(cv2_image)
        
        return result_tensor_img, result_tensor_seg



# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "tri3d-extract-hand": Example
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "tri3d-extract-hand": "Extract hand region using segmentation map"
}
