class Example:
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
    FUNCTION = "test"
    CATEGORY = "ALPHA"
    def test(self, image):
        import cv2
        import numpy as np
        import torch
        def tensor_to_cv2_img(tensor, remove_alpha=False):
            i = 255. * tensor.cpu().numpy()
            img = cv2.cvtColor(np.clip(i, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
            return img
        
        def cv2_img_to_tensor(img):
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img)[None,]
            return img

        print("image", image.shape, type(image))
        cv2_image = tensor_to_cv2_img(image)    
        print("cv2_image", cv2_image.shape, type(cv2_image))            
        result_tensor_img = cv2_img_to_tensor(cv2_image)
        print("result_tensor_img", result_tensor_img.shape, type(result_tensor_img))
        
        return (result_tensor_img,)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "tri3d-extract-hand": Example
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "tri3d-extract-hand": "Extract Hand"
}
