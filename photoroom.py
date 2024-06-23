import http.client
import mimetypes
import os
import uuid
import requests
import numpy as np
import torch
import cv2
from PIL import Image
import io

class TRI3D_photoroom_bgremove_api:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
            },
        }

    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", )
    CATEGORY = "TRI3D"

    def run(self, images):
        import http.client
        import mimetypes
        import os
        import uuid
        import dotenv

        dotenv.load_dotenv()

        # Read the API key from the environment variable
        PHOTOROOM_API_KEY = os.getenv('PHOTOROOM_API_KEY')
        def tensor_to_cv2_img(tensor, remove_alpha=False):
            i = 255. * tensor.cpu().numpy()  # This will give us (H, W, C)
            img = np.clip(i, 0, 255).astype(np.uint8)
            return img

        def cv2_img_to_tensor(img):
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img)[
                None,
            ]
            return img
        

        # Please replace with your own apiKey

        def remove_background(input_image_path, output_image_path,apiKey):
            # Define multipart boundary
            boundary = '----------{}'.format(uuid.uuid4().hex)

            # Get mimetype of image
            content_type, _ = mimetypes.guess_type(input_image_path)
            if content_type is None:
                content_type = 'application/octet-stream'  # Default type if guessing fails

            # Prepare the POST data
            with open(input_image_path, 'rb') as f:
                image_data = f.read()
            filename = os.path.basename(input_image_path)

            body = (
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"image_file\"; filename=\"{filename}\"\r\n"
            f"Content-Type: {content_type}\r\n\r\n"
            ).encode('utf-8') + image_data + f"\r\n--{boundary}--\r\n".encode('utf-8')
            
            # Set up the HTTP connection and headers
            conn = http.client.HTTPSConnection('sdk.photoroom.com')

            headers = {
                'Content-Type': f'multipart/form-data; boundary={boundary}',
                'x-api-key': apiKey
            }

            # Make the POST request
            conn.request('POST', '/v1/segment', body=body, headers=headers)
            response = conn.getresponse()

            # Handle the response
            if response.status == 200:
                response_data = response.read()
                with open(output_image_path, 'wb') as out_f:
                    out_f.write(response_data)
                print("Image saved to", output_image_path)
            else:
                print(f"Error: {response.status} - {response.reason}")
                print(response.read())

            # Close the connection
            conn.close()

        

        OUTPUT_FOLDER = "output/"
        batch_results = []
        for i in range(images.shape[0]):
            image = images[i]
            cv2_image = tensor_to_cv2_img(image)
            cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
            import random 
            random_number = random.randint(0, 100000)
            output_path = OUTPUT_FOLDER + f"output{i}_{random_number}.png"
            input_path = OUTPUT_FOLDER + f"input{i}_{random_number}.png"
            cv2.imwrite(input_path, cv2_image)
            remove_background(input_path, output_path, PHOTOROOM_API_KEY)

            print(input_path, output_path)
            cv2_segm = cv2.imread(output_path, cv2.IMREAD_UNCHANGED)
            cv2_segm = cv2.cvtColor(cv2_segm, cv2.COLOR_BGRA2RGBA)
            b_tensor_img = cv2_img_to_tensor(cv2_segm)
            batch_results.append(b_tensor_img.squeeze(0))

        batch_results = torch.stack(batch_results)

        return (batch_results,)
            


        
      

