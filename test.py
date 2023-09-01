import cv2 
import numpy as np
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


color_code_list = [[128,128,64], [128,128,192]]
seg_img = cv2.imread("input/seg.png",cv2.IMREAD_UNCHANGED)
input_img = cv2.imread("input/input_aligned.jpg",cv2.IMREAD_UNCHANGED)
print(seg_img.shape,"seg_img.shape")
print(input_img.shape,"input_img.shape")
# bimage = bounded_image(seg_img, color_code_list, input_img)
# cv2.imwrite("output/output.png",bimage)