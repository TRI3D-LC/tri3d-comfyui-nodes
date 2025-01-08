import cv2, os, math, json
import numpy as np
import matplotlib


def get_input_pose_type(input_keypoints):
    left_shoulder_x = input_keypoints[2][0]
    right_shoulder_x = input_keypoints[5][0]

    if left_shoulder_x - right_shoulder_x < 0:
        return "front_pose"
    else:
        return "back_pose"
    pass
def draw_bodypose(canvas: np.ndarray, keypoints: list) -> np.ndarray:

    H, W, _ = canvas.shape

    stickwidth = 4

    limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5], 
        [6, 7], [7, 8], [2, 9], [9, 10], 
        [10, 11], [2, 12], [12, 13], [13, 14], 
        [2, 1], [1, 15], [15, 17], [1, 16], 
        [16, 18],
    ]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
            [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
            [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for (k1_index, k2_index), color in zip(limbSeq, colors):

        keypoint1 = keypoints[k1_index - 1]
        keypoint2 = keypoints[k2_index - 1]

        if -1 in keypoint1 or -1 in keypoint2:
            continue

        Y = np.array([keypoint1[0], keypoint2[0]]) 
        X = np.array([keypoint1[1], keypoint2[1]]) 
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, [int(float(c) * 0.6) for c in color])
    for i, (keypoint, color) in enumerate(zip(keypoints, colors)):
        if -1 in keypoint:
            continue
        
        x, y = keypoint[0], keypoint[1]
        cv2.circle(canvas, (int(x), int(y)), 4, color, thickness=-1)
    return canvas

def draw_handpose(canvas: np.ndarray, keypoints: list) -> np.ndarray:
    
    H, W, _ = canvas.shape

    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

    for ie, (e1, e2) in enumerate(edges):
        k1 = keypoints[e1]
        k2 = keypoints[e2]
        if k1 is None or k2 is None:
            continue
      
        x1 = int(k1[0])
        y1 = int(k1[1])
        x2 = int(k2[0])
        y2 = int(k2[1])
        
        cv2.line(canvas, (x1, y1), (x2, y2), matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * 255, thickness=2)

    for keypoint in keypoints:
        if keypoint is None:
            continue

        x, y = keypoint[0], keypoint[1]
        x = int(x)
        y = int(y)
        cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
    return canvas

def rotate(src_keypoints, dest_keypoints, point1_idx, point2_idx):

    x1,y1 = src_keypoints[point1_idx]
    x2,y2 = src_keypoints[point2_idx]
    x3,y3 = dest_keypoints[point1_idx]
    x4,y4 = dest_keypoints[point2_idx]
    
    x4t, y4t = x4-x3, y4-y3
    
    src_angle = np.arctan2(y2 - y1, x2 - x1)
    dest_angle = np.arctan2(y4 - y3, x4 - x3)
    target_angle = src_angle-dest_angle

    x4_rotated = round((x4t * math.cos(target_angle) - y4t * math.sin(target_angle)))
    y4_rotated = round((x4t * math.sin(target_angle) + y4t * math.cos(target_angle)))
    
    x4_rotated, y4_rotated = x4_rotated+x3, y4_rotated+y3
    
    dest_keypoints[point2_idx] = [x4_rotated, y4_rotated]
    
    return dest_keypoints

def scale(src_keypoints, dest_keypoints, ref_point1_idx, ref_point2_idx, point1_idx, point2_idx):
    ref_x1,ref_y1 = src_keypoints[ref_point1_idx]
    ref_x2,ref_y2 = src_keypoints[ref_point2_idx]
    ref_x3,ref_y3 = dest_keypoints[ref_point1_idx]
    ref_x4,ref_y4 = dest_keypoints[ref_point2_idx]
    
    x1,y1 = src_keypoints[point1_idx]
    x2,y2 = src_keypoints[point2_idx]
    x3,y3 = dest_keypoints[point1_idx]
    x4,y4 = dest_keypoints[point2_idx]
    
    src_ref_len = np.linalg.norm(np.array([ref_x1, ref_y1]) - np.array([ref_x2, ref_y2]))             #src ref part distance
    dest_ref_len = np.linalg.norm(np.array([ref_x3, ref_y3]) - np.array([ref_x4, ref_y4]))          #dest ref part distance

    src_targ_len = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))             #src targ part distance
    dest_targ_len = np.linalg.norm(np.array([x3, y3]) - np.array([x4,y4]))          #dest targ part distance

    src_targ_ref_ratio = src_targ_len / src_ref_len                  #src targ to ref ratio
    dest_targ_ref_ratio = dest_targ_len / dest_ref_len            #dest targ to ref ratio

    scale = src_targ_ref_ratio / dest_targ_ref_ratio

    x4_scaled = x3 + (x4-x3) * scale
    y4_scaled = y3 + (y4-y3) * scale
    
    dest_keypoints[point2_idx] = [x4_scaled, y4_scaled]
    
    return dest_keypoints

def scale_hand(src_keypoints, dest_keypoints, ref_point1_idx, ref_point2_idx, starting_idx):
    
    """
        scales all 20 hand lines for the given reference wrist length
    """

    rotate(src_keypoints, dest_keypoints, 0+starting_idx, 1+starting_idx)
    scale(src_keypoints, dest_keypoints, ref_point1_idx, ref_point2_idx, 0+starting_idx, 1+starting_idx)
    rotate(src_keypoints, dest_keypoints, 0+starting_idx, 5+starting_idx)
    scale(src_keypoints, dest_keypoints, ref_point1_idx, ref_point2_idx, 0+starting_idx, 5+starting_idx)
    rotate(src_keypoints, dest_keypoints, 0+starting_idx, 9+starting_idx)
    scale(src_keypoints, dest_keypoints, ref_point1_idx, ref_point2_idx, 0+starting_idx, 9+starting_idx)
    rotate(src_keypoints, dest_keypoints, 0+starting_idx, 13+starting_idx)
    scale(src_keypoints, dest_keypoints, ref_point1_idx, ref_point2_idx, 0+starting_idx, 13+starting_idx)
    rotate(src_keypoints, dest_keypoints, 0+starting_idx, 17+starting_idx)
    scale(src_keypoints, dest_keypoints, ref_point1_idx, ref_point2_idx, 0+starting_idx, 17+starting_idx)
    
    rotate(src_keypoints, dest_keypoints, 1+starting_idx, 2+starting_idx)
    scale(src_keypoints, dest_keypoints, ref_point1_idx, ref_point2_idx, 1+starting_idx, 2+starting_idx)
    rotate(src_keypoints, dest_keypoints, 2+starting_idx, 3+starting_idx)
    scale(src_keypoints, dest_keypoints, ref_point1_idx, ref_point2_idx, 2+starting_idx, 3+starting_idx)
    rotate(src_keypoints, dest_keypoints, 3+starting_idx, 4+starting_idx)
    scale(src_keypoints, dest_keypoints, ref_point1_idx, ref_point2_idx, 3+starting_idx, 4+starting_idx)
    
    rotate(src_keypoints, dest_keypoints, 5+starting_idx, 6+starting_idx)
    scale(src_keypoints, dest_keypoints, ref_point1_idx, ref_point2_idx, 5+starting_idx, 6+starting_idx)
    rotate(src_keypoints, dest_keypoints, 6+starting_idx, 7+starting_idx)
    scale(src_keypoints, dest_keypoints, ref_point1_idx, ref_point2_idx, 6+starting_idx, 7+starting_idx)
    rotate(src_keypoints, dest_keypoints, 7+starting_idx, 8+starting_idx)
    scale(src_keypoints, dest_keypoints, ref_point1_idx, ref_point2_idx, 7+starting_idx, 8+starting_idx)
   
    rotate(src_keypoints, dest_keypoints, 9+starting_idx, 10+starting_idx)
    scale(src_keypoints, dest_keypoints, ref_point1_idx, ref_point2_idx, 9+starting_idx, 10+starting_idx)
    rotate(src_keypoints, dest_keypoints, 10+starting_idx, 11+starting_idx)
    scale(src_keypoints, dest_keypoints, ref_point1_idx, ref_point2_idx, 10+starting_idx, 11+starting_idx)
    rotate(src_keypoints, dest_keypoints, 11+starting_idx, 12+starting_idx)
    scale(src_keypoints, dest_keypoints, ref_point1_idx, ref_point2_idx, 11+starting_idx, 12+starting_idx)
    
    rotate(src_keypoints, dest_keypoints, 13+starting_idx, 14+starting_idx)
    scale(src_keypoints, dest_keypoints, ref_point1_idx, ref_point2_idx, 13+starting_idx, 14+starting_idx)
    rotate(src_keypoints, dest_keypoints, 14+starting_idx, 15+starting_idx)
    scale(src_keypoints, dest_keypoints, ref_point1_idx, ref_point2_idx, 14+starting_idx, 15+starting_idx)
    rotate(src_keypoints, dest_keypoints, 15+starting_idx, 16+starting_idx)
    scale(src_keypoints, dest_keypoints, ref_point1_idx, ref_point2_idx, 15+starting_idx, 16+starting_idx)
    
    rotate(src_keypoints, dest_keypoints, 17+starting_idx, 18+starting_idx)
    scale(src_keypoints, dest_keypoints, ref_point1_idx, ref_point2_idx, 17+starting_idx, 18+starting_idx)
    rotate(src_keypoints, dest_keypoints, 18+starting_idx, 19+starting_idx)
    scale(src_keypoints, dest_keypoints, ref_point1_idx, ref_point2_idx, 18+starting_idx, 19+starting_idx)
    rotate(src_keypoints, dest_keypoints, 19+starting_idx, 20+starting_idx)
    scale(src_keypoints, dest_keypoints, ref_point1_idx, ref_point2_idx, 19+starting_idx, 20+starting_idx)
    
    return dest_keypoints

def rotate_hand(src_keypoints, dest_keypoints, starting_idx):
    
    """
        scales all 20 hand lines for the given reference wrist length
        src_keypoints: ref keypoints
        dest_keypoints: mannequin keypoints
        starting_idx: starting index of hand
    """

    rotate(src_keypoints, dest_keypoints, 0+starting_idx, 1+starting_idx)
    rotate(src_keypoints, dest_keypoints, 0+starting_idx, 5+starting_idx)
    rotate(src_keypoints, dest_keypoints, 0+starting_idx, 9+starting_idx)
    rotate(src_keypoints, dest_keypoints, 0+starting_idx, 13+starting_idx)
    rotate(src_keypoints, dest_keypoints, 0+starting_idx, 17+starting_idx)
    
    rotate(src_keypoints, dest_keypoints, 1+starting_idx, 2+starting_idx)
    rotate(src_keypoints, dest_keypoints, 2+starting_idx, 3+starting_idx)
    rotate(src_keypoints, dest_keypoints, 3+starting_idx, 4+starting_idx)
    
    rotate(src_keypoints, dest_keypoints, 5+starting_idx, 6+starting_idx)
    rotate(src_keypoints, dest_keypoints, 6+starting_idx, 7+starting_idx)
    rotate(src_keypoints, dest_keypoints, 7+starting_idx, 8+starting_idx)
   
    rotate(src_keypoints, dest_keypoints, 9+starting_idx, 10+starting_idx)
    rotate(src_keypoints, dest_keypoints, 10+starting_idx, 11+starting_idx)
    rotate(src_keypoints, dest_keypoints, 11+starting_idx, 12+starting_idx)
    
    rotate(src_keypoints, dest_keypoints, 13+starting_idx, 14+starting_idx)
    rotate(src_keypoints, dest_keypoints, 14+starting_idx, 15+starting_idx)
    rotate(src_keypoints, dest_keypoints, 15+starting_idx, 16+starting_idx)
    
    rotate(src_keypoints, dest_keypoints, 17+starting_idx, 18+starting_idx)
    rotate(src_keypoints, dest_keypoints, 18+starting_idx, 19+starting_idx)
    rotate(src_keypoints, dest_keypoints, 19+starting_idx, 20+starting_idx)
    
    return dest_keypoints

def get_torso_angles(keypoints):
    """
    Function to get torso angles
    """
    #left shoulder angle with x axis
    a,b = keypoints[2],keypoints[1]
    angle_radians = math.tan((a[1]-b[1])/(a[0]-b[0]))
    ls_angle = abs(math.degrees(angle_radians))
    
    #right shoulder angle with x axis
    a,b = keypoints[5],keypoints[1]
    angle_radians = math.tan((a[1]-b[1])/(a[0]-b[0]))
    rs_angle = abs(math.degrees(angle_radians))
    
    #getting bisector of torso and getting angle with y_axis
    a,b,c = keypoints[8], keypoints[1], keypoints[11]

    #torso bisector
    bi = [int((a[0]+c[0])/2), int((a[1]+c[1])/2)]

    #calculatng angle with y axis
    angle_radians = math.atan2((bi[1]-b[1]),(bi[0]-b[0]))
    torso_angle = 90 - abs(math.degrees(angle_radians))

    return ls_angle, rs_angle, torso_angle

def move(prev_point, new_point, dependent_points):
    """e.g if wrist is moved then rest of the hand points also be moved by same offset where """
    
    x_off = new_point[0] - prev_point[0]
    y_off = (new_point[1] - prev_point[1])
    
    for i in range(len(dependent_points)):
        if dependent_points[i] == [-1,-1]: continue
        dependent_points[i] = [dependent_points[i][0]+x_off, dependent_points[i][1]+y_off]
            
    return dependent_points

def switch_to_backpose(input_keypoints, input_width):
    """for the given straight pose flip horizontally """

    for i in range(len(input_keypoints)):
        if i in [0,14,15]:
            input_keypoints[i] = [-1, -1]
            continue
        x,y = input_keypoints[i]
        input_keypoints[i] = [input_width - x, y]

    return input_keypoints

def extract_torso_keypoints(keypoints):
    # Indices for torso-related keypoints
    torso_indices = [8, 9, 10, 11, 12, 13]
    return [keypoints[i] for i in torso_indices if keypoints[i] != [-1, -1]]

