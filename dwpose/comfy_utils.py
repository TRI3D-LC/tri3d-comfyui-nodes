import cv2, os, math, json
import numpy as np


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
    angle_radians = math.atan((bi[1]-b[1])/(bi[0]-b[0]))
    torso_angle = 90 - abs(math.degrees(angle_radians))

    return ls_angle, rs_angle, torso_angle