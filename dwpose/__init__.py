# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)
# 5th Edited by ControlNet (Improved JSON serialization/deserialization, and lots of bug fixs)
# This preprocessor is licensed by CMU for non-commercial use only.


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import torch
import numpy as np
from . import util
from .body import Body, BodyResult, Keypoint
from .hand import Hand
from .face import Face
from .types import PoseResult, HandResult, FaceResult
from huggingface_hub import hf_hub_download
from .wholebody import Wholebody # DW Pose
import warnings
# from ..util import HWC3, resize_image
import cv2
from PIL import Image

from typing import Tuple, List, Callable, Union, Optional

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y
    
def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img

def draw_poses(poses: List[PoseResult], H, W, draw_body=True, draw_hand=True, draw_face=True):
    """
    Draw the detected poses on an empty canvas.

    Args:
        poses (List[PoseResult]): A list of PoseResult objects containing the detected poses.
        H (int): The height of the canvas.
        W (int): The width of the canvas.
        draw_body (bool, optional): Whether to draw body keypoints. Defaults to True.
        draw_hand (bool, optional): Whether to draw hand keypoints. Defaults to True.
        draw_face (bool, optional): Whether to draw face keypoints. Defaults to True.

    Returns:
        numpy.ndarray: A 3D numpy array representing the canvas with the drawn poses.
    """
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    for pose in poses:
        if draw_body:
            canvas = util.draw_bodypose(canvas, pose.body.keypoints)

        if draw_hand:
            canvas = util.draw_handpose(canvas, pose.left_hand)
            canvas = util.draw_handpose(canvas, pose.right_hand)

        if draw_face:
            canvas = util.draw_facepose(canvas, pose.face)

    return canvas


def decode_json_as_poses(json_string: str, normalize_coords: bool = False) -> Tuple[List[PoseResult], int, int]:
    """ Decode the json_string complying with the openpose JSON output format
    to poses that controlnet recognizes.
    https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md

    Args:
        json_string: The json string to decode.
        normalize_coords: Whether to normalize coordinates of each keypoint by canvas height/width.
                          `draw_pose` only accepts normalized keypoints. Set this param to True if
                          the input coords are not normalized.
    
    Returns:
        poses
        canvas_height
        canvas_width                      
    """
    pose_json = json.loads(json_string)
    height = pose_json['canvas_height']
    width = pose_json['canvas_width']

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
    
    def decompress_keypoints(numbers: Optional[List[float]]) -> Optional[List[Optional[Keypoint]]]:
        if not numbers:
            return None
        
        assert len(numbers) % 3 == 0

        def create_keypoint(x, y, c):
            if c < 1.0:
                return None
            keypoint = Keypoint(x, y)
            return keypoint

        return [
            create_keypoint(x, y, c)
            for x, y, c in chunks(numbers, n=3)
        ]
    
    return (
        [
            PoseResult(
                body=BodyResult(keypoints=decompress_keypoints(pose.get('pose_keypoints_2d'))),
                left_hand=decompress_keypoints(pose.get('hand_left_keypoints_2d')),
                right_hand=decompress_keypoints(pose.get('hand_right_keypoints_2d')),
                face=decompress_keypoints(pose.get('face_keypoints_2d'))
            )
            for pose in pose_json['people']
        ],
        height,
        width,
    )


def encode_poses_as_json(poses: List[PoseResult], canvas_height: int, canvas_width: int) -> str:
    """ Encode the pose as a JSON string following openpose JSON output format:
    https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md
    """
    def compress_keypoints(keypoints: Union[List[Keypoint], None]) -> Union[List[float], None]:
        if not keypoints:
            return None
        
        return [
            value
            for keypoint in keypoints
            for value in (
                [float(keypoint.x), float(keypoint.y), 1.0]
                if keypoint is not None
                else [0.0, 0.0, 0.0]
            )
        ]

    return json.dumps({
        'people': [
            {
                'pose_keypoints_2d': compress_keypoints(pose.body.keypoints),
                "face_keypoints_2d": compress_keypoints(pose.face),
                "hand_left_keypoints_2d": compress_keypoints(pose.left_hand),
                "hand_right_keypoints_2d":compress_keypoints(pose.right_hand),
            }
            for pose in poses
        ],
        'canvas_height': canvas_height,
        'canvas_width': canvas_width,
    }, indent=4)

class DwposeDetector:
    """
    A class for detecting human poses in images using the Dwpose model.

    Attributes:
        model_dir (str): Path to the directory where the pose models are stored.
    """
    def __init__(self, dw_pose_estimation):
        self.dw_pose_estimation = dw_pose_estimation
    
    @classmethod
    def from_pretrained(cls, pretrained_model_or_path, det_filename=None, pose_filename=None, cache_dir=None):
        det_filename = det_filename or "yolox_l.onnx"
        pose_filename = pose_filename or "dw-ll_ucoco_384.onnx"

        if os.path.isdir(pretrained_model_or_path):
            det_model_path = os.path.join(pretrained_model_or_path, det_filename)
            pose_model_path = os.path.join(pretrained_model_or_path, pose_filename)
        else:
            det_model_path = hf_hub_download(pretrained_model_or_path, det_filename, cache_dir=cache_dir)
            pose_model_path = hf_hub_download(pretrained_model_or_path, pose_filename, cache_dir=cache_dir)

        return cls(Wholebody(det_model_path, pose_model_path))
    
    def to(self, device):
        warnings.warn("Currently DWPose doesn't support CUDA out-of-the-box.")
        return self

    def detect_poses(self, oriImg) -> List[PoseResult]:
        with torch.no_grad():
            keypoints_info = self.dw_pose_estimation(oriImg.copy())
            return Wholebody.format_result(keypoints_info)
    
    def __call__(self, input_image, detect_resolution=512, image_resolution=512, include_body=True, include_hand=False, include_face=False, hand_and_face=None, output_type="pil", **kwargs):
        if hand_and_face is not None:
            warnings.warn("hand_and_face is deprecated. Use include_hand and include_face instead.", DeprecationWarning)
            include_hand = hand_and_face
            include_face = hand_and_face

        if "return_pil" in kwargs:
            warnings.warn("return_pil is deprecated. Use output_type instead.", DeprecationWarning)
            output_type = "pil" if kwargs["return_pil"] else "np"
        if type(output_type) is bool:
            warnings.warn("Passing `True` or `False` to `output_type` is deprecated and will raise an error in future versions")
            if output_type:
                output_type = "pil"

        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)
        
        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)
        H, W, C = input_image.shape
        poses = self.detect_poses(input_image)
        keypoints = []
        if len(poses) > 0:
            if poses[0].body[0]:
                for i in range(len(poses[0].body[0])):
                    if poses[0].body[0][i] is not None:
                        keypoints.append(
                            (poses[0].body[0][i].x, poses[0].body[0][i].y))
                    else:
                        keypoints.append((-1,-1))
            else:
                keypoints.extend([(-1, -1)]*18)
            # print("appended body")
            
            if poses[0].face:
                for i in range(len(poses[0].face)):
                    if poses[0].face[i] is not None:
                        keypoints.append(
                            (poses[0].face[i].x, poses[0].face[i].y))
                    else:
                        keypoints.append((-1,-1))
            else:
                keypoints.extend([(-1, -1)]*70)
            # print("appended face")

            # print(len(poses[0].left_hand))
            if poses[0].left_hand:
                for i in range(len(poses[0].left_hand)):
                    if poses[0].left_hand[i] is not None:
                        keypoints.append(
                            (poses[0].left_hand[i].x, poses[0].left_hand[i].y))
                    else:
                        keypoints.append((-1,-1))
            else:
                keypoints.extend([(-1, -1)]*21)
            # print("appended left hand")

            if poses[0].right_hand:            
                for i in range(len(poses[0].right_hand)):
                    if poses[0].right_hand[i] is not None:
                        keypoints.append(
                            (poses[0].right_hand[i].x, poses[0].right_hand[i].y))
                    else:
                        keypoints.append((-1,-1))
            else:
                keypoints.extend([(-1, -1)]*21)
            # print("appended right hand")
        else:
            for i in range(130):
                keypoints.append((-1, -1))
        output_dict = {"height":H, "width":W, "keypoints":keypoints}
        # print("json:",output_dict)
        # print(len(keypoints))
        # json.dump(output_dict, open("C:/tri3d/pose_library/testing/garment/boy_trouser/keypoints.json","w"))
        
        canvas = draw_poses(poses, H, W, draw_body=include_body, draw_hand=include_hand, draw_face=include_face) 

        detected_map = canvas
        detected_map = HWC3(detected_map)
        
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)

        return detected_map, output_dict
