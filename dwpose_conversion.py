import os
import json
import torch
import numpy as np
import folder_paths

print("Loading TRI3D_SavePoseKeypointsJSON module")

class SaveFlattenedPoseKpsAsJsonFile:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_kps": ("POSE_KEYPOINT",),
                "file_path": ("STRING", {"default": "dwpose/keypoints/input.json"})
            }
        }
    RETURN_TYPES = (
        "STRING",
    )
    FUNCTION = "save_flattened_pose_kps"
    OUTPUT_NODE = True
    CATEGORY = "ControlNet Preprocessors/Pose Keypoint Postprocess"

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    def _flatten_openpose_dict(self, pose_dict: dict) -> dict:
        """
        Converts a single OpenPose dictionary into flattened format.
        """
        # Get canvas dimensions from the input dictionary
        H = pose_dict.get('canvas_height', 512)
        W = pose_dict.get('canvas_width', 512)
        
        flat_keypoints = []

        # Check if any person was detected
        if not pose_dict.get('people'):
            # If no people, return a list of 130 invalid keypoints
            flat_keypoints.extend([[-1, -1]] * 130)
            return {"height": H, "width": W, "keypoints": flat_keypoints}

        person = pose_dict['people'][0]  # Process the first person found

        # Helper function to process each body part
        def process_part(keypoints_data, expected_length):
            processed_kps = []
            if keypoints_data:
                # Iterate in steps of 3 (x, y, confidence)
                for i in range(0, len(keypoints_data), 3):
                    x, y, conf = keypoints_data[i], keypoints_data[i+1], keypoints_data[i+2]
                    # Use confidence score to check for validity. If 0, it's a missing point.
                    if conf > 0:
                        processed_kps.append([x, y])
                    else:
                        processed_kps.append([-1, -1])
            
            # Ensure the list has the exact expected length
            while len(processed_kps) < expected_length:
                processed_kps.append([-1, -1])
            
            return processed_kps

        # Process parts in order: body -> face -> left hand -> right hand
        body_kps = process_part(person.get('pose_keypoints_2d'), 18)
        face_kps = process_part(person.get('face_keypoints_2d'), 70)
        left_hand_kps = process_part(person.get('hand_left_keypoints_2d'), 21)
        right_hand_kps = process_part(person.get('hand_right_keypoints_2d'), 21)

        # Combine all parts into the final flat list
        flat_keypoints.extend(body_kps)
        flat_keypoints.extend(face_kps)
        flat_keypoints.extend(left_hand_kps)
        flat_keypoints.extend(right_hand_kps)
        
        return {"height": H, "width": W, "keypoints": flat_keypoints}

    def save_flattened_pose_kps(self, pose_kps, file_path):
        # filename_prefix += self.prefix_append
        
        # # Get the save path using the first pose keypoint's dimensions
        # full_output_folder, filename, counter, subfolder, filename_prefix = \
        #     folder_paths.get_save_image_path(filename_prefix, self.output_dir, 
        #                                   pose_kps[0]["canvas_width"], 
        #                                   pose_kps[0]["canvas_height"])
        
        # Process each pose keypoint in the batch
        flattened_poses = []
        for pose_dict in pose_kps:
            flattened_data = self._flatten_openpose_dict(pose_dict)
            flattened_poses.append(flattened_data)
            
        # # Save the flattened data
        # file = f"{filename}_{counter:05}.json"
        # save_path = os.path.join(full_output_folder, file)

        cur_file_dir = os.path.dirname(os.path.realpath(__file__))
        save_path = os.path.join(cur_file_dir,
                                          file_path)
        
        with open(save_path, 'w') as f:
            if len(flattened_poses) == 1:
                json.dump(flattened_poses[0], f, indent=4)  # Save single pose directly
            else:
                json.dump(flattened_poses, f, indent=4)  # Save batch as array
                
        print(f"Saved flattened pose keypoints to: {save_path}")
        return (save_path,)
