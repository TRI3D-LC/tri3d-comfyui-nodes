#!/bin/bash

# Create main checkpoint directory
mkdir -p ckpt

# Create subdirectories
mkdir -p ckpt/densepose
mkdir -p ckpt/humanparsing
mkdir -p ckpt/openpose/ckpts

# Download files for densepose
wget -P ckpt/densepose https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/densepose/model_final_162be9.pkl

# Download files for humanparsing
wget -P ckpt/humanparsing https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/humanparsing/parsing_atr.onnx
wget -P ckpt/humanparsing https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/humanparsing/parsing_lip.onnx

# Download files for openpose
wget -P ckpt/openpose/ckpts https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/openpose/ckpts/body_pose_model.pth

echo "Download completed!"

