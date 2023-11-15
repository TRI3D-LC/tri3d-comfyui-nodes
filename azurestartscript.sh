#!/bin/bash

# Navigate to the project directory
cd /home/ubuntu/GITHUB/comfyanonymous/ComfyUI
# Activate the Anaconda environment
. /opt/anaconda/bin/activate
conda activate comfy

# Start a new tmux session or connect to an existing one
if ! tmux has-session -t run 2>/dev/null; then
	    tmux new-session -d -s run
	        tmux send-keys -t run 'python main.py --port 3000' C-m
fi

# Attach to the tmux session
tmux attach-session -t run

