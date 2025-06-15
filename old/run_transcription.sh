#!/bin/bash

# Set PyTorch CUDA allocation configuration to avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Optional: Set other memory-related environment variables for better performance
export CUDA_LAUNCH_BLOCKING=0

# Run the main script with any passed arguments
python main.py "$@" 