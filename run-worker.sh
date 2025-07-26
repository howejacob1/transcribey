#!/bin/bash

# Function to kill all GPU processes
cleanup_gpu_processes() {
    echo "Cleaning up GPU processes..."
    # Get PIDs of all processes using GPU
    GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    
    if [ -n "$GPU_PIDS" ]; then
        echo "Found GPU processes with PIDs: $GPU_PIDS"
        for pid in $GPU_PIDS; do
            if [ -n "$pid" ] && [ "$pid" != "" ]; then
                echo "Killing GPU process $pid"
                kill -9 "$pid" 2>/dev/null || true
            fi
        done
        # Wait a moment for processes to clean up
        sleep 2
    else
        echo "No GPU processes found to clean up"
    fi
}

# Cleanup before starting
echo "=== GPU CLEANUP BEFORE STARTING ==="
cleanup_gpu_processes

# Setup trap to cleanup on exit
trap 'echo "=== GPU CLEANUP ON EXIT ==="; cleanup_gpu_processes' EXIT

source .venv/bin/activate
python main.py worker --production --url sftp://bantaim@banidk0/media/10900-hdd-0/