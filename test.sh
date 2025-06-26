#!/bin/bash

# Function to kill all GPU processes
kill_gpu_processes() {
    echo "Checking for GPU processes..."
    # Get all PIDs from nvidia-smi, excluding the header and processes that might not exist
    PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -v "No running processes found" | tr -d ' ')
    
    if [ -n "$PIDS" ]; then
        echo "Found GPU processes with PIDs: $PIDS"
        echo "Killing GPU processes..."
        for PID in $PIDS; do
            if [ -n "$PID" ] && [ "$PID" -gt 0 ] 2>/dev/null; then
                echo "Killing process $PID"
                kill -9 "$PID" 2>/dev/null || echo "Process $PID already terminated or permission denied"
            fi
        done
        # Wait a moment for processes to terminate
        sleep 2
        echo "GPU process cleanup completed"
    else
        echo "No GPU processes found"
    fi
}

# Kill GPU processes before starting
echo "=== Cleaning up GPU processes before test ==="
kill_gpu_processes

source .venv/bin/activate
COMMAND="python main.py head --dataset test_recordings"
echo $COMMAND
$COMMAND
COMMAND="python main.py dump_jsonl"
echo $COMMAND
$COMMAND


# Kill GPU processes after test completes
echo "=== Cleaning up GPU processes after test ==="
kill_gpu_processes

python -c "
import json
import sys
sys.path.append('.')
from stats import print_status

# Load and print status
with open('status.json', 'r') as f:
    status = json.load(f)
    print_status(status)
"