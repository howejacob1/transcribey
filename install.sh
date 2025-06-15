#!/usr/bin/env bash
set -e

# 0. Ensure python3-venv is installed (for ensurepip and venv support)
if ! python3 -m venv --help > /dev/null 2>&1; then
    echo "python3-venv is not installed. Installing..."
    sudo apt-get update
    sudo apt-get install -y python3-venv
fi

# 1. Check for a valid venv. If not present, create it.
if [ ! -f ".venv/bin/activate" ]; then
    echo "Virtual environment not found or is invalid. Creating/recreating..."
    # Remove potentially broken venv directory
    rm -rf .venv
    python3 -m venv .venv
    echo "Virtual environment created."
fi

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Run install.py using the venv's python
python install.py