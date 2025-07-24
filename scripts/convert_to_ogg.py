#!/usr/bin/env python3
"""
WAV to OGG Vorbis converter using ffmpeg-python library.
Supports both local and remote operation modes.

Requirements:
- ffmpeg binary: sudo apt update && sudo apt install ffmpeg  
- ffmpeg-python library: pip install ffmpeg-python
- For remote mode: paramiko library for SFTP
"""
import argparse
import hashlib
import os
import queue
import random
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
try:
    import ffmpeg
except ImportError:
    print("ERROR: ffmpeg-python library not found!")
    print("Please install it with: pip install ffmpeg-python")
    exit(1)
# try PyAV first to avoid launching ffmpeg for every file
try:
    import av
    HAVE_PYAV = True
except ImportError:
    HAVE_PYAV = False

# Always use external ffmpeg – PyAV path is slower in practice
HAVE_PYAV = False

from mongo_utils import db
from utils import num_cores
from vcon_class import Vcon
import sftp 