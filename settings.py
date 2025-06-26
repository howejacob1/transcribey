import os
import socket
import subprocess

# Settings for main.py
cache_dir = "/dev/shm/cache/"
processing_dir = os.path.join(cache_dir, "processing/")
downloading_dir = os.path.join(cache_dir, "downloading/")
lang_detect_threshold = 0.2
debug = True
cache_size_bytes = 1 * (1024**3)
hostname = socket.gethostname()
#transcribe_english_model_name = "nvidia/parakeet-tdt-0.6b-v2"
en_model_name = "nvidia/parakeet-tdt_ctc-110m"
non_en_model_name = "nvidia/canary-1b-flash"

gpu_ram_unusable = 3*(1024**3) # 5GB
max_download_threads = 1
gc_limit_bytes = 3*(1024**3)
vcon_queue_max_bytes = 1

# SFTP connection settings for make_vcons_from_sftp.py
sftp_url = "sftp://bantaim@127.0.0.1:22/home/bantaim/conserver/fake_wavs_cute/"
sample_rate = 16000
max_discover_workers = 1
discover_batch_size = 300  # Increased from 1 for much faster discovery
dont_overwhelm_server_time_seconds = 1

# MongoDB connection settings
preprocess_batch_timeout_seconds = 0.1
preprocess_batch_max_size = 8  # Smaller batch size
preprocess_batch_max_len = 4   # Smaller max length

min_audio_duration_seconds = 1.0
status_update_seconds = 20

lang_detect_batch_timeout_seconds = 0.1
lang_detect_batch_max_size = 8
lang_detect_batch_max_len = 4
lang_detect_batch_ready = 0.1

transcribe_batch_timeout_seconds = 0.1
transcribe_batch_max_size = 8
transcribe_batch_max_len = 4

queue_max_size = 200
die_after_no_measurements_time = 10

def get_version():
    """Get the most recent git tag"""
    result = subprocess.run(['git', 'describe', '--tags', '--abbrev=0'], 
                            capture_output=True, text=True, check=True)
    return result.stdout.strip()


version = get_version()