import os
import socket
import subprocess

mark_non_english_as_corrupt = False
put_all_vcons_into_english_queue = False

# Settings for main.py
cache_dir = "/dev/shm/cache4/"
processing_dir = os.path.join(cache_dir, "processing/")
downloading_dir = os.path.join(cache_dir, "downloading/")
lang_detect_threshold = 0.2
debug = True
cache_size_bytes = 2 * (1024**3)  # 2GB cache (increased from 1GB for higher parallelism)
hostname = socket.gethostname()
#transcribe_english_model_name = "nvidia/parakeet-tdt-0.6b-v2"
en_model_name = "nvidia/parakeet-tdt_ctc-110m"
non_en_model_name = "nvidia/canary-180m-flash"

gpu_ram_unusable = 3*(1024**3) # 5GB
max_download_threads = 1
max_preprocess_workers = 1
max_transcription_workers = 4
max_non_en_transcription_workers = 1
max_lang_detect_workers = 2
gc_limit_bytes = 3*(1024**3)


# SFTP connection settings for make_vcons_from_sftp.py
sftp_url = "sftp://bantaim@banidk0:22/home/bantaim/conserver/fake_wavs_cute/"
sample_rate = 16000
max_discover_workers = 1
discover_batch_size = 1024  # Increased from 1 for much faster discovery
dont_overwhelm_server_time_seconds = 1

# SFTP Performance tuning
sftp_buffer_size = 100*1024*1024  # 128KB buffer for downloads (increased from 64KB)
sftp_prefetch_enabled = True  # Enable prefetch for better performance
sftp_parallel_downloads = 1  # Number of parallel downloads per batch (reduced to avoid SFTP corruption)
sftp_download_batch_size = 1  # Number of files to download in parallel (reduced to avoid connection issues)
sftp_download_timeout = 30  # Timeout for individual file downloads (seconds)

# MongoDB connection settings
preprocess_batch_timeout_seconds = 0.1
preprocess_batch_max_size = 8  # Batch size for better throughput
preprocess_batch_max_len = 4   # Smaller max length
preprocess_batch_default_size = 8  # Default batch size for collect_batch_with_timeout

# MongoDB performance settings
mongo_bulk_update_batch_size = 10
mongo_bulk_update_timeout_seconds = 1.0
mongo_reservation_batch_limit = 512
mongo_discovery_batch_size = 512  # Same as discover_batch_size for consistency
reserver_total_batch_size = 1*1024*1024 # 4GB (increased from 2GB for even more aggressive batching)

min_audio_duration_seconds = 1.0
status_update_seconds = 20.0

lang_detect_batch_timeout_seconds = 0.1
lang_detect_batch_max_size = 8
lang_detect_batch_max_len = 4
lang_detect_batch_ready = 0.1

transcribe_batch_timeout_seconds = 0.1
transcribe_batch_max_size = 8
transcribe_batch_max_len = 4

queue_max_size = 200
die_after_no_measurements_time = 1000000000000000000000000

def get_version():
    """Get the most recent git tag"""
    result = subprocess.run(['git', 'describe', '--tags', '--abbrev=0'], 
                            capture_output=True, text=True, check=True)
    return result.stdout.strip()


version = get_version()