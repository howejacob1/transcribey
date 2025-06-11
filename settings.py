import os
import socket

# Settings for main.py
cache_dir = "cache/"
processing_dir = os.path.join(cache_dir, "processing/")
downloading_dir = os.path.join(cache_dir, "downloading/")
lang_detect_threshold = 0.2
debug = True
cache_size_bytes = 1 * (1024**3)
hostname = socket.gethostname()
#transcribe_english_model_name = "nvidia/parakeet-tdt-0.6b-v2"
en_model_name = "nvidia/parakeet-tdt_ctc-110m"
non_en_model_name = "nvidia/canary-1b-flash"
lang_id_model_name = "openai/whisper-tiny"
gpu_ram_unusable = 3*(1024**3) # 5GB
max_download_threads = 1
gc_limit_bytes = 3*(1024**3)
vcon_queue_max_bytes = 100*(1024**2)

# SFTP connection settings for make_vcons_from_sftp.py
#sftp_url = "sftp://bantaim@127.0.0.1:22/home/bantaim/conserver/openslr-12/"
sftp_url = "sftp://bantaim@127.0.0.1:22/home/bantaim/conserver/fake_wavs_cute/"
sample_rate = 16000
max_discover_workers = 1

# MongoDB connection settings
