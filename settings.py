import os
import socket

# Settings for main.py
cache_dir = "working_memory/cache/"
processing_dir = os.path.join(cache_dir, "processing/")
downloading_dir = os.path.join(cache_dir, "downloading/")
lang_detect_threshold = 0.2
debug = True
max_download_threads = 16
cache_size_bytes = 1 * (1024**3)
hostname = socket.gethostname()
#transcribe_english_model_name = "nvidia/parakeet-tdt-0.6b-v2"
en_model_name = "nvidia/parakeet-tdt_ctc-110m"
non_en_model_name = "nvidia/canary-1b-flash"
lang_id_model_name = "openai/whisper-tiny"
gpu_ram_unusable = 5*1024**3 # 5GB
max_download_threads = 16
gc_limit_bytes = 1024**3

# SFTP connection settings for make_vcons_from_sftp.py
sftp_url = "sftp://bantaim@192.168.1.103:/home/bantaim/conserver/openslr-12/"
sample_rate = 16000
