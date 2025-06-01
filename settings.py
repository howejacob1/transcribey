# Settings for main.py

source_dir = "fake_wavs/"
dest_dir = "working_memory/raw_wavs_cache/"
wavs_in_progress_dir = "working_memory/wavs_in_progress_cache/"
non_en_wavs_in_progress_dir = "working_memory/non_en_wavs_in_progress_cache/"
lang_detect_threshold = 0.2
lang_detect_batch_size = 64
en_transcription_batch_size = 64+32
non_en_transcription_batch_size = 20
debug = True
total_vcon_filesize_to_process_gb = 1

# SFTP connection settings for make_vcons_from_sftp.py
sftp_url = "sftp://bantaim@192.168.1.100:/home/bantaim/conserver/fake_wavs/"