# Settings for main.py

dest_dir = "working_memory/raw_wavs_cache/"
lang_detect_threshold = 0.2
lang_detect_batch_size = 64
en_transcription_batch_size = 64+32
non_en_transcription_batch_size = 20
debug = True
total_vcon_filesize_to_process_gb = 0.1
default_batch_bytes = 2* 1024 * 1024 * 1024 # 100MB

transcribe_english_model_name = "nvidia/parakeet-tdt-0.6b-v2"
transcribe_nonenglish_model_name = "nvidia/canary-1b-flash"
identify_languages_model_name = "openai/whisper-tiny"

# SFTP connection settings for make_vcons_from_sftp.py
sftp_url = "sftp://bantaim@192.168.1.100:/home/bantaim/conserver/fake_wavs/"