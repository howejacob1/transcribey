# Settings for main.py

cache_directory = "working_memory/raw_wavs_cache/"
lang_detect_threshold = 0.2
debug = True
total_vcon_filesize_to_process_bytes = 2 * (1024**3)

#transcribe_english_model_name = "nvidia/parakeet-tdt-0.6b-v2"
transcribe_english_model_name = "nvidia/parakeet-tdt_ctc-110m"
transcribe_nonenglish_model_name = "nvidia/canary-1b-flash"
identify_languages_model_name = "openai/whisper-tiny"

# SFTP connection settings for make_vcons_from_sftp.py
sftp_url = "sftp://bantaim@192.168.1.103:/home/bantaim/conserver/fake_wavs_large/"