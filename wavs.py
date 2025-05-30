import os
import torchaudio
import logging

def get_wav_duration(wav_path):
    """
    Return the duration in seconds of a wav file using torchaudio.info.
    Returns None if the file is corrupt or unreadable.
    """
    try:
        info = torchaudio.info(wav_path)
        return info.num_frames / info.sample_rate
    except Exception as e:
        logging.warning(f"Skipping corrupt or unreadable wav file: {wav_path} ({e})")
        return None

def is_readable_wav(file_path):
    duration = get_wav_duration(file_path)
    return duration is not None

def get_valid_wav_files(directory):
    """
    Walk through a directory and return a dict mapping relative paths to absolute paths of valid wav files only.
    """
    valid_wav_files = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.wav'):
                abs_path = os.path.join(root, file)
                if is_readable_wav(abs_path):
                    rel_path = os.path.relpath(abs_path, directory)
                    valid_wav_files[rel_path] = abs_path
    return valid_wav_files

def get_total_wav_size(wav_files):
    total = 0
    for file_path in wav_files:
        total += os.path.getsize(file_path)
    return total

def get_total_wav_duration(wav_files):
    total = 0
    for file_path in wav_files:
        duration = get_wav_duration(file_path)
        if duration is not None:
            total += duration
    return total
