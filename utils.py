import os
import sys
import logging
import time
from contextlib import contextmanager
import torchaudio  # Removed to avoid ModuleNotFoundError

def get_all_filenames(directory):
    """
    Recursively read all files in a directory and return a dict mapping relative paths to absolute paths.
    """
    file_dict = {}
    for root, _, files in os.walk(directory):
        for file in files:
            abs_path = os.path.abspath(os.path.join(root, file))
            rel_path = os.path.relpath(abs_path, directory)
            file_dict[rel_path] = abs_path
    return file_dict

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

@contextmanager
def suppress_output(should_suppress=True):
    """Suppress all stdout and stderr, including output from C extensions."""
    if should_suppress:
        with open(os.devnull, 'w') as devnull:
            old_stdout_fd = os.dup(1)
            old_stderr_fd = os.dup(2)
            try:
                os.dup2(devnull.fileno(), 1)
                os.dup2(devnull.fileno(), 2)
                yield
            finally:
                os.dup2(old_stdout_fd, 1)
                os.dup2(old_stderr_fd, 2)
                os.close(old_stdout_fd)
                os.close(old_stderr_fd)
    else:
        yield