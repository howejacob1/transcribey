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

def get_wav_files(file_dict):
    """
    Filter a file dictionary to only include .wav files.
    Returns a dict mapping relative paths to absolute paths for wav files only.
    """
    return {rel: abs for rel, abs in file_dict.items() if rel.lower().endswith('.wav')}

def wav_file_generator(directory):
    """
    Generator that recursively walks a directory and yields .wav file paths one at a time.
    Retains state between calls.
    """
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith('.wav'):
                yield os.path.join(root, f)

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

def get_total_wav_size(directory):
    total = 0
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith('.wav'):
                total += os.path.getsize(os.path.join(root, f))
    return total