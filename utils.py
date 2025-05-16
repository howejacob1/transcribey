import os
import sys
from contextlib import contextmanager

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
def suppress_output():
    """Suppress all stdout and stderr, including output from C extensions."""
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

