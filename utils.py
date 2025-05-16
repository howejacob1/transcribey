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
    """Context manager to suppress stdout and stderr."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

