import os
from contextlib import contextmanager
import torch
from urllib.parse import urlparse
import socket

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

def dir_size_bytes(dir):
    total_bytes = 0
    for root, _, files in os.walk(dir):
        for file in files:
            total_bytes += os.path.getsize(os.path.join(root, file))
    return total_bytes

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

def hostname():
    """
    Returns the hostname of the current machine.
    """
    return socket.gethostname()

def get_ipv4_address():
    """
    Returns the primary IPv4 address of the current machine.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't need to be reachable
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip


def seconds_to_minutes(seconds):
    return seconds/60

def seconds_to_hours(seconds):
    return seconds_to_minutes(seconds) / 60

def seconds_to_days(seconds):
    return seconds_to_hours(seconds) / 24

def seconds_to_weeks(seconds):
    return seconds_to_days /7

def seconds_to_months(seconds):
    return seconds_to_weeks(seconds)/4

def seconds_to_ydhms(seconds):
    """
    Convert seconds to a string in the format 'Xy Yd Zh Wm Vs'.
    Only nonzero units are included, and units are: years, days, hours, minutes, seconds.
    """
    seconds = int(seconds)
    years, rem = divmod(seconds, 31536000)  # 365*24*60*60
    days, rem = divmod(rem, 86400)          # 24*60*60
    hours, rem = divmod(rem, 3600)          # 60*60
    minutes, secs = divmod(rem, 60)
    parts = []
    if years:
        parts.append(f"{years}y")
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs or not parts:
        parts.append(f"{secs}s")
    return ' '.join(parts)

def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def extension(filename):
    return os.path.splitext(filename)[1]

def what_directory_are_we_in():
    return os.getcwd()

def slurp(filename):
    with open(filename, "rb") as f:
        return f.read()

def wait_for_one_thread_to_finish(threads):
    threads[0].join()
    threads = threads[1:]
    return threads

def wait_for_all_threads_to_finish(threads):
    for thread in threads:
        thread.join()

def delete_files_in_dir(dir):
    for filename in os.listdir(dir):
        os.remove(os.path.join(dir, filename))

def num_cores():
    return os.cpu_count()