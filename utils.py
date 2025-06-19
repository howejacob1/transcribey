import multiprocessing
import os
import socket
import sys
import threading
import time
import traceback
from contextlib import contextmanager

from settings import dont_overwhelm_server_time_seconds

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
    return seconds_to_days(seconds) / 7

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
    return os.path.splitext(filename)[1][1:]

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

def is_audio_filename(filename):
    ext = extension(filename)
    return ext in ["wav", "mp3", "ogg", "m4a", "flac", "aac", "wma", "aiff", "au", "raw", "pcm"]

def dump_thread_stacks():
    """Dumps the stack trace of all running threads to the console."""
    print(f"\n--- Thread Stack Dump ---")
    
    # A map from thread ID to thread name
    thread_names = {th.ident: th.name for th in threading.enumerate()}
    
    for thread_id, frame in sys._current_frames().items():
        thread_name = thread_names.get(thread_id, f"Thread-{thread_id}")
        print(f"\n--- Stack for {thread_name} (ID: {thread_id}) ---")
        
        # The 'traceback' module formats the stack frame into a readable string
        stack_trace = "".join(traceback.format_stack(frame))
        print(stack_trace.strip())
        
    print("--- End of Thread Stack Dump ---\n")

def size_of_file(filename):
    return os.path.getsize(filename)

def clear_screen():
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')

def dont_overwhelm_server():
    time.sleep(dont_overwhelm_server_time_seconds)

def let_other_threads_run():
    time.sleep(0)

def save_to_file(filename, data):
    with open(filename, "w") as f:
        f.write(data)

def die():
    pid = os.getpid()
    os.system(f"kill -9 {pid}")

def die_after_delay(delay):
    time.sleep(delay)
    die()