import os
from contextlib import contextmanager
import torch
from urllib.parse import urlparse

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

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def parse_sftp_url(sftp_url):
    parsed = urlparse(sftp_url)
    username_and_hostname_list = parsed.netloc.split("@")
    print(f"username_and_hostname_list is {username_and_hostname_list}")
    username = username_and_hostname_list[0]
    hostname = username_and_hostname_list[1].split(":")[0]
    print(f"username is {username}")
    print(f"hostname is {hostname}")
    print(f"port is {parsed.port}")
    print(f"path is {parsed.path}")
    return {"username": username,
            "hostname": hostname,
            "port": parsed.port or 22,
            "path": parsed.path}

def is_sftp_file_directory(entry):
    return entry.st_mode & 0o040000

def get_all_filenames_from_sftp(sftp_client, path):
    for entry in sftp_client.listdir_attr(path):
        entry_path = f"{path.rstrip('/')}/{entry.filename}"
        if is_sftp_file_directory(entry): 
            yield from get_all_filenames_from_sftp(sftp_client, entry_path)
        else:
            yield entry_path

