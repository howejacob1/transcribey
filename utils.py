import os
from contextlib import contextmanager
import torch
from urllib.parse import urlparse
import paramiko
import socket
import psutil

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
    username = username_and_hostname_list[0]
    hostname = username_and_hostname_list[1].split(":")[0]
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

def get_file_size(path_or_url, sftp_client=None):
    """ 
    Return the size of a file given a local path or SFTP URL.
    If sftp_client is provided, it will be used for SFTP URLs.
    """
    if path_or_url.startswith("sftp://"):
        parsed = urlparse(path_or_url)
        username_and_hostname = parsed.netloc.split("@")
        username = username_and_hostname[0]
        host_and_port = username_and_hostname[1].split(":")
        hostname = host_and_port[0]
        port = int(host_and_port[1]) if len(host_and_port) > 1 else 22
        sftp_path = parsed.path
        close_client = False
        if sftp_client is None:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(hostname, port=port, username=username)
            sftp_client = ssh.open_sftp()
            close_client = True
        try:
            size = sftp_client.stat(sftp_path).st_size
        finally:
            if close_client:
                sftp_client.close()
                ssh.close()
        return size
    else:
        return os.path.getsize(path_or_url)

def get_hostname():
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

def download_sftp_file(sftp_url, local_path, sftp_client=None):
    """
    Download a file from an SFTP URL to a local path. If sftp_client is not provided, creates a new connection.
    """
    from urllib.parse import urlparse
    import paramiko
    parsed = urlparse(sftp_url)
    username_and_hostname = parsed.netloc.split("@")
    username = username_and_hostname[0]
    host_and_port = username_and_hostname[1].split(":")
    hostname = host_and_port[0]
    port = int(host_and_port[1]) if len(host_and_port) > 1 else 22
    sftp_path = parsed.path
    close_client = False
    if sftp_client is None:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname, port=port, username=username)
        sftp_client = ssh.open_sftp()
        close_client = True
    try:
        sftp_client.get(sftp_path, local_path)
    finally:
        if close_client:
            sftp_client.close()
            ssh.close()

def sftp_connect(sftp_url):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    sftp_url_parsed = parse_sftp_url(sftp_url)
    hostname = sftp_url_parsed["hostname"]
    port = sftp_url_parsed["port"]
    username = sftp_url_parsed["username"]
    client.connect(hostname, port=port, username=username)
    sftp = client.open_sftp()
    return sftp

def gpu_ram_bytes():
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        return props.total_memory
    return None

def calculate_batch_bytes():
    gpu_ram_bytes_cur = gpu_ram_bytes()
    if gpu_ram_bytes_cur is None:
        return psutil.virtual_memory().total // 8
    else:
        # Use 1/16 of GPU RAM to be very conservative and prevent OOM errors
        return gpu_ram_bytes_cur // 8

def max_gpu_memory_usage():
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        max_allocated = torch.cuda.max_memory_allocated(current_device)
        return max_allocated
    return None

def print_gpu_memory_usage():
    """Print current GPU memory usage for debugging."""
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(current_device)
        reserved = torch.cuda.memory_reserved(current_device)
        max_allocated = torch.cuda.max_memory_allocated(current_device)
        max_reserved = torch.cuda.max_memory_reserved(current_device)
        total = torch.cuda.get_device_properties(current_device).total_memory
        
        print(f"Peak - Allocated: {max_allocated/(1024**3):.2f} GB, Reserved: {max_reserved/(1024**3):.2f} GB")
    else:
        print("CUDA not available")

def reset_gpu_memory_stats():
    """Reset peak GPU memory statistics for more accurate tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        print("GPU memory peak statistics reset")