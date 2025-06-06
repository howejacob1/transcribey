import os
import logging
import time
from urllib.parse import urlparse
import paramiko
import settings
from concurrent.futures import ThreadPoolExecutor, as_completed

def parse_url(sftp_url):
    parsed = urlparse(sftp_url)
    username_and_hostname_list = parsed.netloc.split("@")
    username = username_and_hostname_list[0]
    hostname = username_and_hostname_list[1].split(":")[0]
    return {"username": username,
            "hostname": hostname,
            "port": parsed.port or 22,
            "path": parsed.path}

def connect_raw(hostname, port, username, client):
    return client.connect(hostname, port=port, username=username)

def connect(url):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    url_parsed = parse_url(url)
    connect_raw(url_parsed["hostname"], url_parsed["port"], url_parsed["username"], client)
    sftp = client.open_sftp()
    return sftp

def connect_keep_trying(url):
    while True:
        try:
            return connect(url)
        except Exception as e:
            time.sleep(1)

def file_size(filename, sftp):
    return sftp.stat(filename).st_size

def download(url, local_path, sftp):
    parsed = parse_url(url)
    return sftp.get(parsed["path"], local_path)

def is_dir(path):
    return path.st_mode & 0o040000

def ls(path, sftp):
    return sftp.listdir_attr(path)

def get_all_filenames_sequential(root, sftp):
    """
    Recursively lists all filenames in a directory on SFTP, non-parallel.
    """
    all_files = []
    dirs_to_process = [root]
    while dirs_to_process:
        current_dir = dirs_to_process.pop(0)
        try:
            for item in ls(current_dir, sftp):
                path = os.path.join(current_dir, item.filename)
                if is_dir(item):
                    dirs_to_process.append(path)
                else:
                    all_files.append(path)
        except Exception as e:
            logging.error(f"Error listing {current_dir}: {e}")
    return all_files

# filthy AI functions
def _get_all_filenames_worker(root, sftp):
    """
    Worker function to list files in a directory on SFTP.
    This function is intended to be run in a thread pool.
    It's not used in the new implementation but kept for reference.
    """
    try:
        all_paths = ls(root, sftp)
        files = []
        dirs = []
        for path in all_paths:
            base = path.filename
            filename = os.path.join(root, base)
            if is_dir(path):
                dirs.append(filename)
            else:
                files.append(filename)
        return files, dirs
    except Exception as e:
        logging.error(f"Error in worker for dir {root}: {e}")
        return [], []

# filthy AI functions
def get_all_filenames(root, sftp, max_workers=10):
    logging.info(f"Getting all filenames from {root}")
    
    dirs_to_process = [root]
    
    while dirs_to_process:
        current_dir = dirs_to_process.pop(0)
        logging.info(f"Processing directory: {current_dir}")
        try:
            items = ls(current_dir, sftp)
            for item in items:
                path = os.path.join(current_dir, item.filename)
                if is_dir(item):
                    dirs_to_process.append(path)
                else:
                    yield path
        except Exception as e:
            logging.error(f"Could not list directory {current_dir}: {e}")