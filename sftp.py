import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from urllib.parse import urlparse

import paramiko

import settings

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
    logging.info(f"Connecting to {username}@{hostname}:{port}")
    client.connect(hostname, port=port, username=username, timeout=5)
    logging.info(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Done, connected to {username}@{hostname}:{port}")

def connect(url):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    url_parsed = parse_url(url)
    
    connect_raw(url_parsed["hostname"], url_parsed["port"], url_parsed["username"], client)
    
    # Performance optimizations after connection
    transport = client.get_transport()
    transport.set_keepalive(30)
    transport.use_compression(True)
    
    sftp = client.open_sftp()
    
    # SFTP performance optimizations
    sftp.get_channel().settimeout(settings.sftp_download_timeout)
    
    return sftp

def connect_keep_trying(url):
    while True:
        try:
            sftp = connect(url)
            return sftp
        except Exception as e:
            time.sleep(1)

def file_size(filename, sftp):
    return sftp.stat(filename).st_size

def download(remote_path, local_path, sftp):
    return sftp.get(remote_path, local_path)

def download_optimized(remote_path, local_path, sftp, buffer_size=None, prefetch=None):
    """Optimized download with better buffering and prefetch"""
    if buffer_size is None:
        buffer_size = settings.sftp_buffer_size
    if prefetch is None:
        prefetch = settings.sftp_prefetch_enabled
        
    # Check if SFTP connection is still alive
    try:
        channel = sftp.get_channel()
        if not channel or not channel.get_transport().is_active():
            raise ConnectionError("SFTP connection is not active")
    except AttributeError:
        raise ConnectionError("SFTP connection is invalid")
        
    if prefetch:
        # Enable prefetch for better performance
        sftp.get(remote_path, local_path, callback=None, prefetch=True)
    else:
        # Manual buffered download for more control
        with sftp.open(remote_path, 'rb', bufsize=buffer_size) as remote_file:
            with open(local_path, 'wb') as local_file:
                while True:
                    data = remote_file.read(buffer_size)
                    if not data:
                        break
                    local_file.write(data)

def is_dir(path):
    return path.st_mode & 0o040000

# filthy AI functions
def _get_all_filenames_worker(root, transport):
    sftp = None
    while True:
        try:
            sftp = paramiko.SFTPClient.from_transport(transport)
            break
        except Exception as e:
            continue
    try:
        all_entries = sftp.listdir_attr(root)
        files = []
        dirs = []
        sizes = []
        for entry in all_entries:
            path = entry.filename
            base = os.path.basename(path)
            filename = os.path.join(root, base)
            logging.debug(f"Discovered {filename}")
            if is_dir(entry):
                dirs.append(filename)
            else:
                files.append(filename)
                sizes.append(entry.st_size)
        return files, dirs, sizes
    finally:
        sftp.close()

# filthy AI functions
def get_all_filenames_threaded(root, sftp, max_workers=settings.max_discover_workers):
    logging.info(f"Getting all filenames from {root} using {max_workers} workers")
    transport = sftp.sock.get_transport()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_get_all_filenames_worker, root, transport)}
        while futures:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                try:
                    files, dirs, sizes = future.result()
                    for filename, size in zip(files, sizes):
                        yield filename, size
                    for d in dirs:
                        futures.add(executor.submit(_get_all_filenames_worker, d, transport))
                except Exception as exc:
                    logging.error(f"A worker generated an exception: {exc}")

def get_all_filenames(root, sftp):
    queue = [root]
    while queue:
        current_dir = queue.pop(0)
        if "old-thing" in current_dir:
            continue
        try:
            entries = sftp.listdir_attr(current_dir)
        except Exception as e:
            logging.error(f"Failed to list directory {current_dir}: {e}")
            continue
        for entry in entries:
            filename = os.path.join(current_dir, entry.filename)
            if is_dir(entry):
                queue.append(filename)
            else:
                yield filename, entry.st_size
