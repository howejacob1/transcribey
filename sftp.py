import os
import logging
import time
from urllib.parse import urlparse
import paramiko
import settings
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED

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
            time.sleep(5)

def file_size(filename, sftp):
    return sftp.stat(filename).st_size

def download(url, local_path, sftp):
    parsed = parse_url(url)
    return sftp.get(parsed["path"], local_path)

def is_dir(path):
    return path.st_mode & 0o040000

# filthy AI functions
def _get_all_filenames_worker(root, transport):
    sftp = paramiko.SFTPClient.from_transport(transport)
    try:
        all_entries = sftp.listdir_attr(root)
        files = []
        dirs = []
        for entry in all_entries:
            path = entry.filename
            base = os.path.basename(path)
            filename = os.path.join(root, base)
            logging.info(f"Discovered {filename}")
            if is_dir(entry):
                dirs.append(filename)
            else:
                files.append(filename)
        return files, dirs
    finally:
        sftp.close()

# filthy AI functions
def get_all_filenames(root, sftp, max_workers=4):
    logging.info(f"Getting all filenames from {root} using {max_workers} workers")
    transport = sftp.sock.get_transport()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_get_all_filenames_worker, root, transport)}
        while futures:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                try:
                    files, dirs = future.result()
                    for f in files:
                        yield f
                    for d in dirs:
                        futures.add(executor.submit(_get_all_filenames_worker, d, transport))
                except Exception as exc:
                    logging.error(f"A worker generated an exception: {exc}")