import os
import time
import paramiko
from urllib.parse import urlparse

def parse_url(sftp_url):
    parsed = urlparse(sftp_url)
    username_and_hostname_list = parsed.netloc.split("@")
    username = username_and_hostname_list[0]
    hostname = username_and_hostname_list[1].split(":")[0]
    return {"username": username,
            "hostname": hostname,
            "port": parsed.port or 22,
            "path": parsed.path}

def connect_raw(hostname, port, username, ssh):
    return ssh.connect(hostname, port=port, username=username)

def connect(url):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    url_parsed = parse_url(url)
    client = connect_raw(url_parsed["hostname"], 
                         url_parsed["port"], 
                         url_parsed["username"])
    sftp = client.open_sftp()
    return sftp

def connect_keep_trying(url):
    while True:
        try:
            return connect(url)
        except Exception as e:
            time.sleep(1)

def file_size(url, sftp):
    """ 
    Return the size of a file given a local path or SFTP URL.
    If sftp_client is provided, it will be used for SFTP URLs.
    """
    parsed = parse_url(url)
    return sftp.stat(parsed["path"]).st_size

def download_raw(url, local_path, sftp):
    parsed = parse_url(url)
    return sftp.get(parsed["path"], local_path)

def is_dir(path):
    return path.st_mode & 0o040000

def ls(path, sftp):
    return sftp.listdir_attr(path)

def get_all_filenames(root, sftp):
    sftp_paths = ls(root, sftp)
    for sftp_path_cur in sftp_paths:
        base = sftp_path_cur.filename
        filename = os.path.join(root, base)
        if is_dir(sftp_path_cur):
            yield from get_all_filenames(filename, sftp)
        else:
            yield filename