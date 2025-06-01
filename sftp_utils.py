import os
import paramiko
from urllib.parse import urlparse


def get_sftp_file_size(path_or_url, sftp_client=None):
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
