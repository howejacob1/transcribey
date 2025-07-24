import os
import subprocess
import time
from pathlib import Path
import concurrent.futures

IP = "45.55.123.39"
USER = "root"
REMOTE_PREFIX = "/mnt/"
LOCAL_BASE = "/media/10900-hdd-0/"

LIST_PATH = "files_to_download_2.txt"


def compute_local_path(remote_path: str) -> str:
    """Map remote /mnt/<volume>/rest -> /media/10900-hdd-0/rest"""
    if not remote_path.startswith(REMOTE_PREFIX):
        raise ValueError(f"Remote path does not start with {REMOTE_PREFIX}: {remote_path}")
    # Strip '/mnt/'
    rel = remote_path[len(REMOTE_PREFIX):]
    # Remove first path segment (the volume name)
    parts = rel.split(os.sep, 1)
    if len(parts) == 1:
        # No volume segment, just use as is
        rest = parts[0]
    else:
        rest = parts[1]
    return os.path.join(LOCAL_BASE, rest)


def rsync_file(remote_path: str, local_path: str):
    local_dir = os.path.dirname(local_path)
    os.makedirs(local_dir, exist_ok=True)
    remote = f"{USER}@{IP}:{remote_path}"
    subprocess.run(["rsync", "-av", "--whole-file", remote, local_path], check=True)


def copy_one(remote_path: str):
    remote_path = remote_path.strip()
    if not remote_path:
        return
    local_path = compute_local_path(remote_path)
    for attempt in range(11):  # initial try + up to 10 retries
        try:
            rsync_file(remote_path, local_path)
            print(f"copied {remote_path} -> {local_path}")
            return
        except subprocess.CalledProcessError as e:
            if attempt < 10:
                print(f"retry {attempt + 1}/10 for {remote_path}")
                time.sleep(1)
            else:
                print(f"failed to copy {remote_path} after 11 attempts: {e}")
                return


def main():
    if not os.path.exists(LIST_PATH):
        print(f"{LIST_PATH} not found")
        return
    with open(LIST_PATH, "r") as f:
        paths = [line.strip() for line in f if line.strip()]

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(copy_one, paths)


if __name__ == "__main__":
    main() 