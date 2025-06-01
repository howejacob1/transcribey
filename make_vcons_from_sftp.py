from pymongo import ASCENDING
import sys
import time
import paramiko
import os
import settings
from utils import parse_sftp_url, get_all_filenames_from_sftp
from mongo_utils import get_mongo_collection, get_vcons_cache_collection, delete_all_vcons, all_vcon_urls
from vcon_utils import create_vcon_for_wav
from wavs import is_wav_filename
from settings import sftp_url, dest_dir

def main():
    # Build SFTP URL from settings.py

    # Connect using paramiko and public key authentication
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        sftp_url_parsed = parse_sftp_url(sftp_url)
        username = sftp_url_parsed["username"]
        hostname = sftp_url_parsed["hostname"]
        port = sftp_url_parsed["port"]
        path = sftp_url_parsed["path"]

        client.connect(hostname, port=port, username=username)
        sftp = client.open_sftp()
        collection = get_mongo_collection()
        
        start_time = time.time()
        vcons_cache_collection = get_vcons_cache_collection()
        known_filenames = set(all_vcon_urls(collection))
        
        processed = 0
        total_size = 0  # in bytes
        last_update = time.time()
        last_processed = 0
        last_size = 0
        update_interval = 0.5  # seconds
        rate = 0.0
        for filename in get_all_filenames_from_sftp(sftp, path):
            if is_wav_filename(filename):
                url = f"sftp://{username}@{hostname}:{port}{filename}"
                if url not in known_filenames:
                    # Get file size in bytes
                    try:
                        file_size = sftp.stat(filename).st_size
                    except Exception as e:
                        print(f"\nFailed to get size for {filename}: {e}")
                        file_size = 0
                    total_size += file_size
                    vcon_doc = create_vcon_for_wav(url, sftp)
                    collection.insert_one(vcon_doc)
                    processed += 1
                    now = time.time()
                    if now - last_update >= update_interval:
                        rate = (processed - last_processed) / (now - last_update)
                        size_gb = total_size / (1024 ** 3)
                        mb_delta = (total_size - last_size) / (1024 ** 2)
                        time_delta = now - last_update
                        mb_per_sec = mb_delta / time_delta if time_delta > 0 else 0.0
                        sys.stdout.write(f"\rProcessing: {rate:.2f} files/sec (total: {processed}), total size: {size_gb:.4f} GB, {mb_per_sec:.4f} MB/s")
                        sys.stdout.flush()
                        last_update = now
                        last_processed = processed
                        last_size = total_size
        # Print final average rate and total
        total_time = time.time() - start_time
        avg_rate = processed / total_time if total_time > 0 else 0.0
        size_gb = total_size / (1024 ** 3)
        avg_mb_per_sec = (total_size / (1024 ** 2)) / total_time if total_time > 0 else 0.0
        print(f"\nDone. Average rate: {avg_rate:.2f} files/sec, total processed: {processed}, total size: {size_gb:.4f} GB, average speed: {avg_mb_per_sec:.4f} MB/s")

        sftp.close()
        client.close()
    except Exception as e:
        print(f"Failed to connect or list directory: {e}")

if __name__ == "__main__":
    if settings.debug:
        delete_all_vcons()
    main()