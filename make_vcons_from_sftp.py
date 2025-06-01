from pymongo import ASCENDING
import sys
import time
import paramiko
import os
import settings
from utils import parse_sftp_url, get_all_filenames_from_sftp
from mongo_utils import get_mongo_collection, get_vcons_cache_collection, clear_mongo_collections, all_vcon_urls
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
        last_update = time.time()
        last_processed = 0
        update_interval = 0.5  # seconds
        rate = 0.0
        for filename in get_all_filenames_from_sftp(sftp, path):
            if is_wav_filename(filename):
                url = f"sftp://{username}@{hostname}:{port}{filename}"
                if url not in known_filenames:
                    vcon_doc = create_vcon_for_wav(url, sftp)
                    collection.insert_one(vcon_doc)
                    processed += 1
                    now = time.time()
                    if now - last_update >= update_interval:
                        rate = (processed - last_processed) / (now - last_update)
                        sys.stdout.write(f"\rProcessing: {rate:.2f} files/sec (total: {processed})")
                        sys.stdout.flush()
                        last_update = now
                        last_processed = processed
        # Print final rate and total
        print(f"\nDone. Final rate: {rate:.2f} files/sec, total processed: {processed}")

        sftp.close()
        client.close()
    except Exception as e:
        print(f"Failed to connect or list directory: {e}")

if __name__ == "__main__":
    if settings.debug:
        print("Clearing mongo collections")
        start_time = time.time()
        clear_mongo_collections()
        print(f"Time taken to clear mongo collections: {time.time() - start_time:.2f} seconds")
    main()