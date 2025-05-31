from pymongo import ASCENDING
import sys
import time
import paramiko
import os
from utils import parse_sftp_url, get_all_filenames_from_sftp
from mongo_utils import get_mongo_collection, get_vcons_cache_collection
from vcon_utils import create_vcon_for_wav
from wavs import is_wav_filename
from settings import sftp_url, dest_dir

# architecture:
# 1.) scan the directory for wav files. 
# 2.) if the file is not in the database, create a vCon.
# 3.) Encase the wav into the vcon.

# def cache_wav_or_wait(url, filename, vcons_cache_collection, sftp):
#     """
#     Ensure the wav file is cached in MongoDB, respecting the cache size limit.
#     If not enough space, wait until space is available.
#     """
#     # Check if already cached
#     if is_vcon_cached(url, vcons_cache_collection):
#         return filename

#     # Get the size of the wav file on SFTP
#     try:
#         stat = sftp.stat(filename)
#         wav_size = stat.st_size
#     except Exception as e:
#         print(f"Failed to stat {filename} on SFTP: {e}")
#         return None

#     max_bytes = wav_cache_max_size_gb * 1024 ** 3

#     while True:
#         # Calculate current cache size
#         total_size = 0
#         for doc in vcons_cache_collection.find({}, {"size": 1}):
#             total_size += doc.get("size", 0)

#         if total_size + wav_size <= max_bytes:
#             # Download the wav file to dest_dir
#             os.makedirs(dest_dir, exist_ok=True)
#             local_path = os.path.join(dest_dir, os.path.basename(filename))
#             try:
#                 sftp.get(filename, local_path)
#             except Exception as e:
#                 print(f"Failed to download {filename} from SFTP: {e}")
#                 return None
#             # Insert into cache collection
#             vcons_cache_collection.insert_one({
#                 "filename": url,
#                 "local_path": local_path,
#                 "size": wav_size,
#                 "cached_at": time.time()
#             })
#             return filename
#         else:
#             print(f"Waiting for space to be freed in cache. Current size: {total_size} bytes, max size: {max_bytes} bytes")
#             time.sleep(1)

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
        print(f"Time taken to get all filenames: {time.time() - start_time:.2f} seconds")
        vcons_cache_collection = get_vcons_cache_collection()
        for filename in get_all_filenames_from_sftp(sftp, path):
            if is_wav_filename(filename):
                # Construct the SFTP URL using settings and the filename
                url = f"sftp://{username}@{hostname}:{port}{filename}"
                # Check if vCon already exists for this file using the top-level 'filename' field
                if collection.count_documents({"filename": url}, limit=1) == 0:
                    vcon_doc = create_vcon_for_wav(url)
                    print(f"Creating vcon for {os.path.basename(filename)}")
                    collection.insert_one(vcon_doc)
                
        sftp.close()
        client.close()
    except Exception as e:
        print(f"Failed to connect or list directory: {e}")

if __name__ == "__main__":
    main()