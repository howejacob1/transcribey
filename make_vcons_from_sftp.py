import sys
import paramiko
from utils import parse_sftp_url, get_all_filenames_from_sftp
from mongo_utils import get_mongo_collection
from vcon_utils import create_vcon_for_wav
from wavs import is_wav_filename
from settings import sftp_url

# architecture:
# 1.) scan the directory for wav files. 
# 2.) if the file is not in the database, create a vCon.

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
        all_filenames = get_all_filenames_from_sftp(sftp, path)
        for filename in all_filenames:
            if is_wav_filename(filename):
                # Construct the SFTP URL using settings and the filename
                url = f"sftp://{username}@{hostname}:{port}{path}{filename}"
                print(f"url is {url}")
                # Check if vCon already exists for this file using the top-level 'filename' field
                if collection.count_documents({"filename": url}, limit=1) == 0:
                    print(f"Creating vCon for: {url}")
                    vcon_doc = create_vcon_for_wav(url)
                    collection.insert_one(vcon_doc)
                else:
                    print(f"vCon already exists for: {url}")
        sftp.close()
        client.close()
    except Exception as e:
        print(f"Failed to connect or list directory: {e}")

if __name__ == "__main__":
    main()