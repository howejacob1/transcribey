import time
from utils import is_audio_filename
import os
import sftp
import threading
import uuid
import tarfile
import subprocess

ip = "45.55.123.39"
user = "root"

def to_local_filename(foreign_filename):
    parts = foreign_filename.split("/")
    return "/media/10900-hdd-0/" + "/".join(parts[3:])

files_to_download_filename = "files_to_download.txt"
all_files_filename = "all_files_backup.txt"
sync_interval = 60

def update_files_to_download():
    sftp_client, ssh_client = sftp.connect_keep_trying(f"sftp://{user}@{ip}/mnt/")
    total_files_scanned = 0
    total_bytes_scanned = 0
    total_bytes_to_download = 0
    total_files_to_download = 0
    # last_sync_time = time.time()    
    try:
        all_files_generator = sftp.get_all_filenames("/mnt/", sftp_client)
        with open(files_to_download_filename, "w") as files_to_download_file:
            with open(all_files_filename, "w") as all_files_file:
                for foreign_filename, foreign_bytes in all_files_generator:
                    if is_audio_filename(foreign_filename):
                        all_files_file.write(f"{foreign_filename}\n")
                        total_bytes_scanned += foreign_bytes
                        total_files_scanned += 1
                        should_download = True
                        local_filename = to_local_filename(foreign_filename)
                        if os.path.exists(local_filename):
                            local_size = os.path.getsize(local_filename)
                            if local_size == foreign_bytes:
                                should_download = False
                        if should_download:
                            total_bytes_to_download += foreign_bytes
                            total_files_to_download += 1
                            files_to_download_file.write(f"{foreign_filename}\n")
                    total_gb_scanned = total_bytes_scanned / (1024 * 1024 * 1024)
                    total_gb_to_download = total_bytes_to_download / (1024 * 1024 * 1024)
                    print(f"{total_gb_scanned:.2f}GB scanned total {total_files_scanned} files, {total_gb_to_download:.2f}GB to download total {total_files_to_download} files")
                    # if time.time() - last_sync_time > sync_interval:
                    #     last_sync_time = time.time()
                    #     print(f"Syncing {files_to_download_filename} and {all_files_filename}")
                    #     files_to_download_file.flush()
                    #     all_files_file.flush()
    finally:
        if sftp_client:
            sftp_client.close()
        if ssh_client:
            ssh_client.close()

def download_and_extract(remote_tar_path, local_tar_path, ssh_client, semaphore):
    thread_sftp = None
    try:
        print(f"Downloading {remote_tar_path} to {local_tar_path}")
        
        # Use rsync instead of SCP for faster download with progress reporting
        rsync_command = [
            "rsync", 
            "--progress",
            "--stats",
            "-e", "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null",
            f"{user}@{ip}:{remote_tar_path}",
            local_tar_path
        ]
        
        # Run rsync without capturing output so progress is shown in real-time
        result = subprocess.run(rsync_command)
        if result.returncode != 0:
            print(f"Rsync download failed for {remote_tar_path} with exit code {result.returncode}")
            return
            
        print(f"Finished downloading {remote_tar_path}")

        # Still use SFTP for removing the remote file since we need SSH session for that
        thread_sftp = ssh_client.open_sftp()
        print(f"Removing remote file {remote_tar_path}")
        thread_sftp.remove(remote_tar_path)

        print(f"Extracting {local_tar_path}")
        extract_path = "/media/10900-hdd-0/"
        with tarfile.open(local_tar_path, "r") as tar:
            for member in tar.getmembers():
                print(f"Extracting: {member.name}")
                tar.extract(member, path=extract_path)
        print(f"Finished extracting {local_tar_path}")

        print(f"Removing local file {local_tar_path}")
        os.remove(local_tar_path)

    except Exception as e:
        print(f"Error in thread for {remote_tar_path}: {e}")
    finally:
        if thread_sftp:
            thread_sftp.close()
        semaphore.release()

def process_batch(file_batch, ssh_client, threads, semaphore):
    tar_name = f"batch_{uuid.uuid4()}.tar"
    remote_tar_path = f"/root/{tar_name}"
    local_tar_path = f"/home/bantaim/tars/{tar_name}"

    # relative_files = [f.replace('/mnt/freeswitch/', '', 1) for f in file_batch if '/mnt/freeswitch/' in f]

    # if not relative_files:
    #     print("Warning: batch contains no files in /mnt/freeswitch/, skipping.")
    #     return

    command = f"tar -cf {remote_tar_path} --files-from=-"
    print(f"Creating remote tar: {remote_tar_path} with {len(file_batch)} files")

    stdin, stdout, stderr = ssh_client.exec_command(command)
    
    stdin.write('\n'.join(file_batch))
    stdin.channel.shutdown_write()

    exit_status = stdout.channel.recv_exit_status()
    if exit_status != 0:
        error_output = stderr.read().decode()
        print(f"Error creating tar file on remote server. Exit status: {exit_status}")
        print(f"Stderr: {error_output}")
        # Cleanup failed remote tar attempt if it exists
        try:
            sftp = ssh_client.open_sftp()
            sftp.remove(remote_tar_path)
            sftp.close()
        except Exception as e:
            print(f"Could not remove failed tar {remote_tar_path}: {e}")
        return

    print(f"Remote tar created: {remote_tar_path}")
    semaphore.acquire()
    thread = threading.Thread(target=download_and_extract, args=(remote_tar_path, local_tar_path, ssh_client, semaphore))
    threads.append(thread)
    thread.start()

def download_files():
    ssh_client = None
    try:
        _, ssh_client = sftp.connect_keep_trying(f"sftp://{user}@{ip}/mnt/")

        threads = []
        file_batch = []
        batch_size = 10000
        semaphore = threading.Semaphore(4)

        with open(files_to_download_filename, "r") as f:
            for line in f:
                file_batch.append(line.strip())
                if len(file_batch) >= batch_size:
                    process_batch(file_batch, ssh_client, threads, semaphore)
                    file_batch = []
        
        if file_batch:
            process_batch(file_batch, ssh_client, threads, semaphore)

        print("All batches scheduled, waiting for downloads to complete...")
        for thread in threads:
            thread.join()
        print("All downloads finished!")

    finally:
        if ssh_client:
            ssh_client.close()

def main():
    # update_files_to_download()
    download_files()

if __name__ == "__main__":
    main()