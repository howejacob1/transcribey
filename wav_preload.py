import os
import shutil
import threading
import time


def get_total_size(directory):
    total = 0
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith('.wav'):
                total += os.path.getsize(os.path.join(root, f))
    return total


def preload_wavs_threaded(source_dir, dest_dir, size_limit_bytes=1*1024*1024*1024):
    """
    Copies completed wav files from source_dir to dest_dir in a background thread.
    Stops when dest_dir reaches size_limit_bytes or all source wavs are copied.
    Only completed files are copied (atomic copy: temp then rename).
    """
    def worker():
        os.makedirs(dest_dir, exist_ok=True)
        copied = set(os.listdir(dest_dir))
        while True:
            # Check if we've reached the size limit
            total_size = get_total_size(dest_dir)
            if total_size >= size_limit_bytes:
                print(f"[wav_preload] Buffer full: {total_size/(1024*1024):.2f} MB")
                break
            # Find all .wav files in source_dir
            source_wavs = []
            for root, _, files in os.walk(source_dir):
                for f in files:
                    if f.endswith('.wav'):
                        source_wavs.append(os.path.join(root, f))
            # Only copy files not already in dest_dir
            to_copy = [f for f in source_wavs if os.path.basename(f) not in copied]
            if not to_copy:
                print("[wav_preload] No more wavs to copy. Exiting thread.")
                break
            for src in to_copy:
                dest = os.path.join(dest_dir, os.path.basename(src))
                temp_dest = dest + ".tmp"
                try:
                    # Copy to temp file first
                    with open(src, 'rb') as fsrc, open(temp_dest, 'wb') as fdst:
                        shutil.copyfileobj(fsrc, fdst)
                    # Rename to final name (atomic)
                    os.rename(temp_dest, dest)
                    copied.add(os.path.basename(src))
                    print(f"[wav_preload] Copied: {src} -> {dest}")
                except Exception as e:
                    print(f"[wav_preload] Error copying {src}: {e}")
                # Check size after each copy
                total_size = get_total_size(dest_dir)
                if total_size >= size_limit_bytes:
                    print(f"[wav_preload] Buffer full: {total_size/(1024*1024):.2f} MB")
                    break
            # Sleep briefly to avoid tight loop if not enough files
            time.sleep(1)
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread
