import os
import shutil
import threading
import time
from utils import wav_file_generator


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
        copied = set()
        for root, _, files in os.walk(dest_dir):
            for f in files:
                if f.endswith('.wav'):
                    # Store relative path from dest_dir
                    rel_path = os.path.relpath(os.path.join(root, f), dest_dir)
                    copied.add(rel_path)
        wav_gen = wav_file_generator(source_dir)
        while True:
            # Check if we've reached the size limit
            total_size = get_total_size(dest_dir)
            if total_size >= size_limit_bytes:
                print(f"[wav_preload] Buffer full: {total_size/(1024*1024):.2f} MB")
                break
            # Use wav_file_generator to get .wav files one at a time
            to_copy = []
            try:
                while True:
                    src = next(wav_gen)
                    rel_path = os.path.relpath(src, source_dir)
                    if rel_path not in copied:
                        to_copy.append((src, rel_path))
                    # Only collect a batch per loop
                    if len(to_copy) >= 10:
                        break
            except StopIteration:
                pass
            if not to_copy:
                print("[wav_preload] No more wavs to copy. Exiting thread.")
                break
            for src, rel_path in to_copy:
                dest = os.path.join(dest_dir, rel_path)
                temp_dest = dest + ".tmp"
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                try:
                    # Copy to temp file first
                    with open(src, 'rb') as fsrc, open(temp_dest, 'wb') as fdst:
                        shutil.copyfileobj(fsrc, fdst)
                    # Rename to final name (atomic)
                    os.rename(temp_dest, dest)
                    copied.add(rel_path)
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
