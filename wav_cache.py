import os
import shutil
import threading
import time

def clear_wav_cache():
    """
    Remove all files and directories under working_memory.
    """
    logging.info("Starting to clear working_memory cache...")
    start_time = time.time()
    cache_dir = 'working_memory'
    for entry in os.listdir(cache_dir):
        path = os.path.join(cache_dir, entry)
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
    elapsed = time.time() - start_time
    logging.info(f"Finished clearing working_memory cache in {elapsed:.2f} seconds.")
