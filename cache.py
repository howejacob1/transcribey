import os
import settings
from settings import cache_dir, downloading_dir, processing_dir
from utils import dir_size_bytes, ensure_dir_exists, delete_files_in_dir

def cache_size_bytes():
    return dir_size_bytes(cache_dir)

def processing_size_bytes():
    return dir_size_bytes(processing_dir)

def downloading_size_bytes():
    return dir_size_bytes(downloading_dir)

def filename_to_processing_filename(filename):
    return os.path.join(processing_dir, os.path.basename(filename))

def filename_to_cache_filename(filename):
    return os.path.join(cache_dir, os.path.basename(filename))

def filename_to_downloading_filename(filename):
    return os.path.join(downloading_dir, os.path.basename(filename))

def move_filename_to_processing(filename):
    os.rename(filename_to_downloading_filename(filename), 
              filename_to_processing_filename(filename))

def move_downloading_to_processing():
    for filename in os.listdir(downloading_dir):
        move_filename_to_processing(filename)

def clear_processing():
    delete_files_in_dir(processing_dir)

def clear_downloading():
    delete_files_in_dir(downloading_dir)

def clear():
    clear_processing()
    clear_downloading()

def init():
    ensure_dir_exists(cache_dir)
    ensure_dir_exists(processing_dir)

def bytes_to_reserve():
    return settings.cache_size_bytes - cache_size_bytes()