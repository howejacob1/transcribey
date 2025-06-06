import os
import settings
from utils import dir_size_bytes, ensure_dir_exists, delete_files_in_dir

def cache_size_bytes():
    return dir_size_bytes(settings.cache_dir)

def processing_size_bytes():
    return dir_size_bytes(settings.processing_dir)

def downloading_size_bytes():
    return dir_size_bytes(settings.downloading_dir)

def filename_to_processing_filename(filename):
    return os.path.join(settings.processing_dir, os.path.basename(filename))

def filename_to_cache_filename(filename):
    return os.path.join(settings.cache_dir, os.path.basename(filename))

def filename_to_downloading_filename(filename):
    return os.path.join(settings.downloading_dir, os.path.basename(filename))

def move_filename_to_processing(filename):
    os.rename(filename_to_downloading_filename(filename), 
              filename_to_processing_filename(filename))

def move_downloading_to_processing():
    for filename in os.listdir(settings.downloading_dir):
        move_filename_to_processing(filename)

def clear_processing():
    delete_files_in_dir(settings.processing_dir)

def clear_downloading():
    delete_files_in_dir(settings.downloading_dir)

def clear():
    clear_processing()
    clear_downloading()

def init():
    ensure_dir_exists(settings.cache_dir)
    ensure_dir_exists(settings.processing_dir)
    ensure_dir_exists(settings.downloading_dir)

def bytes_to_reserve():
    return settings.cache_size_bytes - cache_size_bytes()