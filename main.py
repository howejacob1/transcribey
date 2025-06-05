import cache
import gc
from queue import Queue
import logging
from cache import cache_size_bytes, processing_size_bytes, downloading_size_bytes, filename_to_processing_filename, filename_to_cache_filename, filename_to_downloading_filename, move_filename_to_processing, move_downloading_to_processing, cache_size_bytes, processing_size_bytes, downloading_size_bytes, filename_to_processing_filename, filename_to_cache_filename, filename_to_downloading_filename, move_filename_to_processing, move_downloading_to_processing, clear_processing, clear_downloading, 
import secrets_utils
from ai import load_whisper_tiny, load_nvidia, whisper_token_ids, whisper_start_transcription_token_id, whisper_tokens, identify_languages, transcribe_many
import mongo_utils as mongo
import vcon_utils as vcon
from vcon_utils import process_invalids, load_processing_into_ram, convert_to_mono_many, resample_many, apply_vad_many, move_to_gpu_many, make_batches, batch_to_audio_data, set_languages, set_transcript, transcript, languages, start_discover, mark_vcons_as_done, update_vcons_on_db
import settings
from settings import hostname, max_download_threads, en_model_name, non_en_model_name, lang_id_model_name
from gpu import move_to_gpu_maybe
from utils import batch_bytes, reset_gpu_memory_stats, max_gpu_memory_usage, get_all_filenames_from_sftp, seconds_to_ydhms, ensure_dir_exists, dir_size_bytes, get_device, put_on_gpu, extension
from audio import is_wav_filename, clear_cache_directory, is_valid_audio, get_wav_duration, slurp_audio, resample_audio, apply_vad, move_audio_to_gpu_maybe, load_audio_onto_cpu, audio_dict_to_batches, convert_to_mono
import os
import threading
import time
from sftp import download_sftp_file, sftp_connect, get_sftp_file_size, parse_sftp_url, connect_keep_trying, connect
import argparse
import torch

def reserver_thread(sftp_url, vcons_ready_queue):
    sftp = None
    while True:
        if sftp is None:
            sftp = connect_keep_trying(sftp_url)
        bytes_to_reserve = cache.bytes_to_reserve()
        if bytes_to_reserve > 0:
            try:
                vcons = vcon.find_and_reserve_many(bytes_to_reserve)
                vcon.cache_vcon_audio_many(vcons, sftp)
                vcons_ready_queue.put(vcons)
            except Exception as e:
                logging.info(f"Failed to connect in reserver_thread: {e}")
                sftp = None
        time.sleep(1)

def start_reserver_thread(sftp_url, vcons_ready_queue):
    thread = threading.Thread(target=reserver_thread, args=(sftp_url, vcons_ready_queue), daemon=True)
    thread.start()
    return thread

def main(sftp_url):
    cache.init()
    cache.clear()
    vcon.unmarked_all_reserved()
    vcons_ready_queue = Queue(maxsize=1)
    reserver_thread = start_reserver_thread(sftp_url, vcons_ready_queue)
    lang_detect_model, lang_detect_processor = load_whisper_tiny()
    en_model = load_nvidia(en_model_name)
    non_en_model = load_nvidia(non_en_model_name)

    while True:
        vcons = vcons_ready_queue.get()
        move_downloading_to_processing()
        valid_vcons = process_invalids(vcons)
        vcons_in_ram = load_processing_into_ram(valid_vcons)
        vcons_mono = convert_to_mono_many(vcons_in_ram)
        vcons_resampled = resample_many(vcons_mono)
        vcons_vad = apply_vad_many(vcons_resampled)
        vcons_on_gpu = move_to_gpu_many(vcons_vad)
        vcons_batched = make_batches(vcons_on_gpu)
        vcons_detected = identify_languages(vcons_batched, lang_detect_model, lang_detect_processor)
        vcons_en, vcons_non_en = vcon.split_by_language(vcons_detected)
        vcons_en_batched = make_batches(vcons_en)
        vcons_non_en_batched = make_batches(vcons_non_en)
        vcons_en_transcribed = transcribe_many(vcons_en_batched, en_model)
        vcons_non_en_transcribed = transcribe_many(vcons_non_en_batched, non_en_model)
        for vcons in [vcons_en_transcribed, vcons_non_en_transcribed]:
            mark_vcons_as_done(vcons)
            update_vcons_on_db(vcons)
        cache.clear_processing()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribey main entry point")
    parser.add_argument("mode", choices=["head", "worker", "slurp", "print", "delete_all"], help="head:slurp and run worker. ")
    parser.add_argument("--url", type=str, default=settings.sftp_url, help="Override SFTP URL (applies to both head and worker)")
    parser.add_argument("--production", action="store_true", default=False, help="Enable production mode (applies to both head and worker)")
    args = parser.parse_args()

    settings.debug = not args.production

    if args.mode == "head":
        if settings.debug:
            vcon.delete_all()
        start_discover(args.url)
        main(args.url)
    elif args.mode == "worker":
        main(args.url)
    elif args.mode == "slurp":
        if settings.debug:
            vcon.delete_all()
        start_discover(args.url)
    elif args.mode == "print":
        vcon.load_and_print_all()
    elif args.mode == "delete_all":
        vcon.delete_all()

# architecture
# We can fit all AI models in GPU memory. 
# So, what if we do the following:
# In background, always reserve vcons until cache is full, download said vcons. 

# now repeat. 
# Move all wavs in cache to processing.
# move all wavs to RAM.
# Apply VAD.
# Move all wavs to GPU.
# Resample to 16000kHz.
# Identify languages 
# transcribe english
# transcribe non-english
# Start a thread to update vcons. 
# (If model size is a problem, save to disk, handle later.)
