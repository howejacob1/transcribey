import argparse
import logging
import threading
import time
from queue import Queue
import torch
import cache
import mongo_utils as mongo
import reserver
import secrets_utils
import settings
import vcon_utils as vcon
from vcon_utils import apply_vad_many, batch_to_audio_data, convert_to_mono_many, load_processing_into_ram, make_batches, mark_vcons_as_done, move_to_gpu_many, process_invalids, resample_many, start_discover, update_vcons_on_db
from ai import identify_languages, load_nvidia, load_whisper_tiny, transcribe_many
from cache import move_downloading_to_processing
from gpu import move_to_gpu_maybe
from sftp import connect_keep_trying
from settings import en_model_name, hostname, lang_id_model_name, max_download_threads, non_en_model_name
from log_utils import info_header, with_timing

def main(sftp_url):
    info_header("Initializing cache.")
    cache.init()
    with with_timing("Clearing cache."):
        cache.clear()

    with with_timing("Unmarking all reserved."):
        vcon.unmarked_all_reserved()

    vcons_ready_queue = Queue(maxsize=1)
    info_header(f"Starting reserver thread for {sftp_url}")
    reserver_thread = reserver.start(sftp_url, vcons_ready_queue)

    lang_detect_model = None
    lang_detect_processor = None
    with with_timing("Loading whisper."):
        lang_detect_model, lang_detect_processor = load_whisper_tiny()
        
    en_model = None
    with with_timing("Loading en model."):
        en_model = load_nvidia(en_model_name)

    non_en_model = None
    with with_timing("Loading non-en model."):
        non_en_model = load_nvidia(non_en_model_name)
    
    info_header("Starting main loop.")
    while True:
        vcons = None
        with with_timing("Waiting for vcons."):
            vcons = vcons_ready_queue.get()
        logging.info(f"Got {len(vcons)} vcons.")

        with with_timing("Moving downloading to processing."):
            move_downloading_to_processing()
        
        vcons_len = len(vcons)
        valid_vcons = None
        with with_timing("Processing invalids."):
            valid_vcons = process_invalids(vcons)
        logging.info(f"Eliminated {vcons_len - len(valid_vcons)} invalid vcons")

        vcons_in_ram = None
        with with_timing("Loading valid vcons into RAM."):
            vcons_in_ram = load_processing_into_ram(valid_vcons)

        vcons_mono = None
        with with_timing("Converting to mono."):
            vcons_mono = convert_to_mono_many(vcons_in_ram)

        vcons_resampled = None
        with with_timing("Resampling."):
            vcons_resampled = resample_many(vcons_mono)

        vcons_vad = None
        with with_timing("Applying VAD."):
            vcons_vad = apply_vad_many(vcons_resampled)

        vcons_on_gpu = None
        with with_timing("Moving to GPU."):
            vcons_on_gpu = move_to_gpu_many(vcons_vad)

        vcons_batched = None
        with with_timing("Batching."):
            vcons_batched = make_batches(vcons_on_gpu)

        vcons_detected = None
        with with_timing("Identifying languages."):
            vcons_detected = identify_languages(vcons_batched, lang_detect_model, lang_detect_processor)

        with with_timing("Splitting by language."):
            vcons_en, vcons_non_en = vcon.split_by_language(vcons_detected)

        info_header(f"Batching {len(vcons_en)} en vcons.")
        vcons_en_batched = make_batches(vcons_en)

        info_header(f"Batching {len(vcons_non_en)} non-en vcons.")
        vcons_non_en_batched = make_batches(vcons_non_en)
        
        vcons_en_transcribed = None
        with with_timing("Transcribing en vcons."):
            vcons_en_transcribed = transcribe_many(vcons_en_batched, en_model)
        
        vcons_non_en_transcribed = None
        with with_timing("Transcribing non-en vcons."):
            vcons_non_en_transcribed = transcribe_many(vcons_non_en_batched, non_en_model)

        with with_timing("Marking as done."):
            for vcons in [vcons_en_transcribed, vcons_non_en_transcribed]:
                mark_vcons_as_done(vcons)
        
        with with_timing("Updating on DB."):
            update_vcons_on_db(vcons_en_transcribed)
            update_vcons_on_db(vcons_non_en_transcribed)

        with with_timing("Clearing processing."):
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
