import argparse
import logging
import threading
import time
from queue import Queue, Empty
import torch
import cache
import mongo_utils as mongo
import reserver
import secrets_utils
import settings
from utils import dump_thread_stacks
import vcon_utils as vcon
import ai
import sftp as sftp_utils
from log_utils import info_header, with_timing
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(threadName)s] %(levelname)s - %(message)s')
logging.getLogger("paramiko").setLevel(logging.INFO)

def main(sftp_url):
    # sftp = sftp_utils.connect_keep_trying(sftp_url)
    info_header("Initializing cache.")

    with with_timing("Unmarking all reserved."):
        vcon.unmarked_all_reserved()

    vcons_ready_queue = Queue(maxsize=1)
    vcons_lock = threading.Lock()

    info_header(f"Starting reserver thread for {sftp_url}")
    reserver_thread = reserver.start(sftp_url, vcons_ready_queue, vcons_lock)
    lang_detect_model = None
    lang_detect_processor = None
    with with_timing("Loading whisper."):
        lang_detect_model, lang_detect_processor = ai.load_whisper_tiny()
        
    en_model = None
    with with_timing("Loading en model."):
        en_model = ai.load_nvidia(settings.en_model_name)

    non_en_model = None
    with with_timing("Loading non-en model."):
        non_en_model = ai.load_nvidia(settings.non_en_model_name)
    
    info_header("Starting main loop.")
    while True:
        vcons = None

        with with_timing("Moving downloading to processing."):
            with vcons_lock:
                vcons = vcons_ready_queue.get()
                cache.move_downloading_to_processing()

        vcons_len = len(vcons)
        valid_vcons = None
        with with_timing("Processing invalids."):
            valid_vcons = vcon.process_invalids(vcons)
        logging.info(f"Eliminated {vcons_len - len(valid_vcons)} invalid vcons")

        print(f"Valid vcons: {len(valid_vcons)}")
        vcons_in_ram = None
        with with_timing("Loading valid vcons into RAM."):
            vcons_in_ram = vcon.load_processing_into_ram(valid_vcons)
            print(f"Vcons in RAM: {len(vcons_in_ram)}")
        print(f"Vcons in RAM: {vcons_in_ram[0]}")
        
        vcons_mono = None
        with with_timing("Converting to mono."):
            vcons_mono = vcon.convert_to_mono_many(vcons_in_ram)
            print(f"Mono vcons: {len(vcons_mono)}")
        print(f"Mono vcons: {vcons_mono[0]}")

        vcons_resampled = None
        with with_timing("Resampling."):
            vcons_resampled = vcon.resample_many(vcons_mono)
            print(f"Resampled vcons: {len(vcons_resampled)}")
        print(f"Resampled vcons: {vcons_resampled[0]}")

        vcons_vad = None
        with with_timing("Applying VAD."):
            vcons_vad = vcon.apply_vad_many(vcons_resampled)
            print(f"VAD vcons: {len(vcons_vad)}")
        print(f"VAD vcons: {vcons_vad[0]}")

        vcons_padded = None
        with with_timing("Padding."):
            vcons_padded = vcon.pad_many(vcons_vad)
        print(f"Padded vcons: {vcons_padded[0]}")

        # vcons_on_gpu = None
        # with with_timing("Moving to GPU."):
        #     vcons_on_gpu = vcon.move_to_gpu_many(vcons_vad)
        # print(f"Vcons on GPU: {vcons_on_gpu[0]}")

        vcons_batched = None
        with with_timing("Batching."):
            vcons_batched = vcon.make_batches(vcons_padded)
        print(f"Batched vcons: {vcons_batched[0]}")

        vcons_detected = None
        
        with with_timing("Identifying languages."):
            vcons_detected = ai.identify_languages(vcons_batched, lang_detect_model, lang_detect_processor)
        print(f"Detected vcons: {vcons_detected[0]}")

        with with_timing("Splitting by language."):
            vcons_en, vcons_non_en = vcon.split_by_language(vcons_detected)
        print(f"En vcons: {vcons_en[0]}")
        print(f"Non-en vcons: {vcons_non_en[0]}")

        info_header(f"Batching {len(vcons_en)} en vcons.")
        vcons_en_batched = vcon.make_batches(vcons_en)
        print(f"En batched vcons: {vcons_en_batched[0]}")

        info_header(f"Batching {len(vcons_non_en)} non-en vcons.")
        vcons_non_en_batched = vcon.make_batches(vcons_non_en)
        print(f"Non-en batched vcons: {vcons_non_en_batched[0]}")
        
        vcons_en_transcribed = None
        with with_timing("Transcribing en vcons."):
            vcons_en_transcribed = ai.transcribe_many(vcons_en_batched, en_model)
        print(f"En transcribed vcons: {vcons_en_transcribed[0]}")
        
        vcons_non_en_transcribed = None
        with with_timing("Transcribing non-en vcons."):
            vcons_non_en_transcribed = ai.transcribe_many(vcons_non_en_batched, non_en_model)
        print(f"Non-en transcribed vcons: {vcons_non_en_transcribed[0]}")

        with with_timing("Marking as done."):
            for vcons_batch in [vcons_en_transcribed, vcons_non_en_transcribed]:
                vcon.mark_vcons_as_done(vcons_batch)
        
        with with_timing("Updating on DB."):
            vcon.update_vcons_on_db(vcons_en_transcribed)
            vcon.update_vcons_on_db(vcons_non_en_transcribed)

        with with_timing("Clearing processing."):
            cache.clear_processing()
        time.sleep(1)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribey main entry point")
    parser.add_argument("mode", choices=["head", "worker", "discover", "reserver", "print", "delete_all"], help="head:slurp and run worker. ")
    parser.add_argument("--url", type=str, default=settings.sftp_url, help="Override SFTP URL (applies to both head and worker)")
    parser.add_argument("--production", action="store_true", default=False, help="Enable production mode (applies to both head and worker)")
    args = parser.parse_args()

    debug = not args.production
    
    cache.init()
    with with_timing("Clearing cache."):
        cache.clear()


    if args.mode == "head":
        if debug:
            vcon.delete_all()
        vcon.discover(args.url)
        main(args.url)
    elif args.mode == "worker":
        main(args.url)
    elif args.mode == "discover":
        if debug:
            vcon.delete_all()
        vcon.discover(args.url)
    elif args.mode == "reserver":
        if debug:
            vcon.delete_all()
        vcon.discover(args.url)
        reserver.actually_start(args.url, Queue(maxsize=1), threading.Lock(), once=True)
    elif args.mode == "print":
        vcon.load_and_print_all()
    elif args.mode == "delete_all":
        vcon.delete_all()
