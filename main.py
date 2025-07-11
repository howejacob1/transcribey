import torch.multiprocessing as multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import argparse
import logging
import threading
import time
from concurrent.futures import as_completed, ThreadPoolExecutor
from queue import Empty

import numpy as np
import torch
import torchaudio

import cache
import discover
import gpu
import lang_detect
import preprocess
import reserver
import send_results
import settings
import stats
import transcribe
import vcon_utils as vcon
from log_utils import info_header, with_timing
from process import stop_threads_and_processes
from utils import dump_thread_stacks, dir_size_bytes, size_of_file, clear_screen, die_after_delay

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(threadName)s] %(levelname)s - %(message)s')
logging.getLogger("paramiko").setLevel(logging.INFO)

def main(sftp_url, stats_queue=None):
    # sftp = sftp_utils.connect_keep_trying(sftp_url)
    vcon.unmarked_all_reserved()
    programs = []
    reserved_vcons_queue = multiprocessing.Queue(maxsize=1026)
    programs.append(reserver.start_process(sftp_url, reserved_vcons_queue, stats_queue))
    preprocessed_vcons_queue = multiprocessing.Queue(maxsize=1026)
    programs.append(preprocess.start_process(reserved_vcons_queue, preprocessed_vcons_queue, stats_queue))
    lang_detected_en_vcons_queue = multiprocessing.Queue(maxsize=10)
    lang_detected_non_en_vcons_queue = multiprocessing.Queue(maxsize=10)    
    programs.append(lang_detect.start_process(preprocessed_vcons_queue, lang_detected_en_vcons_queue, lang_detected_non_en_vcons_queue, stats_queue))
    transcribed_vcons_queue = multiprocessing.Queue(maxsize=10)
    programs.append(transcribe.start_process_en(lang_detected_en_vcons_queue, transcribed_vcons_queue, stats_queue))
    if not settings.mark_non_english_as_corrupt:
        programs.append(transcribe.start_process_non_en(lang_detected_non_en_vcons_queue, transcribed_vcons_queue, stats_queue))
    programs.append(send_results.start_process(transcribed_vcons_queue, stats_queue))

    # Simple queue watching function instead of watch_vcon_queue
    def watch_queue(queue_to_watch):
        try:
            while True:
                vcon_item = queue_to_watch.get()
                print(vcon_item)
        except KeyboardInterrupt:
            print("Stopped watching queue.")
        except Exception as e:
            print(f"Error in watch_queue: {e}")

    # Uncomment to see queue flow during debugging
    stats.run(stats_queue)
    stop_threads_and_processes(programs, block=False)
    
    die_after_delay(5)

    print("Done.")
        

def print_out_statistics():
    from mongo_utils import db
    
    # Get all vcons count
    total_vcons = db.count_documents({})
    print(f"Total vcons: {total_vcons}")
    
    # Get done vcons count
    done_vcons = db.count_documents({"done": True})
    print(f"Total vcons that are done: {done_vcons}")
    
    # Get corrupt vcons count
    corrupt_vcons = db.count_documents({"corrupt": True})
    print(f"Total vcons that are corrupt: {corrupt_vcons}")
    
    # Calculate percentages
    if done_vcons > 0:
        corrupt_percentage = (corrupt_vcons / done_vcons) * 100
        print(f"Percentage corrupt / done: {corrupt_percentage:.2f}%")
    else:
        print("Percentage corrupt / done: N/A (no done vcons)")
    
    if total_vcons > 0:
        done_percentage = (done_vcons / total_vcons) * 100
        print(f"Percentage done / total: {done_percentage:.2f}%")
    else:
        print("Percentage done / total: N/A (no vcons)")
    
    # Get English vcons count
    english_vcons = db.count_documents({
        "analysis": {
            "$elemMatch": {
                "type": "language_identification",
                "body.languages": "en"
            }
        }
    })
    print(f"Total English vcons: {english_vcons}")
    
    # Calculate English percentage
    if done_vcons > 0:
        english_percentage = (english_vcons / done_vcons) * 100
        print(f"Percentage English vcons / done: {english_percentage:.2f}%")
    else:
        print("Percentage English vcons / done: N/A (no done vcons)")
    
    # Get Spanish vcons count
    spanish_vcons = db.count_documents({
        "analysis": {
            "$elemMatch": {
                "type": "language_identification",
                "body.languages": "es"
            }
        }
    })
    print(f"Total Spanish vcons: {spanish_vcons}")
    
    # Calculate Spanish percentage
    if done_vcons > 0:
        spanish_percentage = (spanish_vcons / done_vcons) * 100
        print(f"Percentage Spanish vcons / done: {spanish_percentage:.2f}%")
    else:
        print("Percentage Spanish vcons / done: N/A (no done vcons)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribey main entry point")
    parser.add_argument("mode", choices=["head", "worker", "discover", "print", "delete_all", "measure", "dump_jsonl", "stats"], help="head:slurp and run worker. ")
    parser.add_argument("--url", type=str, default=settings.sftp_url, help="Override SFTP URL (applies to both head and worker)")
    parser.add_argument("--production", action="store_true", default=False, help="Enable production mode (applies to both head and worker)")
    parser.add_argument("--dataset", choices=["fast", "med", "slow", "test_recordings"], help="use precompiled dataset")
    args = parser.parse_args()
    print(f"start method: {multiprocessing.get_start_method()}")
    assert multiprocessing.get_start_method() == "spawn", f"Expected spawn, got {multiprocessing.get_start_method()}"

    if args.dataset == "fast":
        args.url = "sftp://bantaim@127.0.0.1:22/home/bantaim/conserver/fake_wavs_cute/"
    elif args.dataset == "med":
        args.url = "sftp://bantaim@127.0.0.1:22/home/bantaim/conserver/openslr-12/"
    elif args.dataset == "slow":
        args.url = "sftp://bantaim@127.0.0.1:22/home/bantaim/conserver/fake_wavs_medlarge/"
    elif args.dataset == "test_recordings":
        args.url = "sftp://bantaim@127.0.0.1:22/home/bantaim/conserver/recordings_2025-06-19/"
    debug = not args.production
    stats_queue = multiprocessing.Queue(maxsize=100000000)
    discover_process = None

    logging.info(f"Start in mode {args.mode}.")
    if args.mode == "head":
        if debug:
            vcon.delete_all()
        discover_process = discover.start_process(args.url, stats_queue)
        main(args.url, stats_queue)
    elif args.mode == "measure":
        if debug:
            vcon.delete_all()
        discover_process = discover.start_process(args.url, stats_queue)
        main(args.url, stats_queue)
    elif args.mode == "worker":
        main(args.url, stats_queue)
    elif args.mode == "discover":
        if debug:
            vcon.delete_all()
        discover_process = discover.start_process(args.url, None, print_status=True)
        #stats.run(stats_queue)
    elif args.mode == "print":
        vcon.load_and_print_all()
    elif args.mode == "delete_all":
        vcon.delete_all()
    elif args.mode == "dump_jsonl":
        vcon.dump_jsonl()
    elif args.mode == "stats":
        print_out_statistics()
        

    # We have a DB full of Vcons. 
    # Many are not analyzed. Many do not have langid or something. 
    # However, we also need to discover. 
    # Things we have to do: 
    # see if a vcon exists. If so, skip.
    # Actually, this is very easy to do if no more than two discoveries run on the same dir. 
    # All we do is pull all existing vcons, then pull all sftp things. 
    # for each filename, see if it's in the list. If not, add it to the list, and add it to a list of things to upload.
    # Optimize accordingly. 

    # We also need to be able to do a reserve. What reserve does is find a vcon that does not have analysis and is not reserved.
    # We should just reserve a larger number of vcons...

# We have many target dirs full of vcons. 
# running discover is fine. 
# When we run discover.... we need to transfer them to a backup.
# I mean that's what we're doing rn
# Perhaps instead.....
# When we do discover....

# we have an SFTP target dir full of vcons. 
# We need to both backup and transfer the 
# wondeirng if downloading them we should process them as we go--- idk tho. 
# Discover is fine...
# Now we have links to all vcons...
# 
