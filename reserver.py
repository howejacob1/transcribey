from sftp import connect
import cache
import vcon_utils as vcon
from utils import dump_thread_stacks
import threading
import time
import logging
import queue
import settings

def actually_start(sftp_url, vcons_ready_queue, vcons_lock, keep_running, once=False):
    logging.info(f"Starting reserver")
    
    try:
        sftp = None
        while keep_running.is_set():
            while sftp is None:
                try:
                    sftp = connect(sftp_url)
                except Exception as e:
                    sftp = None
                    time.sleep(1)
            #logging.info(f"attempting to reserve {settings.cache_size_bytes} bytes")
            vcons = vcon.find_and_reserve_many(settings.cache_size_bytes)
            if vcons:
                print(f"Reserving {len(vcons)} vcons")
                with vcons_lock:
                    vcon.cache_vcon_audio_many(vcons, sftp)
                    print(f"Finished caching {len(vcons)} vcons. Putting.")
                    while keep_running.is_set():
                        try:
                            vcons_ready_queue.put(vcons, timeout=0.1)
                            break
                        except queue.Full:
                            pass
            if once:
                break
            time.sleep(1) # don't overwhelm network
    finally:
        if sftp is not None:
            sftp.close()

def start(sftp_url, vcons_ready_queue, vcons_lock, keep_running):
    thread = threading.Thread(target=actually_start, args=(sftp_url,vcons_ready_queue, vcons_lock, keep_running), daemon=False)
    thread.name = "reserver_thread"
    thread.start()
    return thread
