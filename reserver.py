from sftp import connect
import cache
import vcon_utils as vcon
from utils import dump_thread_stacks
import threading
import time
import logging
import queue

def actually_start(sftp_url, vcons_ready_queue, vcons_lock, keep_running, once=False):
    logging.info(f"Starting reserver")
    try:
        sftp = None
        while keep_running.is_set():
            if sftp is None:
                try:
                    sftp = connect(sftp_url)
                    logging.info(f"!!!!!!!!!!!!!!!!!!!!!Connected to {sftp_url}")
                except Exception as e:
                    logging.info(f"Error connecting.")
                    sftp = None
                    time.sleep(1)
                    continue
            bytes_to_reserve = cache.bytes_to_reserve()
            if bytes_to_reserve > 0:
                vcons = vcon.find_and_reserve_many(bytes_to_reserve)
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
                        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1Done putting {len(vcons)} vcons on queue.")
            else:
                time.sleep(1)
            
                # except Exception as e:
                #     logging.info(f"Error in reserver: {e}")
                #     dump_thread_stacks
                #     sftp.close()
                #     sftp = None
            if once:
                break
            time.sleep(1)
    finally:
        if sftp is not None:
            sftp.close()

def start(sftp_url, vcons_ready_queue, vcons_lock, keep_running):
    thread = threading.Thread(target=actually_start, args=(sftp_url,vcons_ready_queue, vcons_lock, keep_running), daemon=False)
    thread.name = "reserver_thread"
    thread.start()
    return thread
