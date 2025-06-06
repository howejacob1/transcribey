from sftp import connect_keep_trying
import cache
import vcon_utils as vcon
import threading
import time
import logging

def actually_start(sftp_url, vcons_ready_queue):
    sftp = None
    while True:
        if sftp is None:
            sftp = connect_keep_trying(sftp_url)
        bytes_to_reserve = cache.bytes_to_reserve()
        if bytes_to_reserve > 0:
            try:
                vcons = vcon.find_and_reserve_many(bytes_to_reserve)
                if vcons:
                    vcon.cache_vcon_audio_many(vcons, sftp)
                    vcons_ready_queue.put(vcons)
            except Exception as e:
                logging.info(f"Failed to connect in reserver_thread: {e}")
                sftp = None
        time.sleep(1)

def start(sftp_url, vcons_ready_queue):
    thread = threading.Thread(target=actually_start, args=(sftp_url, vcons_ready_queue), daemon=True)
    thread.start()
    return thread
