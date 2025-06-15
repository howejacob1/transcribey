import logging
import threading
import time
from multiprocessing import Queue

import settings
import stats
from transcribe import load, transcribe
from vcon_queue import VconQueue

def start_thread(lang_detected_en_vcons_queue: VconQueue, transcribed_vcons_queue: VconQueue, stats_queue: Queue):
    """Start transcription thread for English vcons"""
    stats.add(stats_queue, "start_time", time.time())
    model = load()
    
    def transcribe_worker():
        try:
            while True:
                vcon_cur = lang_detected_en_vcons_queue.get()
                # Process transcription here
                transcribed_vcons_queue.put(vcon_cur)
        except Exception as e:
            logging.error(f"Error in transcribe worker: {e}")
    
    thread = threading.Thread(target=transcribe_worker)
    thread.daemon = True
    thread.start()
    return thread
