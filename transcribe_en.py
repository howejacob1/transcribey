import settings
import time
from utils import suppress_output, move_to_gpu_maybe
import logging
from log_utils import with_timing
from transcribe import load_nvidia

def load():
    
    en_model = load_nvidia(settings.en_model_name)
    return en_model

def start_thread(lang_detected_en_vcons_queue, transcribed_vcons_queue):
    logging.info("Starting transcribe en thread.")
    en_model = load()
    while True:
        vcons = lang_detected_en_vcons_queue.get()
        transcribe_en(vcons, en_model)
        transcribed_vcons_queue.put(vcons)