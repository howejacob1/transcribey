import logging

import settings
from transcribe import load_nvidia

def load():
    non_en_model_name = settings.non_en_model_name
    non_en_model = load_nvidia(non_en_model_name)
    return non_en_model

def start_thread(lang_detected_en_vcons_queue, transcribed_vcons_queue):
    logging.info("Starting transcribe en thread.")
    en_model = load()
    while True:
        vcons = lang_detected_en_vcons_queue.get()
        transcribe_en(vcons, en_model)
        transcribed_vcons_queue.put(vcons)