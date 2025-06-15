import threading
import time
from multiprocessing import Queue

import settings
import stats
from transcribe import load_nvidia, transcribe
from vcon_queue import VconQueue

def load():
    """Load English transcription model"""
    model_name = settings.en_model_name
    model = load_nvidia(model_name)
    return model

def transcribe_en(lang_detected_en_vcons_queue: VconQueue,
                  transcribed_vcons_queue: VconQueue,
                  stats_queue: Queue):
    """Main transcription function for English audio"""
    stats.add(stats_queue, "start_time", time.time())
    model = load()
    transcribe(lang_detected_en_vcons_queue, transcribed_vcons_queue, model, stats_queue, language="en")

def start_thread(lang_detected_en_vcons_queue: VconQueue,
                 transcribed_vcons_queue: VconQueue,
                 stats_queue: Queue):
    """Start the English transcription thread"""
    thread = threading.Thread(target=transcribe_en, args=(lang_detected_en_vcons_queue, transcribed_vcons_queue, stats_queue))
    thread.name = "transcribe_en"
    thread.start()
    return thread