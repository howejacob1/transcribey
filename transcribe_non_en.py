import time
from multiprocessing import Queue

import process
import settings
import stats
from transcribe import transcribe

def transcribe_non_en(lang_detected_non_en_vcons_queue: Queue,
                  transcribed_vcons_queue: Queue,
                  stats_queue: Queue):
    """Main transcription function for non-English audio"""
    transcribe(lang_detected_non_en_vcons_queue, transcribed_vcons_queue, stats_queue, settings.non_en_model_name, "auto")

def start_process(lang_detected_non_en_vcons_queue: Queue,
                 transcribed_vcons_queue: Queue,
                 stats_queue: Queue):
    """Start the non-English transcription thread"""
    return process.start_process(target=transcribe_non_en, args=(lang_detected_non_en_vcons_queue, transcribed_vcons_queue, stats_queue))