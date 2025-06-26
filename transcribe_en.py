import time
from torch.multiprocessing import Queue

import process
import settings
import stats
from transcribe import transcribe

def transcribe_en(lang_detected_en_vcons_queue: Queue,
                  transcribed_vcons_queue: Queue,
                  stats_queue: Queue):
    """Main transcription function for English audio"""
    transcribe(lang_detected_en_vcons_queue, transcribed_vcons_queue, stats_queue, settings.en_model_name, "en")

def start_process(lang_detected_en_vcons_queue: Queue,
                  transcribed_vcons_queue: Queue,
                  stats_queue: Queue):
    """Start the English transcription thread"""
    return process.start_process(target=transcribe_en, args=(lang_detected_en_vcons_queue, transcribed_vcons_queue, stats_queue))

    