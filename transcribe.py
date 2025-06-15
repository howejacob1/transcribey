import logging
import threading
import time
from multiprocessing import Queue
from queue import Empty
from time import perf_counter
from typing import List

from nemo.collections.asr.models import ASRModel, EncDecSpeakerLabelModel

import gpu
import settings
import stats
from process import ShutdownException
from stats import with_blocking_time
from vcon_class import Vcon
from vcon_queue import VconQueue

from lang_detect import identify_languages

def load_nvidia_raw(model_name):
    # with suppress_output(should_suppress=True):
    model = ASRModel.from_pretrained(model_name=model_name)
    return model

def load_nvidia(model_name):
    """Load nvidia model using NVIDIA NeMo."""
    total_start_time = time.time()
    print(f"Loading model {model_name}.")
    model = load_nvidia_raw(model_name)
    # model = move_to_gpu_maybe(model)
    print(f"Model {model_name} loaded in {time.time() - total_start_time:.2f} seconds total")
    return model

def load():
    model_name = settings.en_model_name
    model = load_nvidia(model_name)
    return model

def is_batch_ready(batch : List[Vcon], batch_start : float, total_size : int):
    time_passed = perf_counter() - batch_start
    if time_passed > settings.lang_detect_batch_ready:
        return True
    if len(batch) > settings.land_detect_batch_max_len:
        return True
    if total_size > settings.lang_detect_batch_max_size:
        return True
    return False

def collect_vcons(preprocessed_vcons_queue : VconQueue, target_vcon: Vcon | None, stats_queue: Queue):
    try:
        if not target_vcon:
            with with_blocking_time(stats_queue):
                target_vcon = preprocessed_vcons_queue.get(timeout=settings.lang_detect_batch_timeout_seconds)
    except Empty:
        return [target_vcon], None
    
    batch_start : float = perf_counter()
    total_size : int = 0
    batch : List = [target_vcon]
    while not is_batch_ready(batch, batch_start, total_size):
        try:
            with with_blocking_time(stats_queue):
                cur_vcon = preprocessed_vcons_queue.get(timeout=settings.lang_detect_batch_timeout_seconds)
            if cur_vcon.size == target_vcon.size:
                batch.append(cur_vcon)
                total_size += cur_vcon.size
            else: 
                return batch, cur_vcon
        except TimeoutError:
            return batch, target_vcon
    return batch, target_vcon

def transcribe_batch(vcon_batch, model):
    """Transcribe a batch of vcons using the model"""
    # This is a placeholder - actual transcription logic would depend on the model
    # For now, just return the vcons with placeholder transcripts
    for vcon in vcon_batch:
        # In real implementation, you'd process audio through the model
        # vcon.transcript_text = model.transcribe(vcon.audio)
        vcon.transcript_text = f"Transcribed audio for {vcon.filename}"
    return vcon_batch

def transcribe(lang_detected_queue: VconQueue,
               transcribed_queue: VconQueue,
               model,
               stats_queue: Queue):
    stats.add(stats_queue, "start_time", time.time())
    target_vcon : Vcon | None = None
    vcons_bytes : int = 0
    vcons_count : int = 0
    vcons_duration : int = 0
    try:
        while True: # just run thread forever
            batch, target_vcon = collect_vcons(lang_detected_queue, target_vcon, stats_queue)
            transcribed_vcons = transcribe_batch(batch, model)
            
            for vcon_cur in transcribed_vcons:
                vcons_count += 1
                vcons_bytes += vcon_cur.size
                vcons_duration += vcon_cur.duration
                stats.add(stats_queue, "vcons_count", vcons_count)
                stats.add(stats_queue, "vcons_bytes", vcons_bytes)
                stats.add(stats_queue, "vcons_duration", vcons_duration)
                with with_blocking_time(stats_queue):
                    transcribed_queue.put(vcon_cur)
    except ShutdownException:
        pass

def start_thread(lang_detected_queue: VconQueue,
                 transcribed_queue: VconQueue,
                 stats_queue: Queue):
    thread = threading.Thread(target=transcribe, args=(lang_detected_queue, transcribed_queue, model, stats_queue))
    thread.start()
    return thread
