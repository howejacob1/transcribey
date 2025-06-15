import time
from queue import Empty
from time import perf_counter
from typing import List

import cupy as cp
import torch
import whisper

import gpu
import process
import settings
import stats
import vcon_utils
from multiprocessing import Queue
from gpu import move_to_gpu_maybe, we_have_a_gpu, gc_collect_maybe, gpu_ram_free_bytes
from process import ShutdownException
from stats import with_blocking_time
from utils import let_other_threads_run, dump_thread_stacks
from vcon_class import Vcon

def load_nvidia(model_name):
    """Load nvidia model using NVIDIA NeMo."""
    total_start_time = time.time()
    print(f"Loading model {model_name}.")
    with suppress_output(should_suppress=False):
        model = ASRModel.from_pretrained(model_name=model_name)
    model.eval()
    print(f"Model {model_name} loaded in {time.time() - total_start_time:.2f} seconds total")
    return model

def transcribe_batch(vcon_batch, model, language="en"):
    """Transcribe a batch of vcons using the model"""
    print(f"transcribing {len(vcon_batch)} vcons (gpu memory: {gpu_ram_free_bytes()})")
    audio_data_batch = vcon_utils.batch_to_audio_data(vcon_batch)
    
    # Configure transcription based on language
    config = {"batch_size": min(len(audio_data_batch), 32)}
    if language != "en":
        config.update({
            "source_lang": language,
            "target_lang": language, 
            "task": "asr",
            "pnc": "yes"
        })
    
    try:
        # Transcribe the batch
        all_transcriptions = model.transcribe(audio_data_batch, **config)
        
        # Process results
        vcons = []
        for vcon_cur, transcription in zip(vcon_batch, all_transcriptions):
            if hasattr(transcription, 'text'):
                text = transcription.text
            elif isinstance(transcription, str):
                text = transcription
            else:
                text = str(transcription)
            
            vcon_cur = vcon_utils.set_transcript(vcon_cur, text)
            vcons.append(vcon_cur)
        
        gc_collect_maybe()
        print(f"Transcribed {len(vcons)} vcons")
        return vcons
        
    except Exception as e:
        print(f"Error transcribing batch: {e}")
        # Return vcons with empty transcripts as fallback
        vcons = []
        for vcon_cur in vcon_batch:
            vcon_cur = vcon_utils.set_transcript(vcon_cur, "")
            vcons.append(vcon_cur)
        return vcons

def is_batch_ready(batch: List[Vcon], batch_start: float, total_size: int):
    time_passed = perf_counter() - batch_start
    if time_passed > settings.transcribe_batch_timeout_seconds:
        return True
    if len(batch) > settings.transcribe_batch_max_len:
        return True
    if total_size > settings.transcribe_batch_max_size:
        return True
    return False

def collect_vcons(lang_detected_queue: Queue, target_vcon: Vcon | None, stats_queue: Queue):
    try:
        if not target_vcon:
            with with_blocking_time(stats_queue):
                target_vcon = lang_detected_queue.get(timeout=settings.transcribe_batch_timeout_seconds)
    except Empty:
        return [target_vcon], None
    
    batch_start : float = perf_counter()
    total_size : int = 0
    batch : List = [target_vcon]
        
    while not is_batch_ready(batch, batch_start, total_size):
        try:
            with with_blocking_time(stats_queue):
                cur_vcon = lang_detected_queue.get(timeout=settings.transcribe_batch_timeout_seconds)
            if cur_vcon.size == target_vcon.size:
                # Audio data is already on GPU from preprocessing
                batch.append(cur_vcon)
                total_size += cur_vcon.size
            else: 
                return batch, cur_vcon
        except TimeoutError:
            return batch, target_vcon
    return batch, target_vcon

def transcribe(lang_detected_queue: Queue,
               transcribed_queue: Queue,
               stats_queue: Queue):
    model = load_nvidia()
    
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
