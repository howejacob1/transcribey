import logging
import multiprocessing
import numpy as np
import torchaudio
import threading
from multiprocessing import Queue
from queue import Empty
from time import time
from typing import List

import torch

import audio
import cache
import process
import settings
import stats
import vcon_utils as vcon
from utils import let_other_threads_run
from gpu import move_to_gpu_maybe
from process import ShutdownException
from stats import with_blocking_time
from vcon_class import Vcon
def preprocess_vcon_one(vcon_cur: Vcon, stats_queue: Queue):
    try:
        filename = vcon.processing_filename(vcon_cur)
        vcon.move_to_processing(vcon_cur)
        audio_data : torch.Tensor
        sample_rate : int
        audio_data, sample_rate = audio.load_to_cpu(filename)
        duration = audio.duration(audio_data, sample_rate)
        audio_data = audio.ensure_mono(audio_data)
        resampler = torchaudio.transforms.Resample(sample_rate, settings.sample_rate)
        audio_data = resampler(audio_data)
        audio_data = audio.vad(audio_data)
        # Convert PyTorch tensor to NumPy array for downstream processing
        audio_data = np.asarray(audio_data.detach().cpu().numpy())
        audio_data = audio_data.squeeze()
        if audio_data.ndim != 1:
            audio_data = audio_data[0]
        audio_data = audio_data.astype(np.float32)
        audio_data = move_to_gpu_maybe(audio_data)
        bytes = audio.get_size(audio_data)
        vcon_cur.size = bytes
        vcon_cur.duration = duration
        vcon_cur.audio = audio_data
        vcon_cur.sample_rate = sample_rate
        return vcon_cur
    except RuntimeError:
        vcon.mark_vcon_as_invalid(vcon_cur)
        vcon.remove_vcon_from_processing(vcon_cur)
        return None

# Technically may have some max size problems but whatever
def preprocess(reserved_vcons_queue: multiprocessing.Queue,
          preprocessed_vcons_queue: multiprocessing.Queue,
          stats_queue: multiprocessing.Queue):
    stats.start(stats_queue)
    vcons_in_memory = []
    try:
        process.setup_signal_handlers()
        while True:
            with with_blocking_time(stats_queue):
                vcon_cur = reserved_vcons_queue.get()
            vcons_in_memory.append(vcon_cur)
            
            vcon_cur = preprocess_vcon_one(vcon_cur, stats_queue)
            if vcon_cur is not None:  # Only proceed if preprocessing succeeded
                stats.count(stats_queue)
                stats.bytes(stats_queue, vcon_cur.size)
                stats.duration(stats_queue, vcon_cur.duration)
                with with_blocking_time(stats_queue):
                    preprocessed_vcons_queue.put(vcon_cur)
            
            # Remove from our tracking list once processed
            if vcon_cur in vcons_in_memory:
                vcons_in_memory.remove(vcon_cur)
                
    except ShutdownException:
        #print("PREPROCESS: Received shutdown signal, cleaning up...")
        # Clean up vcons in memory
        stats.stop(stats_queue)
        exit()

def start_process(reserved_vcons_queue: multiprocessing.Queue, preprocessed_vcons_queue: multiprocessing.Queue, stats_queue: multiprocessing.Queue):
    """Start preprocess process"""
    return process.start_process(target=preprocess, args=(reserved_vcons_queue, preprocessed_vcons_queue, stats_queue))
