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
        bytes = audio.get_size(audio_data)
        vcon_cur.size = bytes
        vcon_cur.duration = duration
        vcon_cur.audio = audio_data
        vcon_cur.sample_rate = settings.sample_rate
        return vcon_cur
    except RuntimeError:
        vcon.mark_vcon_as_invalid(vcon_cur)
        vcon.remove_vcon_from_processing(vcon_cur)
        return None

def collect_batch_with_timeout(reserved_vcons_queue: multiprocessing.Queue, batch_size: int = 16, timeout: float = 0.1):
    """Collect vcons for a batch with timeout"""
    batch = []
    batch_start = time()
    
    while len(batch) < batch_size:
        time_passed = time() - batch_start
        if time_passed > timeout:
            break
            
        try:
            # Calculate remaining timeout
            remaining_timeout = max(0, timeout - time_passed)
            if remaining_timeout <= 0:
                break
                
            # Try to get a vcon with the remaining timeout
            vcon_cur = reserved_vcons_queue.get(timeout=remaining_timeout)
            batch.append(vcon_cur)
        except Empty:
            # Timeout reached, break and process what we have
            break
    
    return batch

def preprocess_batch(batch: List[Vcon], stats_queue: Queue):
    """Preprocess a batch of vcons and pad them to the same length"""
    processed_batch = []
    
    # Process each vcon in the batch
    for vcon_cur in batch:
        processed_vcon = preprocess_vcon_one(vcon_cur, stats_queue)
        if processed_vcon is not None:
            processed_batch.append(processed_vcon)
    
    if not processed_batch:
        return None
    
    # Find the longest duration in the batch
    max_duration = 0
    for vcon_cur in processed_batch:
        if vcon_cur.duration > max_duration:
            max_duration = vcon_cur.duration
    
    # Pad all vcons to the same length
    for vcon_cur in processed_batch:
        audio_data = vcon_cur.audio
        # Convert to torch tensor for padding
        if isinstance(audio_data, np.ndarray):
            audio_tensor = torch.from_numpy(audio_data)
        else:
            audio_tensor = audio_data
        
        # Pad to max duration
        padded_audio = audio.pad_audio(audio_tensor, settings.sample_rate, max_duration)
        
        # Convert back to numpy
        if isinstance(padded_audio, torch.Tensor):
            vcon_cur.audio = padded_audio.detach().cpu().numpy().astype(np.float32)
        else:
            vcon_cur.audio = padded_audio.astype(np.float32)
    
    return processed_batch

# Technically may have some max size problems but whatever
def preprocess(reserved_vcons_queue: multiprocessing.Queue,
          preprocessed_vcons_queue: multiprocessing.Queue,
          stats_queue: multiprocessing.Queue):
    stats.start(stats_queue)
    vcons_in_memory = []
    try:
        process.setup_signal_handlers()
        while True:
            # Collect a batch with timeout
            with with_blocking_time(stats_queue):
                batch = collect_batch_with_timeout(reserved_vcons_queue, batch_size=16, timeout=0.1)
            
            if not batch:
                continue
                
            vcons_in_memory.extend(batch)
            
            # Process the batch
            processed_batch = preprocess_batch(batch, stats_queue)
            
            if processed_batch is not None:
                # Update stats for the entire batch
                for vcon_cur in processed_batch:
                    stats.count(stats_queue)
                    stats.bytes(stats_queue, vcon_cur.size)
                    stats.duration(stats_queue, vcon_cur.duration)
                
                # Put the entire batch in the queue
                with with_blocking_time(stats_queue):
                    preprocessed_vcons_queue.put(processed_batch)
            
            # Remove from our tracking list once processed
            for vcon_cur in batch:
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
