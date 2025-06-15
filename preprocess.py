import logging
import threading
from multiprocessing import Queue
from queue import Empty
from time import perf_counter
from typing import List

import cupy as cp
import torch

import audio
import cache
import process
import settings
import stats
import vcon_utils as vcon
from gpu import move_to_gpu_maybe
from process import ShutdownException
from stats import with_blocking_time
from vcon_class import Vcon
from vcon_queue import VconQueue

def preprocess_vcon_one(vcon_cur: Vcon, stats_queue: Queue):
    try:
        filename = vcon.processing_filename(vcon_cur)
        vcon.move_to_processing(vcon_cur)
        audio_data, sample_rate = audio.load_to_cpu(filename)
        audio_data : torch.Tensor
        sample_rate : int
        duration = audio.duration(audio_data, sample_rate)
        audio_data = audio.ensure_mono(audio_data)
        audio_data = audio.resample(audio_data)
        audio_data = audio.vad(audio_data)
        audio_data = cp.asarray(audio_data.detach())
        audio_data = audio_data.squeeze()
        if audio_data.ndim != 1:
            audio_data = audio_data[0]
        audio_data = audio_data.astype(cp.float32)
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

# Essentially, we will accumulate vcons in a batch,
# until the longest * number of vcons is equal to the max_bytesa
# We will batch in groups of 512 or so.
# Do not wait too long between batches.

def process_batch(batch: List[Vcon], stats_queue: Queue) -> List[Vcon]:
    batch = vcon.pad_many(batch)
    return batch

def is_batch_ready(batch: List[Vcon], time_passed: float) -> bool:
    if time_passed > settings.preprocess_batch_timeout_seconds:
        return True
    if len(batch) > settings.preprocess_batch_max_len:
        return True
    if vcon.size_of_list(batch) > settings.preprocess_batch_max_size:
        return True
    return False

# Technically may have some max size problems but whatever
def start(reserved_vcons_queue: VconQueue,
          preprocessed_vcons_queue: VconQueue,
          stats_queue: Queue):
    batch : List[Vcon] = []
    try:
        vcons_bytes : int = 0
        vcons_count : int = 0
        vcons_duration : int = 0
        time_start : float = perf_counter()
        while True:
            time_now : float = perf_counter()
            time_passed : float = time_now - time_start
            if is_batch_ready(batch, time_passed):
                batch = process_batch(batch, stats_queue)
                for vcon_cur in batch:
                    vcons_count += 1
                    vcons_bytes += vcon_cur.bytes
                    vcons_duration += vcon_cur.duration
                    stats.add(stats_queue, "vcons_count", vcons_count)
                    stats.add(stats_queue, "vcons_bytes", vcons_bytes)
                    stats.add(stats_queue, "vcons_duration", vcons_duration)
                    print(f"Preprocessed {vcon_cur}")
                    with with_blocking_time():
                        preprocessed_vcons_queue.put(vcon_cur)
                    batch = []
                    time_start = perf_counter()
            with with_blocking_time():
                try:
                    vcon_cur : Vcon | None = reserved_vcons_queue.get(timeout=settings.preprocess_batch_timeout_seconds)
                except Empty:
                    pass
            vcon_cur = preprocess_vcon_one(vcon_cur, stats_queue)
            if vcon_cur:
                batch.append(vcon_cur)
    except ShutdownException:
        pass

def start_thread(reserved_vcons_queue: VconQueue, preprocessed_vcons_queue: VconQueue, stats_queue: Queue):
    import time
    stats.add(stats_queue, "start_time", time.time())
    thread = threading.Thread(target=start, args=(reserved_vcons_queue, preprocessed_vcons_queue, stats_queue,))
    thread.name = "preprocess"
    thread.start()
