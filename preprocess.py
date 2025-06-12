import time

import logging
from log_utils import with_blocking_time

import audio
import stats
import vcon_utils as vcon
import cache

def preprocess_vcon_one(vcon_cur, stats_queue):
    try:
        filename = vcon.processing_filename(vcon_cur)
        vcon.move_to_processing(vcon_cur)
        audio_data, sample_rate = audio.load_to_cpu(filename)
        vcon_cur.audio = audio_data
        vcon_cur["sample_rate"] = sample_rate
        duration = audio.audio_data_duration(audio_data, sample_rate)
        vcon_cur = convert_to_mono_maybe(vcon_cur)
        vcon_cur = resample_vcon_one(vcon_cur)
        #vcon_cur = apply_vad_one(vcon_cur, vad)
        bytes = audio.get_size(audio_data)
        vcon_cur["size"] = bytes
        audio_data = vcon_cur.audio
        audio_data = audio_data.squeeze()
        vcon_cur.audio = audio_data
        return vcon_cur, bytes, duration
    except RuntimeError:
        vcon.mark_vcon_as_invalid(vcon_cur)
        vcon.remove_vcon_from_processing(vcon_cur)
        return None, 0, 0


def start_thread(reserved_vcons_queue, preprocessed_vcons_queue, stats_queue):
    stats.add(stats_queue, "start_time", time.time())
    vcons_count = 0
    vcons_bytes = 0
    vcons_duration = 0
    while True:
        with with_blocking_time(stats_queue):
            vcon_cur = reserved_vcons_queue.get()
        vcon_cur, bytes, duration = preprocess_vcon_one(vcon_cur, stats_queue)
        vcons_count += 1
        vcons_bytes += bytes
        vcons_duration += duration
        stats.add(stats_queue, "vcons_count", vcons_count)
        stats.add(stats_queue, "vcons_bytes", vcons_bytes)
        stats.add(stats_queue, "vcons_duration", vcons_duration)
        if vcon_cur is not None:
            with with_blocking_time(stats_queue):
                preprocessed_vcons_queue.put(vcon_cur)
