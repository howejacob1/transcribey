import logging
from multiprocessing import Queue
import time

from process import ShutdownException
import vcon_utils
import process
import stats
from vcon_class import Vcon
from stats import with_blocking_time

def start_thread(transcribed_vcons_queue):
    logging.info("Starting send results thread.")

def send_results(transcribed_vcons_queue: Queue, stats_queue: Queue):
    vcons_count = 0
    vcons_bytes = 0
    vcons_duration = 0
    try:
        while True:
            with with_blocking_time(stats_queue):
                vcon_cur : Vcon = transcribed_vcons_queue.get()
            vcon_utils.update_vcon_on_db(vcon_cur)
            vcons_count += 1
            vcons_bytes += vcon_cur.size
            vcons_duration += vcon_cur.duration
            stats.add(stats_queue, "vcons_count", vcons_count)
            stats.add(stats_queue, "vcons_bytes", vcons_bytes)
            stats.add(stats_queue, "vcons_duration", vcons_duration)
    except ShutdownException:
        pass

def start_process(url, stats_queue):
    """Start discovery process"""
    stats.add(stats_queue, "start_time", time.time())
    return process.start_process(target=send_results, args=(url, stats_queue))
