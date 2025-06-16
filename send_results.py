import logging
import time
from multiprocessing import Queue

import process
import stats
import vcon_utils
from process import ShutdownException, setup_signal_handlers
from stats import with_blocking_time
from vcon_class import Vcon

def send_results(transcribed_vcons_queue: Queue, stats_queue: Queue):
    setup_signal_handlers()
    stats.start(stats_queue)
    try:
        while True:
            with with_blocking_time(stats_queue):
                vcon_cur : Vcon = transcribed_vcons_queue.get()
            vcon_utils.update_vcon_on_db(vcon_cur)
            stats.count(stats_queue)
            stats.bytes(stats_queue, vcon_cur.size)
            stats.duration(stats_queue, vcon_cur.duration)
    except ShutdownException:
        pass

def start_process(transcribed_vcons_queue, stats_queue):
    """Start discovery process"""
    return process.start_process(target=send_results, args=(transcribed_vcons_queue, stats_queue))
