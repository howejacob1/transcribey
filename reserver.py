import logging
import time
from time import perf_counter

import cache
import process
import stats
import vcon_utils as vcon
from process import ShutdownException
from sftp import connect_keep_trying
from stats import with_blocking_time
from utils import dont_overwhelm_server

def start(sftp_url, vcons_ready_queue, stats_queue):
    cache.init()
    try:
        sftp = None
        vcons_count = 0
        vcons_bytes = 0
        while True:
            while sftp is None:
                try:
                    sftp = connect_keep_trying(sftp_url)
                except Exception as e:
                    sftp = None
                    dont_overwhelm_server()
            vcon_cur = vcon.find_and_reserve()
            if vcon_cur:
                vcon.cache_audio(vcon_cur, sftp)
                vcons_count += 1
                vcons_bytes += vcon_cur.size
                stats.add(stats_queue, "vcons_count", vcons_count)
                stats.add(stats_queue, "vcons_bytes", vcons_bytes)
                with with_blocking_time(stats_queue):
                    vcons_ready_queue.put(vcon_cur)
            else:
                dont_overwhelm_server()
    except ShutdownException:
        pass
    finally:
        if sftp is not None:
            sftp.close()
        stats.add(stats_queue, "stop_time", time.time())

def start_process(sftp_url, vcons_ready_queue, stats_queue):
    stats.add(stats_queue, "start_time", time.time())
    return process.start_process(target=start, args=(sftp_url, vcons_ready_queue, stats_queue))
