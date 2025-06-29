import time

import settings
from utils import let_other_threads_run
import cache
import process
import stats
import vcon_utils as vcon
from process import ShutdownException
from sftp import connect_keep_trying
from stats import with_blocking_time
from utils import dont_overwhelm_server, dump_thread_stacks

def reserver(sftp_url, vcons_ready_queue, stats_queue):
    stats.start(stats_queue)
    cache.init()
    try:
        sftp = None
        stats.start_blocking(stats_queue)
        while True:
            while sftp is None:
                try:
                    sftp = connect_keep_trying(sftp_url)
                except Exception as e:
                    sftp = None
                    with with_blocking_time(stats_queue):
                        dont_overwhelm_server()
            vcon_cur = vcon.find_and_reserve()
            if vcon_cur:
                stats.stop_blocking(stats_queue)
                vcon.cache_audio(vcon_cur, sftp)
                stats.start_blocking(stats_queue)
                stats.count(stats_queue)
                stats.bytes(stats_queue, vcon_cur.size)
                #duration = vcon_cur.size / (settings.sample_rate*2)
                #stats.duration(stats_queue, duration)
                vcons_ready_queue.put(vcon_cur)
            else:
                dont_overwhelm_server()

    except ShutdownException:
        dump_thread_stacks()
    finally:
        stats.stop(stats_queue)
        if sftp is not None:
            sftp.close()
            


def start_process(sftp_url, vcons_ready_queue, stats_queue):
    return process.start_process(target=reserver, args=(sftp_url, vcons_ready_queue, stats_queue))
