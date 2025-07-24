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
    # Set process title for identification in nvidia-smi and ps
    try:
        from setproctitle import setproctitle
        import os
        setproctitle("transcribey-reserver")
        print(f"[PID {os.getpid()}] Set process title to: transcribey-reserver")
    except ImportError:
        print("setproctitle not available for reserver process")
    
    stats.start(stats_queue)
    cache.init()
    cache.clear()

    try:
        sftp = None
        while True:
            while sftp is None:
                try:
                    with with_blocking_time(stats_queue):
                        sftp, _ = connect_keep_trying(sftp_url)
                except Exception as e:
                    sftp = None
                    with with_blocking_time(stats_queue):
                        dont_overwhelm_server()
            #print("Reserving vcons")
            vcons_found = vcon.find_and_reserve_many(settings.reserver_total_batch_size)
            #print(f"Found {len(vcons_found)} vcons")
            if vcons_found:
                # Process files in parallel batches to reduce total caching time
                batch_size = settings.sftp_download_batch_size
                for i in range(0, len(vcons_found), batch_size):
                    batch = [vcon_cur for vcon_cur in vcons_found[i:i+batch_size] if vcon_cur]
                    
                    if batch:
                        # Time the entire batch
                        batch_start = time.time()
                        
                        # Cache all files in the batch in parallel
                        results = vcon.cache_audio_batch(batch, sftp)
                        
                        batch_end = time.time()
                        batch_time = batch_end - batch_start
                        
                        # Process results
                        for vcon_cur, error in results:
                            if error is None:
                                
                                with with_blocking_time(stats_queue):
                                    vcons_ready_queue.put(vcon_cur)
                            else:
                                print(f"Error caching {vcon_cur.uuid}: {error}")
                                vcon.mark_vcon_as_invalid(vcon_cur)
                            
                            stats.count(stats_queue)
                            stats.bytes(stats_queue, vcon_cur.size)
            else:
                with with_blocking_time(stats_queue):
                    dont_overwhelm_server()

    except ShutdownException:
        dump_thread_stacks()
    finally:
        stats.stop(stats_queue)
        if sftp is not None:
            sftp.close()
            


def start_process(sftp_url, vcons_ready_queue, stats_queue):
    return process.start_process(target=reserver, args=(sftp_url, vcons_ready_queue, stats_queue))
