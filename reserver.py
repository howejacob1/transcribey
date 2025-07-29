import time
import os

import settings
from utils import let_other_threads_run
import cache
import process
import stats
import vcon_utils as vcon
from process import ShutdownException
from stats import with_blocking_time
from utils import dont_overwhelm_server, dump_thread_stacks
from setproctitle import setproctitle
import os

def reserver_nfs(vcons_ready_queue, stats_queue):
    """Reserve and verify vcons for NFS - no downloading needed"""
    # Set process title for identification in nvidia-smi and ps
    setproctitle("transcribey-reserver-nfs")
    
    stats.start(stats_queue)

    try:
        while True:
            #print("Reserving vcons")
            vcons_found = vcon.find_and_reserve_many(settings.reserver_total_batch_size)
            #print(f"Found {len(vcons_found)} vcons")
            if vcons_found:
                # Process files in batches - just verify they exist and are readable
                batch_size = settings.sftp_download_batch_size  # Reuse same batch size setting
                for i in range(0, len(vcons_found), batch_size):
                    batch = [vcon_cur for vcon_cur in vcons_found[i:i+batch_size] if vcon_cur]
                    
                    if batch:
                        # Time the entire batch
                        batch_start = time.time()
                        
                        # Verify all files in the batch exist and are accessible
                        for vcon_cur in batch:
                            try:
                                # Check if file exists and is readable
                                if os.path.exists(vcon_cur.filename) and os.access(vcon_cur.filename, os.R_OK):
                                    # File is accessible - add to processing queue
                                    with with_blocking_time(stats_queue):
                                        vcons_ready_queue.put(vcon_cur)
                                    
                                    stats.count(stats_queue)
                                    stats.bytes(stats_queue, vcon_cur.size)
                                else:
                                    # File doesn't exist or isn't readable
                                    print(f"File not accessible: {vcon_cur.filename}")
                                    vcon.mark_vcon_as_invalid(vcon_cur)
                                    
                            except Exception as e:
                                print(f"Error verifying {vcon_cur.filename}: {e}")
                                vcon.mark_vcon_as_invalid(vcon_cur)
                                
                        batch_end = time.time()
                        batch_time = batch_end - batch_start
                        
                        # Optional: print batch processing stats
                        files_per_sec = len(batch) / max(batch_time, 0.001)
                        #print(f"Verified {len(batch)} files in {batch_time:.2f}s ({files_per_sec:.1f} files/s)")
            else:
                with with_blocking_time(stats_queue):
                    dont_overwhelm_server()

    except ShutdownException:
        dump_thread_stacks()
    finally:
        stats.stop(stats_queue)

# Keep old function for compatibility, but redirect to NFS version
def reserver(sftp_url, vcons_ready_queue, stats_queue):
    """Reserve vcons - now works with NFS, ignores sftp_url parameter"""
    print("Using NFS-based reserver (ignoring SFTP URL)")
    return reserver_nfs(vcons_ready_queue, stats_queue)

def start_process(sftp_url, vcons_ready_queue, stats_queue):
    return process.start_process(target=reserver, args=(sftp_url, vcons_ready_queue, stats_queue))
