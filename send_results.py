import logging
import os
import queue
import time
from torch.multiprocessing import Queue
from typing import List

import process
import settings
import stats
import vcon_utils
from process import ShutdownException, setup_signal_handlers
from stats import with_blocking_time
from vcon_class import Vcon

def send_and_process_vcon_batch(batch: List[Vcon], stats_queue: Queue):
    """Send batch to database and process: update stats and delete files for done, non-corrupt vcons"""
    # Send batch to database
    vcon_utils.update_vcons_on_db_bulk(batch)
    
    # Process each vcon in the batch
    for vcon in batch:
        print(f"Sending {vcon.transcript()[:140]}")
        stats.count(stats_queue)
        stats.bytes(stats_queue, vcon.size)
        stats.duration(stats_queue, vcon.duration)
        
        # Delete file if vcon is done and not corrupt
        if vcon.done and not vcon.corrupt and vcon.filename:
            try:
                # Handle old SFTP URLs vs new NFS local paths
                if vcon.filename.startswith('sftp://'):
                    # Cannot delete old SFTP URLs - print warning
                    print(f"Cannot delete SFTP URL: {vcon.filename}")
                else:
                    # Delete local/NFS files normally
                    os.remove(vcon.filename)
                    #print(f"Deleted file: {vcon.filename}")
            except FileNotFoundError:
                print(f"File already deleted or not found: {vcon.filename}")
            except Exception as e:
                print(f"Error deleting file {vcon.filename}: {e}")

def send_results(transcribed_vcons_queue: Queue, stats_queue: Queue):
    # Set process title for identification in nvidia-smi and ps
    try:
        from setproctitle import setproctitle
        import os
        setproctitle("transcribey-send_results")
        print(f"[PID {os.getpid()}] Set process title to: transcribey-send_results")
    except ImportError:
        print("setproctitle not available for send_results process")
    
    setup_signal_handlers()
    stats.start(stats_queue)
    
    # Batch configuration for better performance
    batch_size = settings.mongo_bulk_update_batch_size
    batch_timeout = settings.mongo_bulk_update_timeout_seconds
    
    batch = []
    last_batch_time = time.time()
    
    try:
        while True:
            try:
                # Try to get a vcon with short timeout for batching
                with with_blocking_time(stats_queue):
                    vcon_cur: Vcon = transcribed_vcons_queue.get(timeout=0.1)
                vcon_cur.processed_by = None
                vcon_cur.done = True
                batch.append(vcon_cur)
                
                # Check if we should send the batch
                current_time = time.time()
                should_send_batch = (
                    len(batch) >= batch_size or 
                    (current_time - last_batch_time) >= batch_timeout
                )
                
                if should_send_batch:
                    send_and_process_vcon_batch(batch, stats_queue)
                    
                    # Reset batch
                    batch = []
                    last_batch_time = current_time
                    
            except queue.Empty:
                # Timeout occurred, check if we have pending items
                if batch:
                    current_time = time.time()
                    if (current_time - last_batch_time) >= batch_timeout:
                        send_and_process_vcon_batch(batch, stats_queue)
                        
                        # Reset batch
                        batch = []
                        last_batch_time = current_time
                
    except ShutdownException:
        # Send any remaining items in batch before shutdown
        if batch:
            send_and_process_vcon_batch(batch, stats_queue)
        pass

def start_process(transcribed_vcons_queue: Queue, stats_queue: Queue):
    return process.start_process(target=send_results, args=(transcribed_vcons_queue, stats_queue))
