import logging
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

def send_results(transcribed_vcons_queue: Queue, stats_queue: Queue):
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
                    # Send batch to database
                    vcon_utils.update_vcons_on_db_bulk(batch)
                    
                    # Update stats for all vcons in batch
                    for vcon in batch:
                        print(f"Sending {vcon.transcript()}")
                        stats.count(stats_queue)
                        stats.bytes(stats_queue, vcon.size)
                        stats.duration(stats_queue, vcon.duration)
                    
                    # Reset batch
                    batch = []
                    last_batch_time = current_time
                    
            except queue.Empty:
                # Timeout occurred, check if we have pending items
                if batch:
                    current_time = time.time()
                    if (current_time - last_batch_time) >= batch_timeout:
                        # Send remaining items in batch
                        vcon_utils.update_vcons_on_db_bulk(batch)
                        
                        # Update stats for all vcons in batch
                        for vcon in batch:
                            stats.count(stats_queue)
                            stats.bytes(stats_queue, vcon.size)
                            stats.duration(stats_queue, vcon.duration)
                        
                        # Reset batch
                        batch = []
                        last_batch_time = current_time
                
    except ShutdownException:
        # Send any remaining items in batch before shutdown
        if batch:
            vcon_utils.update_vcons_on_db_bulk(batch)
            for vcon in batch:
                stats.count(stats_queue)
                stats.bytes(stats_queue, vcon.size)
                stats.duration(stats_queue, vcon.duration)
        pass

def start_process(transcribed_vcons_queue: Queue, stats_queue: Queue):
    return process.start_process(target=send_results, args=(transcribed_vcons_queue, stats_queue))
