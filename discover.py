import sftp
import time
from time import perf_counter
from vcon_class import Vcon
from vcon_utils import insert_many_maybe_async, is_audio_filename
import logging
from settings import discover_batch_size
from typing import List
import stats
import process
from process import ShutdownException, block_until_threads_and_processes_finish, setup_signal_handlers

def discover(url, stats_queue):
    """Discover audio files and create vcons, with clean shutdown handling"""    
    threads = []
    def add_many(vcons: List[Vcon]):
        thread = insert_many_maybe_async(vcons)
        threads.append(thread)
    
    sftp_client: sftp.SFTPClient | None = None
    try:
        sftp_client = sftp.connect_keep_trying(url)
        parsed = sftp.parse_url(url)
        path = parsed["path"]
        vcons = []
        vcons_count = 0
        vcons_bytes = 0
        for filename, bytes in sftp.get_all_filenames(path, sftp_client):
            vcons_bytes += bytes
            stats.add(stats_queue, "vcons_bytes", vcons_bytes)
            if is_audio_filename(filename):
                vcons_count += 1
                stats.add(stats_queue, "vcons_count", vcons_count)
                vcon = Vcon.create_from_url(filename)
                vcon.size = bytes
                vcons.append(vcon)
                if len(vcons) > discover_batch_size:
                    add_many(vcons)
                    vcons = []
        # Process any remaining vcons
        if vcons:
            add_many(vcons)
            
        block_until_threads_and_processes_finish(threads)
        stats.add(stats_queue, "vcons_count", vcons_count)
        stats.add(stats_queue, "vcons_bytes", vcons_bytes)
    except ShutdownException as e:
        pass
    finally: 
        if sftp_client:
            sftp_client.close()
        stats.add(stats_queue, "stop_time", time.time())

def start_process(url, stats_queue):
    """Start discovery process"""
    stats.add(stats_queue, "start_time", time.time())
    return process.start_process(target=discover, args=(url, stats_queue))
