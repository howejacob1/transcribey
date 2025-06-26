import logging
import time
from pprint import pprint
from time import time
from typing import List

import process
import settings
import sftp
import stats
from process import ShutdownException, block_until_threads_and_processes_finish, setup_signal_handlers
from settings import discover_batch_size
from utils import let_other_threads_run
from vcon_class import Vcon
from vcon_utils import insert_many_maybe_async, is_audio_filename
from utils import dump_thread_stacks

def discover(url, stats_queue):
    """Discover audio files and create vcons, with clean shutdown handling"""    
    stats.start(stats_queue)
    threads = []
    def add_many(vcons: List[Vcon]):
        thread = insert_many_maybe_async(vcons)
        threads.append(thread)
    sftp_client: sftp.SFTPClient | None = None
    count = 0
    try:
        sftp_client = sftp.connect_keep_trying(url)
        parsed = sftp.parse_url(url)
        path = parsed["path"]
        vcons = []
        for filename, bytes in sftp.get_all_filenames(path, sftp_client):
            if is_audio_filename(filename):
                vcon = Vcon.create_from_url(filename)
                stats.bytes(stats_queue, bytes)
                vcon.size = bytes
                #duration = vcon.size / (settings.sample_rate*2)
                stats.count(stats_queue)
                count += 1
                #print(f"Discovered {count} vcons")
                #stats.duration(stats_queue, duration)
                vcons.append(vcon)
                if len(vcons) > discover_batch_size:
                    add_many(vcons)
                    vcons = []
        # Process any remaining vcons
        if vcons:
            add_many(vcons)
        stats.stop(stats_queue)            
        block_until_threads_and_processes_finish(threads)
    except ShutdownException as e:
        dump_thread_stacks()
    except Exception as e:
        print(f"Error: {e}")
        dump_thread_stacks()
    finally: 
        if sftp_client:
            sftp_client.close()


def start_process(url, stats_queue):
    """Start discovery process"""
    return process.start_process(target=discover, args=(url, stats_queue))
