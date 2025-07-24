import logging
import time
import os
from pprint import pprint
from time import time
from typing import List

import process
import settings
import sftp
import stats
from process import ShutdownException, setup_signal_handlers
from utils import let_other_threads_run
from vcon_class import Vcon
from vcon_utils import is_audio_filename
from utils import dump_thread_stacks

def discover(url, stats_queue, print_status=False):
    """Discover audio files and create vcons, with clean shutdown handling"""    
    stats.start(stats_queue)
    sftp_client: sftp.SFTPClient | None = None
    count = 0

    try:        
        sftp_client, _ = sftp.connect_keep_trying(url)
        
        parsed = sftp.parse_url(url)
        path = parsed["path"]

        for filename_local, bytes in sftp.get_all_filenames(path, sftp_client):
            if is_audio_filename(filename_local):
                basename = os.path.splitext(os.path.basename(filename_local))[0]
                
                # Check if vcon with this basename already exists
                from vcon_utils import exists_by_basename
                if exists_by_basename(basename):
                    if print_status:
                        print(f"Skipping {filename_local} - vcon with basename {basename} already exists")
                    continue
                
                filename = filename_local
                vcon = Vcon.create_from_url(filename)
                stats.bytes(stats_queue, bytes)
                vcon.size = bytes
                vcon.basename = basename
                stats.count(stats_queue)
                count += 1
                
                # Insert one vcon at a time (simple, no batching)
                from vcon_utils import insert_one
                insert_one(vcon)
                
                if print_status:
                    print(f"Added vcon {count}: {basename}")
        
        stats.stop(stats_queue)            
    except ShutdownException as e:
        dump_thread_stacks()
    except Exception as e:
        print(f"Error: {e}")
        dump_thread_stacks()
    finally: 
        dump_thread_stacks()
        if sftp_client:
            sftp_client.close()

def start_process(url, stats_queue, print_status=False):
    """Start discovery process"""
    return process.start_process(target=discover, args=(url, stats_queue, print_status))
