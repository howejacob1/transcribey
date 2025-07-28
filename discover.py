import logging
import os
from pprint import pprint
import time
from typing import List

import process
import settings
import sftp
import stats
from process import ShutdownException, setup_signal_handlers
from utils import let_other_threads_run
from vcon_class import Vcon
from vcon_utils import is_audio_filename, get_by_basename, insert_one
from utils import dump_thread_stacks

def discover(url, stats_queue, print_status=False):
    """Discover audio files and create vcons, with clean shutdown handling"""    
    stats.start(stats_queue)
    sftp_client: sftp.SFTPClient | None = None
    count = 0
    consecutive_db_errors = 0
    max_consecutive_db_errors = 5
    
    # Overall speed tracking
    overall_start_time = time.time()
    total_bytes_processed = 0

    try:        
        sftp_client, _ = sftp.connect_keep_trying(url)
        
        parsed = sftp.parse_url(url)
        path = parsed["path"]

        for filename_local, bytes in sftp.get_all_filenames(path, sftp_client):
            if is_audio_filename(filename_local):
                basename = os.path.splitext(os.path.basename(filename_local))[0]
                action = "unknown"
                
                # Add to total bytes processed
                total_bytes_processed += bytes
                
                # Check if vcon with this basename already exists (unique now)
                try:
                    existing_vcon_dict = get_by_basename(basename)
                    
                    # If vcon exists
                    if existing_vcon_dict:
                        # Remove _id field to avoid ObjectId serialization issues
                        if "_id" in existing_vcon_dict:
                            del existing_vcon_dict["_id"]
                        vcon_obj = Vcon.from_dict(existing_vcon_dict)
                        
                        if vcon_obj.done:
                            # Vcon is done - delete the file
                            action = "deleted"
                            try:
                                parsed = sftp.parse_url(url)
                                full_path = parsed["path"] + "/" + filename_local
                                if sftp.sftp_is_local(sftp_client):
                                    if os.path.exists(full_path):
                                        os.remove(full_path)
                                else:
                                    try:
                                        sftp_client.remove(full_path)
                                    except FileNotFoundError:
                                        pass  # File already deleted
                            except Exception as delete_error:
                                action = f"delete_error: {delete_error}"
                        else:
                            # Vcon exists but is not done - skip
                            action = "skipped"
                        
                        consecutive_db_errors = 0  # Reset error counter on success
                    else:
                        # No vcon exists - proceed with creation
                        try:
                            filename = filename_local
                            vcon = Vcon.create_from_url(filename)
                            stats.bytes(stats_queue, bytes)
                            vcon.size = bytes
                            vcon.basename = basename
                            # Use full path for filename field
                            parsed = sftp.parse_url(url)
                            full_path = parsed["path"] + "/" + filename_local
                            vcon.filename = full_path
                            stats.count(stats_queue)
                            count += 1
                            
                            # Insert one vcon at a time (simple, no batching)
                            insert_one(vcon)
                            consecutive_db_errors = 0  # Reset error counter on success
                            action = "added"
                            
                        except Exception as e:
                            consecutive_db_errors += 1
                            action = f"insert_error: {e}"
                            if consecutive_db_errors >= max_consecutive_db_errors:
                                overall_elapsed = time.time() - overall_start_time
                                overall_speed_mbps = (total_bytes_processed / (1024 * 1024)) / max(overall_elapsed, 0.001)
                                print(f"{action} {filename_local} (overall: {overall_speed_mbps:.1f} MB/s)")
                                print(f"Too many consecutive database errors ({consecutive_db_errors}). Stopping discovery.")
                                break
                    
                except Exception as e:
                    consecutive_db_errors += 1
                    action = f"db_error: {e}"
                    if consecutive_db_errors >= max_consecutive_db_errors:
                        overall_elapsed = time.time() - overall_start_time
                        overall_speed_mbps = (total_bytes_processed / (1024 * 1024)) / max(overall_elapsed, 0.001)
                        print(f"{action} {filename_local} (overall: {overall_speed_mbps:.1f} MB/s)")
                        print(f"Too many consecutive database errors ({consecutive_db_errors}). Stopping discovery.")
                        break
                
                # Print one line per file with action, filename, and overall speed
                overall_elapsed = time.time() - overall_start_time
                overall_speed_mbps = (total_bytes_processed / (1024 * 1024)) / max(overall_elapsed, 0.001)
                print(f"{action} {filename_local} (overall: {overall_speed_mbps:.1f} MB/s)")
        
        stats.stop(stats_queue)            
    except ShutdownException as e:
        print(f"Shutdown requested: {e}")
        dump_thread_stacks()
    except Exception as e:
        print(f"Unexpected error in discover: {type(e).__name__}: {e}")
        dump_thread_stacks()
    finally: 
        if sftp_client:
            try:
                sftp_client.close()
            except Exception as e:
                print(f"Error closing SFTP client: {e}")

def start_process(url, stats_queue, print_status=False):
    """Start discovery process"""
    return process.start_process(target=discover, args=(url, stats_queue, print_status))
