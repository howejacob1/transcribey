import logging
import os
import time
from typing import List

import process
import settings
import stats
from process import ShutdownException, setup_signal_handlers
from utils import let_other_threads_run
from vcon_class import Vcon
from vcon_utils import is_audio_filename, get_by_basename, insert_one
from utils import dump_thread_stacks

def discover_local(base_path, stats_queue, print_status=False):
    """Discover audio files in local/NFS directories and create vcons"""    
    stats.start(stats_queue)
    count = 0
    consecutive_db_errors = 0
    max_consecutive_db_errors = 5
    
    # Overall speed tracking
    overall_start_time = time.time()
    total_bytes_processed = 0

    try:        
        if not os.path.exists(base_path):
            print(f"Error: Path {base_path} does not exist")
            return
            
        print(f"Scanning local directory: {base_path}")
        
        # Walk through all directories and files
        for root, dirs, files in os.walk(base_path):
            for filename in files:
                if is_audio_filename(filename):
                    full_path = os.path.join(root, filename)
                    
                    try:
                        # Get file size
                        file_size = os.path.getsize(full_path)
                        basename = os.path.splitext(filename)[0]
                        action = "unknown"
                        
                        # Add to total bytes processed
                        total_bytes_processed += file_size
                        
                        # Check if vcon with this basename already exists
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
                                        if os.path.exists(full_path):
                                            os.remove(full_path)
                                    except Exception as delete_error:
                                        action = f"delete_error: {delete_error}"
                                else:
                                    # Vcon exists but is not done - skip
                                    action = "skipped"
                                
                                consecutive_db_errors = 0  # Reset error counter on success
                            else:
                                # No vcon exists - proceed with creation
                                try:
                                    vcon = Vcon.create_from_url(filename)
                                    stats.bytes(stats_queue, file_size)
                                    vcon.size = file_size
                                    vcon.basename = basename
                                    # Use local path for filename field
                                    vcon.filename = full_path
                                    # Explicitly set done=False so it appears in database
                                    vcon.done = False
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
                                        print(f"{action} {filename} (overall: {overall_speed_mbps:.1f} MB/s)")
                                        print(f"Too many consecutive database errors ({consecutive_db_errors}). Stopping discovery.")
                                        break
                            
                        except Exception as e:
                            consecutive_db_errors += 1
                            action = f"db_error: {e}"
                            if consecutive_db_errors >= max_consecutive_db_errors:
                                overall_elapsed = time.time() - overall_start_time
                                overall_speed_mbps = (total_bytes_processed / (1024 * 1024)) / max(overall_elapsed, 0.001)
                                print(f"{action} {filename} (overall: {overall_speed_mbps:.1f} MB/s)")
                                print(f"Too many consecutive database errors ({consecutive_db_errors}). Stopping discovery.")
                                break
                        
                        # Print one line per file with action, filename, and overall speed
                        if print_status:
                            overall_elapsed = time.time() - overall_start_time
                            overall_speed_mbps = (total_bytes_processed / (1024 * 1024)) / max(overall_elapsed, 0.001)
                            relative_path = os.path.relpath(full_path, base_path)
                            print(f"{action} {relative_path} (overall: {overall_speed_mbps:.1f} MB/s)")
                            
                    except OSError as e:
                        if print_status:
                            relative_path = os.path.relpath(full_path, base_path)
                            print(f"file_error {relative_path}: {e}")
                        
        stats.stop(stats_queue)            
    except ShutdownException as e:
        print(f"Shutdown requested: {e}")
        dump_thread_stacks()
    except Exception as e:
        print(f"Unexpected error in discover: {type(e).__name__}: {e}")
        dump_thread_stacks()

# Keep old function for compatibility, but redirect to local discovery
def discover(url_or_path, stats_queue, print_status=False):
    """Discover audio files - now works with local paths instead of SFTP"""
    # If it looks like an SFTP URL, extract the path
    if url_or_path.startswith('sftp://'):
        # Parse SFTP URL to get local path
        import sftp
        parsed = sftp.parse_url(url_or_path) 
        local_path = parsed["path"]
    else:
        # Assume it's already a local path
        local_path = url_or_path
        
    return discover_local(local_path, stats_queue, print_status)

def start_process(url_or_path, stats_queue, print_status=False):
    """Start discovery process"""
    return process.start_process(target=discover, args=(url_or_path, stats_queue, print_status))
