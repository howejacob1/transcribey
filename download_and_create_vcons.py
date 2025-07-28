#!/usr/bin/env python3
"""
Script to incrementally download S3 objects and create vcons.

This script:
1. Lists S3 objects in batches from vol1-eon through vol8-eon buckets  
2. For each object, checks vcon status by basename
3. If vcon exists and done + local file exists → delete local file
4. If vcon exists and not done → ensure file downloaded (compare by size)
5. If no vcon exists → download file and create vcon
6. Uses aggressive parallel downloads (up to 512 connections)
7. Retries downloads 5 times on failure
8. Only creates vcons for successfully downloaded files
9. Prints one line per file with action and running data rate
"""

import os
import sys
import subprocess
import tempfile
import hashlib
# import concurrent.futures  # Removed - no longer using threading
import time
from pathlib import Path
from typing import Set, List, Dict, Tuple, Optional
import json
import threading
from collections import defaultdict

# Add current directory to path for imports
sys.path.append('.')

from mongo_utils import db
from vcon_class import Vcon
from vcon_utils import insert_one, get_by_basename, update_vcon_on_db

# Configuration
S3_ENDPOINT = "https://nyc3.digitaloceanspaces.com"
BUCKET_PREFIXES = [f"vol{i}-eon" for i in range(8, 0, -1)]  # vol8-eon down to vol1-eon
LOCAL_BASE_DIR = "/media/10900-hdd-0"
BATCH_SIZE = 6000  # Number of S3 objects to process in each batch
MAX_PARALLEL_DOWNLOADS = 1024  # Reduced from 512 to avoid timeouts
DOWNLOAD_RETRIES = 3  # Reduced from 5 to fail faster
S3_LIST_BATCH_SIZE = 1000  # Process per hour folder - much smaller batches

# Test mode - set to True to process only a few files for debugging
TEST_MODE = os.environ.get('TEST_MODE', 'false').lower() == 'true'
TEST_FILES_LIMIT = 50  # Only process this many files in test mode

# s5cmd configuration for aggressive downloads
S5CMD_CONFIG = {
    'endpoint-url': S3_ENDPOINT,
    'no-verify-ssl': True,
    'retry-count': str(DOWNLOAD_RETRIES),
    'numworkers': str(MAX_PARALLEL_DOWNLOADS)
}

# Global statistics with thread safety
stats_lock = threading.Lock()
global_stats = {
    'files_processed': 0,
    'files_downloaded': 0,
    'files_deleted': 0,
    'vcons_created': 0,
    'bytes_downloaded': 0,
    'bytes_deleted': 0,
    'download_errors': 0,
    'vcon_errors': 0,
    'start_time': None,
    'last_heartbeat': None
}

def print_heartbeat(operation: str):
    """Print simple progress"""
    pass  # Removed verbose heartbeat

def setup_s5cmd():
    """Install s5cmd if not available"""
    try:
        result = subprocess.run(['s5cmd', 'version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ s5cmd available: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("s5cmd not found. Please install s5cmd first.")
    return False

def extract_basename_from_s3_key(s3_key: str) -> str:
    """Extract basename from S3 key (filename without extension)"""
    filename = os.path.basename(s3_key)
    basename = os.path.splitext(filename)[0]
    return basename

def get_s3_key_to_local_path(s3_key: str) -> str:
    """Convert S3 key to local path, maintaining full directory structure"""
    # For vol1-eon/vol1-eon/Freeswitch1/2025-07-03/06/file.wav
    # We want to keep the full path: /media/10900-hdd-0/vol1-eon/Freeswitch1/2025-07-03/06/file.wav
    return os.path.join(LOCAL_BASE_DIR, s3_key)

def get_running_data_rate() -> float:
    """Calculate running average data rate in MB/s"""
    with stats_lock:
        if global_stats['start_time'] is None:
            return 0.0
        
        elapsed = time.time() - global_stats['start_time']
        if elapsed <= 0:
            return 0.0
        
        total_bytes = global_stats['bytes_downloaded'] + global_stats['bytes_deleted']
        return (total_bytes / (1024 * 1024)) / elapsed

def format_bytes(size_bytes: int) -> str:
    """Format bytes in human readable format"""
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f}{size_names[i]}"

def print_file_result(basename: str, action: str, bytes_affected: int, success: bool, error: str = None):
    """Print one simple line per file"""
    status = "OK" if success else "ERR"
    size_text = format_bytes(bytes_affected) if bytes_affected > 0 else ""
    error_text = f" ({error[:30]})" if error else ""
    print(f"{status} {action} {basename[:50]} {size_text}{error_text}")

def list_directories(bucket: str, path: str) -> List[str]:
    """List directories in a given S3 path"""
    cmd = [
        's5cmd',
        '--endpoint-url', S3_ENDPOINT,
        '--no-sign-request',
        'ls',
        f's3://{bucket}/{path}'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return []
        
        directories = []
        lines = result.stdout.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line.endswith('/') and 'DIR' in line:
                # Extract directory name from "                                  DIR  Freeswitch1/"
                parts = line.split()
                if len(parts) >= 2:
                    dir_name = parts[-1].rstrip('/')
                    directories.append(dir_name)
        
        return directories
        
    except Exception as e:
        print(f"Error listing directories in {bucket}/{path}: {e}")
        return []

def list_files_in_hour_folder(bucket: str, hour_path: str) -> List[Tuple[str, str, int]]:
    """List all files in a specific hour folder (all are audio files)"""
    cmd = [
        's5cmd',
        '--endpoint-url', S3_ENDPOINT,
        '--no-sign-request',
        '--no-verify-ssl',  # Speed up SSL
        '--numworkers', '16',  # Optimized parallelism for listing
        'ls',
        # Remove --show-fullpath to get size information
        f's3://{bucket}/{hour_path}*'  # List all files in hour folder
    ]
    
    try:
        # Reduce timeout for hour folder listing to 20s to fail faster
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if result.returncode != 0:
            return []
        
        objects = []
        lines = result.stdout.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Parse standard s5cmd ls output: "2023/01/01 00:00:00    1234567 filename.wav"
            # Format: date time size filename
            parts = line.split()
            if len(parts) >= 4:
                try:
                    # Extract size (3rd element from the end: date, time, size, filename)
                    size = int(parts[-2])
                    filename = parts[-1]
                    
                    # Construct full S3 key from hour_path and filename
                    key = f"{hour_path}{filename}"
                    
                    # Skip if it's a directory
                    if not key.endswith('/'):
                        objects.append((bucket, key, size))
                except (ValueError, IndexError):
                    # If parsing fails, fall back to size 0
                    if len(parts) >= 1:
                        filename = parts[-1]
                        key = f"{hour_path}{filename}"
                        if not key.endswith('/'):
                            objects.append((bucket, key, 0))
                    
        return objects
        
    except subprocess.TimeoutExpired:
        print(f"❌ S3_TIMEOUT: Listing {bucket}/{hour_path} timed out after 20s")
        return []
    except Exception as e:
        print(f"❌ S3_ERROR: Error listing files in {bucket}/{hour_path}: {e}")
        return []

def process_hour_folder(bucket: str, freeswitch: str, date: str, hour: str) -> int:
    """Process all files in a single hour folder"""
    hour_path = f"{bucket}/{freeswitch}/{date}/{hour}/"
    
    # List files in this hour folder
    hour_objects = list_files_in_hour_folder(bucket, hour_path)
    
    if not hour_objects:
        return 0
    
    # In test mode, limit the number of files processed
    if TEST_MODE and len(hour_objects) > TEST_FILES_LIMIT:
        hour_objects = hour_objects[:TEST_FILES_LIMIT]
    
    # Process this hour's files in batches
    files_processed = 0
    for batch_start in range(0, len(hour_objects), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(hour_objects))
        batch = hour_objects[batch_start:batch_end]
        
        batch_start_time = time.time()
        
        # First pass: check vcon status and collect downloads needed
        download_requests = []
        file_actions = {}  # basename -> (action, local_path, etc.)
        
        for bucket, key, size in batch:
            basename = extract_basename_from_s3_key(key)
            local_path = get_s3_key_to_local_path(key)
            
            try:
                # Check vcon status
                vcon_status = check_vcon_status(basename)
                
                # Check if local file exists
                local_exists = os.path.exists(local_path)
                local_size = os.path.getsize(local_path) if local_exists else 0
                
                if vcon_status['exists'] and vcon_status['done'] and not vcon_status['corrupt']:
                    # Check if filename field is set, try to remove file
                    ensure_vcon_filename(basename, key)
                    if local_exists:
                        try:
                            os.remove(local_path)
                            file_actions[basename] = ('deleted_local', local_size, True, None)
                            with stats_lock:
                                global_stats['files_deleted'] += 1
                                global_stats['bytes_deleted'] += local_size
                        except Exception as e:
                            file_actions[basename] = ('deleted_local', 0, False, f"Failed to delete: {e}")
                    else:
                        file_actions[basename] = ('skipped_done', 0, True, None)
                        
                elif vcon_status['exists'] and vcon_status['corrupt']:
                    # Redownload file, set filename, unmark corrupt and done
                    download_requests.append((bucket, key, local_path, size))
                    file_actions[basename] = ('download_uncorrupt', local_path, vcon_status, key)
                    
                elif vcon_status['exists'] and not vcon_status['done']:
                    # Just redownload
                    download_requests.append((bucket, key, local_path, size))
                    file_actions[basename] = ('download_existing', local_path, vcon_status, key)
                        
                elif not vcon_status['exists']:
                    # Add new vcon and redownload file
                    download_requests.append((bucket, key, local_path, size))
                    file_actions[basename] = ('download_new', local_path, vcon_status, key)
                else:
                    file_actions[basename] = ('skipped_done', 0, True, None)
                    
            except Exception as e:
                file_actions[basename] = ('error', 0, False, f"Unexpected error: {e}")
        
        # Batch download all files that need downloading
        download_results = {}
        if download_requests:
            download_results = download_files_batch(download_requests)
            
            # Update stats for successful downloads
            for bucket, key, local_path, size in download_requests:
                basename = extract_basename_from_s3_key(key)
                if download_results.get(basename, False):
                    file_size = os.path.getsize(local_path) if os.path.exists(local_path) else 0
                    with stats_lock:
                        global_stats['files_downloaded'] += 1
                        global_stats['bytes_downloaded'] += file_size
                else:
                    with stats_lock:
                        global_stats['download_errors'] += 1
        
        # Second pass: process results and create vcons
        for bucket, key, size in batch:
            basename = extract_basename_from_s3_key(key)
            
            if basename not in file_actions:
                continue
                
            action_type, local_path_or_size, success_or_vcon_status, extra = file_actions[basename]
            
            try:
                result = {
                    'basename': basename,
                    'action': 'none',
                    'success': True,
                    'bytes_affected': 0,
                    'error': None
                }
                
                if action_type == 'deleted_local':
                    result['action'] = 'deleted_local'
                    result['bytes_affected'] = local_path_or_size
                    result['success'] = success_or_vcon_status
                    result['error'] = extra
                    
                elif action_type == 'download_uncorrupt':
                    local_path = local_path_or_size
                    s3_key = extra
                    if download_results.get(basename, False):
                        # Set filename field and unmark corruption
                        ensure_vcon_filename(basename, s3_key)
                        if unmark_vcon_corrupt_and_done(basename):
                            result['action'] = 'downloaded_uncorrupted'
                            result['bytes_affected'] = os.path.getsize(local_path) if os.path.exists(local_path) else 0
                        else:
                            result['success'] = False
                            result['error'] = "Downloaded but failed to unmark corruption"
                    else:
                        result['success'] = False
                        result['error'] = "Download failed for corrupted vcon"
                        
                elif action_type == 'download_existing':
                    local_path = local_path_or_size
                    s3_key = extra
                    if download_results.get(basename, False):
                        # Set filename field for existing vcon
                        ensure_vcon_filename(basename, s3_key)
                        result['action'] = 'downloaded'
                        result['bytes_affected'] = os.path.getsize(local_path) if os.path.exists(local_path) else 0
                    else:
                        result['success'] = False
                        result['error'] = "Download failed after retries"
                        
                elif action_type == 'download_new':
                    local_path = local_path_or_size
                    s3_key = extra
                    if download_results.get(basename, False):
                        # Create vcon for new file
                        if create_vcon_for_file(basename, s3_key):
                            result['action'] = 'downloaded_and_vcon_created'
                            result['bytes_affected'] = os.path.getsize(local_path) if os.path.exists(local_path) else 0
                            with stats_lock:
                                global_stats['vcons_created'] += 1
                        else:
                            result['action'] = 'downloaded'
                            result['bytes_affected'] = os.path.getsize(local_path) if os.path.exists(local_path) else 0
                            result['success'] = False
                            result['error'] = "Downloaded but failed to create vcon"
                            with stats_lock:
                                global_stats['vcon_errors'] += 1
                    else:
                        result['success'] = False
                        result['error'] = "Download failed, vcon not created"
                        
                elif action_type == 'skipped_done':
                    result['action'] = 'skipped_done'
                    
                elif action_type == 'error':
                    result['success'] = False
                    result['error'] = extra
                
                # Print result
                print_file_result(
                    result['basename'],
                    result['action'],
                    result['bytes_affected'],
                    result['success'],
                    result['error']
                )
                
                files_processed += 1
                
                # In test mode, check if we've hit the global limit
                if TEST_MODE and global_stats['files_processed'] >= TEST_FILES_LIMIT:
                    return files_processed
                    
                with stats_lock:
                    global_stats['files_processed'] += 1
                    
            except KeyboardInterrupt:
                print(f"\n🛑 INTERRUPTED: Stopping batch processing...")
                raise
            except Exception as e:
                files_processed += 1
        
        batch_time = time.time() - batch_start_time
        #print(f"      ✅ Batch completed in {batch_time:.1f}s")
    
    return files_processed

def discover_and_process_s3_hierarchically(bucket: str) -> int:
    """Discover S3 structure hierarchically and process hour by hour"""
    total_files_processed = 0
    
    # Structure: vol1-eon/vol1-eon/Freeswitch1/2025-07-03/06/
    freeswitch_folders = list_directories(bucket, f"{bucket}/")
    
    for freeswitch in freeswitch_folders:
        freeswitch_path = f"{bucket}/{freeswitch}/"
        date_folders = list_directories(bucket, freeswitch_path)
        
        for date in date_folders:
            date_path = f"{freeswitch_path}{date}/"
            hour_folders = list_directories(bucket, date_path)
            
            for hour in hour_folders:
                try:
                    files_in_hour = process_hour_folder(bucket, freeswitch, date, hour)
                    total_files_processed += files_in_hour
                except Exception as e:
                    pass  # Ignore processing errors
    
    return total_files_processed

def check_vcon_status(basename: str) -> Dict:
    """Check vcon status for a basename"""
    try:
        # Add debug info for potential hangs - but only for every 100th file to avoid spam
        #(global_stats['files_processed'] % 100 == 0)
        debug_this = False
        # if debug_this:
            #print(f"🔍 DB_QUERY_START: Checking vcon status for {basename}...")
        
        existing_vcon = get_by_basename(basename)
        
        if existing_vcon:
            status = {
                'exists': True,
                'done': existing_vcon.get('done', False),
                'corrupt': existing_vcon.get('corrupt', False)
            }
            if debug_this:
                print(f"📋 VCON_STATUS: {basename} - exists=True, done={status['done']}, corrupt={status['corrupt']}")
            return status
        else:
            if debug_this:
                print(f"📋 VCON_STATUS: {basename} - exists=False")
            return {
                'exists': False,
                'done': False,
                'corrupt': False
            }
    except Exception as e:
        return {
            'exists': False,
            'done': False,
            'corrupt': False
        }

def mark_vcon_corrupt(basename: str) -> bool:
    """Mark a vcon as corrupt in the database"""
    try:
        db.update_one(
            {"basename": basename}, 
            {"$set": {"corrupt": True}}
        )
        # Assume success with async writes
        return True
    except Exception as e:
        return False

def unmark_vcon_corrupt_and_done(basename: str) -> bool:
    """Unmark a vcon as corrupt and done in the database"""
    try:
        db.update_one(
            {"basename": basename}, 
            {"$set": {"corrupt": False, "done": False}}
        )
        # Assume success with async writes
        return True
    except Exception as e:
        return False

def ensure_vcon_filename(basename: str, s3_key: str) -> bool:
    """Ensure existing vcon has filename field set to full local path"""
    try:
        local_path = get_s3_key_to_local_path(s3_key)  # Full local path
        existing_vcon = get_by_basename(basename)
        
        if existing_vcon and not existing_vcon.get('filename'):
            # Update vcon to add missing filename field (fire and forget)
            db.update_one(
                {"basename": basename},
                {"$set": {"filename": local_path}}
            )
            # Assume success with async writes
        return True
    except Exception as e:
        return False

def download_files_batch(download_requests: List[Tuple[str, str, str, int]]) -> Dict[str, bool]:
    """Download multiple files in a single s5cmd batch operation"""
    if not download_requests:
        return {}
    
    results = {}
    
    # Create a temporary file with all download commands
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as cmd_file:
        temp_file_path = cmd_file.name
        
        try:
            # Write all cp commands to the file
            for bucket, key, local_path, expected_size in download_requests:
                s3_url = f"s3://{bucket}/{key}"
                # Create directory for each file
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                # Remove existing file if it exists
                if os.path.exists(local_path):
                    os.remove(local_path)
                
                cmd_file.write(f"cp {s3_url} {local_path}\n")
            
            cmd_file.flush()
            
            # Run s5cmd with the batch file
            cmd = [
                's5cmd',
                '--endpoint-url', S3_ENDPOINT,
                '--no-sign-request',
                '--no-verify-ssl',
                '--numworkers', str(min(len(download_requests), MAX_PARALLEL_DOWNLOADS)),
                'run',
                temp_file_path
            ]
            
            pass  # Removed verbose batch download message
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout for batch
            )
            
            # Check results for each file
            for bucket, key, local_path, expected_size in download_requests:
                basename = extract_basename_from_s3_key(key)
                
                if os.path.exists(local_path):
                    actual_size = os.path.getsize(local_path)
                    
                    # Verify size if expected
                    if expected_size > 0 and actual_size != expected_size:
                        os.remove(local_path)
                        results[basename] = False
                    elif actual_size == 0:
                        os.remove(local_path)
                        results[basename] = False
                    else:
                        results[basename] = True
                else:
                    results[basename] = False
            
            # Ignore s5cmd errors for now
                        
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file_path)
            except:
                pass
    
    return results

def download_file_with_retries(bucket: str, key: str, local_path: str, expected_size: int = 0) -> bool:
    """Download a single file using batch download (for compatibility)"""
    download_requests = [(bucket, key, local_path, expected_size)]
    results = download_files_batch(download_requests)
    basename = extract_basename_from_s3_key(key)
    return results.get(basename, False)

def create_vcon_for_file(basename: str, s3_key: str) -> bool:
    """Create a new vcon for a file"""
    try:
        # Use the full local path as the filename reference
        local_path = get_s3_key_to_local_path(s3_key)
        vcon = Vcon.create_from_url(os.path.basename(s3_key))
        vcon.basename = basename
        vcon.filename = local_path
        
        # Get file size for the vcon
        local_path = get_s3_key_to_local_path(s3_key)
        if os.path.exists(local_path):
            vcon.size = os.path.getsize(local_path)
        else:
            return False
        
        # Insert vcon to database (fire and forget with async writes)
        try:
            insert_one(vcon)
            # Assume success with async writes
            return True
        except Exception as db_e:
            return False
            
    except Exception as e:
        return False

def process_s3_object(bucket: str, key: str, size: int) -> Dict:
    """Process a single S3 object according to the logic"""
    basename = extract_basename_from_s3_key(key)
    local_path = get_s3_key_to_local_path(key)
    
    result = {
        'basename': basename,
        'action': 'none',
        'success': True,
        'bytes_affected': 0,
        'error': None
    }
    
    try:
        # Check vcon status
        vcon_status = check_vcon_status(basename)
        
        # Check if local file exists
        local_exists = os.path.exists(local_path)
        local_size = os.path.getsize(local_path) if local_exists else 0
        
        if vcon_status['exists'] and vcon_status['done'] and not vcon_status['corrupt'] and local_exists:
            # Ensure filename field is set for existing vcon
            ensure_vcon_filename(basename, key)
            
            # Vcon exists and done and not corrupt + local file exists → delete local file
            try:
                os.remove(local_path)
                result['action'] = 'deleted_local'
                result['bytes_affected'] = local_size
                
                with stats_lock:
                    global_stats['files_deleted'] += 1
                    global_stats['bytes_deleted'] += local_size
                    
            except Exception as e:
                result['success'] = False
                result['error'] = f"Failed to delete local file: {e}"
                
        elif vcon_status['exists'] and vcon_status['corrupt']:
            # Ensure filename field is set for existing vcon
            ensure_vcon_filename(basename, key)
            
            # Vcon exists and is corrupt → download file, unmark as corrupt and done
            if download_file_with_retries(bucket, key, local_path, size):
                # Successfully downloaded, now unmark corruption
                if unmark_vcon_corrupt_and_done(basename):
                    result['action'] = 'downloaded_uncorrupted'
                    result['bytes_affected'] = os.path.getsize(local_path) if os.path.exists(local_path) else 0
                    
                    with stats_lock:
                        global_stats['files_downloaded'] += 1
                        global_stats['bytes_downloaded'] += result['bytes_affected']
                else:
                    result['success'] = False
                    result['error'] = "Downloaded but failed to unmark corruption"
            else:
                result['success'] = False
                result['error'] = "Download failed for corrupted vcon"
                
                with stats_lock:
                    global_stats['download_errors'] += 1
                    
        elif vcon_status['exists'] and not vcon_status['done']:
            # Ensure filename field is set for existing vcon
            ensure_vcon_filename(basename, key)
            
            # Vcon exists but not done → ensure file downloaded (compare by size)
            if not local_exists:
                # Download the file
                if download_file_with_retries(bucket, key, local_path, size):
                    result['action'] = 'downloaded'
                    result['bytes_affected'] = os.path.getsize(local_path) if os.path.exists(local_path) else 0
                    
                    with stats_lock:
                        global_stats['files_downloaded'] += 1
                        global_stats['bytes_downloaded'] += result['bytes_affected']
                else:
                    result['success'] = False
                    result['error'] = "Download failed after retries"
                    
                    with stats_lock:
                        global_stats['download_errors'] += 1
            else:
                # File exists, check size if we know it
                if size > 0 and local_size != size:
                    # Size mismatch → mark as corrupt, download, then unmark corrupt and done
                    if mark_vcon_corrupt(basename):
                        if download_file_with_retries(bucket, key, local_path, size):
                            if unmark_vcon_corrupt_and_done(basename):
                                result['action'] = 'redownloaded_fixed_corruption'
                                result['bytes_affected'] = os.path.getsize(local_path) if os.path.exists(local_path) else 0
                                
                                with stats_lock:
                                    global_stats['files_downloaded'] += 1
                                    global_stats['bytes_downloaded'] += result['bytes_affected']
                            else:
                                result['success'] = False
                                result['error'] = "Downloaded but failed to unmark corruption after size fix"
                        else:
                            result['success'] = False
                            result['error'] = "Re-download failed after marking corrupt"
                            
                            with stats_lock:
                                global_stats['download_errors'] += 1
                    else:
                        result['success'] = False
                        result['error'] = "Failed to mark as corrupt for size mismatch"
                else:
                    result['action'] = 'already_exists'
                    
        elif not vcon_status['exists']:
            # No vcon exists → download file and create vcon
            # Download file if not exists or size mismatch
            need_download = not local_exists
            if local_exists and size > 0 and local_size != size:
                need_download = True
            elif local_exists and local_size == 0:
                # Always re-download zero-byte files (likely corrupted)
                need_download = True
            
            if need_download:
                if download_file_with_retries(bucket, key, local_path, size):
                    result['action'] = 'downloaded'
                    result['bytes_affected'] = os.path.getsize(local_path) if os.path.exists(local_path) else 0
                    
                    with stats_lock:
                        global_stats['files_downloaded'] += 1
                        global_stats['bytes_downloaded'] += result['bytes_affected']
                else:
                    result['success'] = False
                    result['error'] = "Download failed, vcon not created"
                    
                    with stats_lock:
                        global_stats['download_errors'] += 1
                    return result
            
            # Create vcon only if file download was successful or file already exists
            if os.path.exists(local_path):
                if create_vcon_for_file(basename, key):
                    if result['action'] == 'none':
                        result['action'] = 'vcon_created'
                    else:
                        result['action'] += '_and_vcon_created'
                    
                    with stats_lock:
                        global_stats['vcons_created'] += 1
                else:
                    result['success'] = False
                    result['error'] = "Failed to create vcon"
                    
                    with stats_lock:
                        global_stats['vcon_errors'] += 1
            else:
                result['success'] = False
                result['error'] = "File not available for vcon creation"
        else:
            # Vcon exists and done, no local file - nothing to do
            result['action'] = 'skipped_done'
        
        with stats_lock:
            global_stats['files_processed'] += 1
            
    except Exception as e:
        result['success'] = False
        result['error'] = f"Unexpected error: {e}"
    
    return result

def print_progress_summary():
    """Print minimal progress summary"""
    with stats_lock:
        stats = global_stats.copy()
    print(f"Progress: {stats['files_processed']} processed, {stats['vcons_created']} vcons created, {stats['files_downloaded']} downloaded")

def main():
    """Main function to process S3 buckets incrementally"""
    if not setup_s5cmd():
        return
    
    # Add signal handling for graceful shutdown
    import signal
    
    def signal_handler(signum, frame):
        print("\n\n🛑 INTERRUPTED: Graceful shutdown...")
        # stop_persistent_s5cmd() # No longer needed
        print_progress_summary()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination
    
    # Check local directory
    if not os.path.exists(LOCAL_BASE_DIR):
        try:
            os.makedirs(LOCAL_BASE_DIR, exist_ok=True)
        except Exception as e:
            return 1
    
    print("Starting S3 processing...")
    
    # Initialize global timer
    with stats_lock:
        global_stats['start_time'] = time.time()
    
    # Process each bucket incrementally, hour by hour
    for bucket in BUCKET_PREFIXES:        
        try:
            # Process hierarchically - one hour folder at a time
            total_files = discover_and_process_s3_hierarchically(bucket)
        except Exception as e:
            pass
    
    # Stop persistent s5cmd process # No longer needed
    # stop_persistent_s5cmd()
    
    # Final summary
    with stats_lock:
        stats = global_stats.copy()
    
    print(f"Complete: {stats['files_processed']} processed, {stats['vcons_created']} vcons, {stats['files_downloaded']} downloads")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("Interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) 