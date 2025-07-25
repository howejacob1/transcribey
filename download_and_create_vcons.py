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
from vcon_utils import insert_one, get_by_basename

# Configuration
S3_ENDPOINT = "https://nyc3.digitaloceanspaces.com"
BUCKET_PREFIXES = [f"vol{i}-eon" for i in range(8, 0, -1)]  # vol8-eon down to vol1-eon
LOCAL_BASE_DIR = "/media/10900-hdd-0"
BATCH_SIZE = 50  # Number of S3 objects to process in each batch - reduced for faster iteration
MAX_PARALLEL_DOWNLOADS = 32  # Reduced from 512 to avoid timeouts
DOWNLOAD_RETRIES = 3  # Reduced from 5 to fail faster
S3_LIST_BATCH_SIZE = 512  # Process per hour folder - much smaller batches

# s5cmd configuration for aggressive downloads
S5CMD_CONFIG = {
    'endpoint-url': S3_ENDPOINT,
    'no-verify-ssl': False,
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
    """Print periodic heartbeat to show script is alive"""
    current_time = time.time()
    with stats_lock:
        if global_stats['last_heartbeat'] is None or current_time - global_stats['last_heartbeat'] > 30:
            elapsed = current_time - global_stats['start_time'] if global_stats['start_time'] else 0
            
            # Calculate rate directly to avoid nested lock acquisition
            if elapsed <= 0:
                rate = 0.0
            else:
                total_bytes = global_stats['bytes_downloaded'] + global_stats['bytes_deleted']
                rate = (total_bytes / (1024 * 1024)) / elapsed
                
            print(f" HEARTBEAT: {operation} - {elapsed:.0f}s elapsed, {rate:.1f}MB/s avg, {global_stats['files_processed']} files processed")
            global_stats['last_heartbeat'] = current_time

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
    """Print one line per file with action and running data rate"""
    data_rate = get_running_data_rate()
    
    # Format action with emoji and status
    if not success:
        status_emoji = "❌"
        action_text = f"{action}_ERROR"
    elif action == "deleted_local":
        status_emoji = "🗑️"
        action_text = "DELETED_LOCAL"
    elif action == "downloaded":
        status_emoji = "⬇️"
        action_text = "DOWNLOADED"
    elif action == "redownloaded":
        status_emoji = "🔄"
        action_text = "REDOWNLOADED"
    elif action == "vcon_created":
        status_emoji = "📝"
        action_text = "VCON_CREATED"
    elif action.endswith("_and_vcon_created"):
        status_emoji = "⬇️📝"
        action_text = action.replace("_and_vcon_created", "").upper() + "+VCON"
    elif action == "already_exists":
        status_emoji = "✅"
        action_text = "EXISTS"
    elif action == "skipped_done":
        status_emoji = "✅"
        action_text = "DONE_SKIP"
    else:
        status_emoji = "?"
        action_text = action.upper()
    
    # Format bytes affected
    bytes_text = format_bytes(bytes_affected) if bytes_affected > 0 else ""
    
    # Build output line
    parts = [
        status_emoji,
        action_text,
        basename[:60],  # Truncate long basenames
        f"{data_rate:.1f}MB/s"
    ]
    
    if bytes_text:
        parts.insert(-1, bytes_text)
    
    if error:
        parts.append(f"({error[:50]})")  # Truncate long errors
    
    print(" ".join(parts))

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
        '--show-fullpath',  # Faster output format
        f's3://{bucket}/{hour_path}*'  # List all files in hour folder
    ]
    
    try:
        # Reduce timeout for hour folder listing to 20s to fail faster
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if result.returncode != 0:
            print(f"⚠️ S3_LIST_FAILED: {bucket}/{hour_path} returned code {result.returncode}")
            if result.stderr:
                print(f"    Stderr: {result.stderr.strip()}")
            return []
        
        objects = []
        lines = result.stdout.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # With --show-fullpath, we just get the S3 path: s3://bucket/path/file.wav
            s3_path = line
            if s3_path.startswith('s3://'):
                # Extract key from s3://bucket/key format
                key = s3_path[len(f's3://{bucket}/'):]
                # Skip if it's a directory (shouldn't happen with * but just in case)
                if not key.endswith('/'):
                    objects.append((bucket, key, 0))  # Size will be determined during download if needed
                    
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
    
    print(f"    📂 Listing files in {freeswitch}/{date}/{hour}...")
    print_heartbeat(f"Listing {freeswitch}/{date}/{hour}")
    start_time = time.time()
    
    # List files in this hour folder
    hour_objects = list_files_in_hour_folder(bucket, hour_path)
    
    list_time = time.time() - start_time
    
    if not hour_objects:
        print(f"    📁 {freeswitch}/{date}/{hour}: 0 files (listed in {list_time:.1f}s)")
        return 0
        
    print(f"    📁 {freeswitch}/{date}/{hour}: {len(hour_objects)} files (listed in {list_time:.1f}s)")
    
    # Process this hour's files in small batches
    files_processed = 0
    for batch_start in range(0, len(hour_objects), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(hour_objects))
        batch = hour_objects[batch_start:batch_end]
        
        batch_start_time = time.time()
        #print(f"      🔄 Processing batch {batch_start//BATCH_SIZE + 1}/{(len(hour_objects)-1)//BATCH_SIZE + 1} ({len(batch)} files)...")
        #print_heartbeat(f"Processing batch {batch_start//BATCH_SIZE + 1} in {freeswitch}/{date}/{hour}")
        
        # Process batch sequentially (no threading to avoid database hangs)
        completed_count = 0
        for i, (bucket, key, size) in enumerate(batch):
            try:
                #print(f"      📄 Processing file {i+1}/{len(batch)}: {os.path.basename(key)}")
                result = process_s3_object(bucket, key, size)
                print_file_result(
                    result['basename'],
                    result['action'],
                    result['bytes_affected'],
                    result['success'],
                    result['error']
                )
                files_processed += 1
                completed_count += 1
            except KeyboardInterrupt:
                print(f"\n🛑 INTERRUPTED: Stopping batch processing...")
                raise
            except Exception as e:
                print(f"❌ PROCESS_ERROR: {e}")
                files_processed += 1
        
        batch_time = time.time() - batch_start_time
        #print(f"      ✅ Batch completed in {batch_time:.1f}s ({completed_count}/{len(batch)} succeeded)")
    
    return files_processed

def discover_and_process_s3_hierarchically(bucket: str) -> int:
    """Discover S3 structure hierarchically and process hour by hour"""
    total_files_processed = 0
    
    # Structure: vol1-eon/vol1-eon/Freeswitch1/2025-07-03/06/
    print(f"  Discovering Freeswitch folders in {bucket}...")
    freeswitch_folders = list_directories(bucket, f"{bucket}/")
    print(f"  Found {len(freeswitch_folders)} Freeswitch folders: {freeswitch_folders[:5]}...")
    
    for freeswitch in freeswitch_folders:
        freeswitch_path = f"{bucket}/{freeswitch}/"
        
        print(f"  📂 Processing {freeswitch}...")
        date_folders = list_directories(bucket, freeswitch_path)
        
        for date in date_folders:
            date_path = f"{freeswitch_path}{date}/"
            
            # Discover hour folders
            hour_folders = list_directories(bucket, date_path)
            print(f"    📅 {date}: {len(hour_folders)} hours")
            
            for hour in hour_folders:
                try:
                    files_in_hour = process_hour_folder(bucket, freeswitch, date, hour)
                    total_files_processed += files_in_hour
                except Exception as e:
                    print(f"❌ Error processing {freeswitch}/{date}/{hour}: {e}")
                
                # Print progress summary every few hours
                if total_files_processed > 0 and total_files_processed % 100 == 0:
                    print_progress_summary()
    
    return total_files_processed

def check_vcon_status(basename: str) -> Dict:
    """Check vcon status for a basename"""
    try:
        # Add debug info for potential hangs
        #print(f"🔍 DB_QUERY_START: Checking vcon status for {basename}...")
        db_start = time.time()
        existing_vcon = get_by_basename(basename)
        db_time = time.time() - db_start
        
        #print(f"✅ DB_QUERY_DONE: Query completed for {basename} in {db_time:.1f}s")
        # Log slow database queries
        if db_time > 5.0:
            print(f"⚠️ SLOW_DB: Query for {basename} took {db_time:.1f}s")
        
        if existing_vcon:
            return {
                'exists': True,
                'done': existing_vcon.get('done', False),
                'corrupt': existing_vcon.get('corrupt', False)
            }
        else:
            return {
                'exists': False,
                'done': False,
                'corrupt': False
            }
    except Exception as e:
        print(f"❌ DB_ERROR: Error checking vcon for {basename}: {e}")
        return {
            'exists': False,
            'done': False,
            'corrupt': False
        }

def download_file_with_retries(bucket: str, key: str, local_path: str, expected_size: int = 0) -> bool:
    """Download file with retries using s5cmd"""
    s3_url = f"s3://{bucket}/{key}"
    
    # Create directory
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    for attempt in range(DOWNLOAD_RETRIES):
        try:
            print_heartbeat(f"Downloading {os.path.basename(key)} (attempt {attempt + 1})")
            
            cmd = [
                's5cmd',
                '--endpoint-url', S3_ENDPOINT,
                '--no-sign-request',
                '--numworkers', str(min(16, MAX_PARALLEL_DOWNLOADS)),  # Limit per-file workers
                'cp',
                s3_url,
                local_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                # Verify download
                if os.path.exists(local_path):
                    actual_size = os.path.getsize(local_path)
                    
                    # If expected size is known, verify it
                    if expected_size > 0 and actual_size != expected_size:
                        print(f"Size mismatch for {local_path}: expected {expected_size}, got {actual_size}")
                        if attempt < DOWNLOAD_RETRIES - 1:
                            os.remove(local_path)
                            continue
                        else:
                            return False
                    
                    return True
                else:
                    print(f"Download verification failed for {s3_url} (attempt {attempt + 1})")
            else:
                print(f"Download failed for {s3_url} (attempt {attempt + 1}): {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"❌ DOWNLOAD_TIMEOUT: {s3_url} timed out after 60s (attempt {attempt + 1})")
        except Exception as e:
            print(f"Error downloading {s3_url} (attempt {attempt + 1}): {e}")
        
        # Wait before retry
        if attempt < DOWNLOAD_RETRIES - 1:
            time.sleep(2 ** attempt)  # Exponential backoff
    
    print(f"❌ DOWNLOAD_FAILED: {s3_url} failed after {DOWNLOAD_RETRIES} attempts")
    return False

def create_vcon_for_file(basename: str, s3_key: str) -> bool:
    """Create a new vcon for a file"""
    try:
        # Use the S3 key as the filename reference
        filename = os.path.basename(s3_key)
        vcon = Vcon.create_from_url(filename)
        vcon.basename = basename
        vcon.filename = filename
        
        # Get file size for the vcon
        local_path = get_s3_key_to_local_path(s3_key)
        if os.path.exists(local_path):
            vcon.size = os.path.getsize(local_path)
        
        # Insert vcon to database (relies on MongoDB timeouts and retry logic)
        #print(f"💾 DB_INSERT_START: Inserting vcon for {basename}...")
        db_start = time.time()
        
        insert_one(vcon)
        
        db_time = time.time() - db_start
        #print(f"✅ DB_INSERT_DONE: Insert completed for {basename} in {db_time:.1f}s")
        if db_time > 5.0:
            #print(f"⚠️ SLOW_INSERT: Database insert for {basename} took {db_time:.1f}s")
            pass
        return True
            
    except Exception as e:
        print(f"❌ VCON_ERROR: Error creating vcon for {basename}: {e}")
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
        
        if vcon_status['exists'] and vcon_status['done'] and local_exists:
            # Vcon exists and done + local file exists → delete local file
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
                
        elif vcon_status['exists'] and not vcon_status['done']:
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
                    # Size mismatch, re-download
                    if download_file_with_retries(bucket, key, local_path, size):
                        result['action'] = 'redownloaded'
                        result['bytes_affected'] = os.path.getsize(local_path) if os.path.exists(local_path) else 0
                        
                        with stats_lock:
                            global_stats['files_downloaded'] += 1
                            global_stats['bytes_downloaded'] += result['bytes_affected']
                    else:
                        result['success'] = False
                        result['error'] = "Re-download failed after retries"
                        
                        with stats_lock:
                            global_stats['download_errors'] += 1
                else:
                    result['action'] = 'already_exists'
                    
        elif not vcon_status['exists']:
            # No vcon exists → download file and create vcon
            # Download file if not exists or size mismatch
            need_download = not local_exists
            if local_exists and size > 0 and local_size != size:
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
            
            # Create vcon only if file download was successful
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
    """Print current progress summary"""
    with stats_lock:
        stats = global_stats.copy()
    
    data_rate = get_running_data_rate()
    
    print(f"\n=== PROGRESS SUMMARY ===")
    print(f"Files processed: {stats['files_processed']}")
    print(f"Downloads: {stats['files_downloaded']} ({format_bytes(stats['bytes_downloaded'])})")
    print(f"Deletions: {stats['files_deleted']} ({format_bytes(stats['bytes_deleted'])})")
    print(f"VCons created: {stats['vcons_created']}")
    print(f"Errors: {stats['download_errors']} download, {stats['vcon_errors']} vcon")
    print(f"Average data rate: {data_rate:.1f} MB/s")
    print("=" * 25)

def main():
    """Main function to process S3 buckets incrementally"""
    if not setup_s5cmd():
        return
    
    # Add signal handling for graceful shutdown
    import signal
    
    def signal_handler(signum, frame):
        print("\n\n🛑 INTERRUPTED: Graceful shutdown...")
        print_progress_summary()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination
    
    print("S3 Incremental Download and VCon Management Tool")
    print("=" * 60)
    
    # Check local directory
    if not os.path.exists(LOCAL_BASE_DIR):
        print(f"Local directory {LOCAL_BASE_DIR} does not exist. Creating...")
        try:
            os.makedirs(LOCAL_BASE_DIR, exist_ok=True)
        except Exception as e:
            print(f"Failed to create {LOCAL_BASE_DIR}: {e}")
            return 1
    
    print(f"Local base directory: {LOCAL_BASE_DIR}")
    print(f"Max parallel downloads: {MAX_PARALLEL_DOWNLOADS}")
    print(f"Download retries: {DOWNLOAD_RETRIES}")
    print()
    print("Format: [STATUS] [ACTION] [BASENAME] [SIZE] [DATA_RATE] [ERROR]")
    print("=" * 60)
    print()
    
    # Initialize global timer
    with stats_lock:
        global_stats['start_time'] = time.time()
    
    # Process each bucket incrementally, hour by hour
    for bucket in BUCKET_PREFIXES:
        print(f"\n>>> Processing bucket: {bucket}")
        
        try:
            # Process hierarchically - one hour folder at a time
            total_files = discover_and_process_s3_hierarchically(bucket)
            print(f"  ✅ Completed {bucket}: {total_files} files processed")
        except Exception as e:
            print(f"  ❌ Error processing bucket {bucket}: {e}")
        
        # Print summary after each bucket
        print_progress_summary()
    
    # Final summary
    elapsed = time.time() - global_stats['start_time']
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    with stats_lock:
        stats = global_stats.copy()
    
    print(f"Total files processed: {stats['files_processed']}")
    print(f"Files downloaded: {stats['files_downloaded']} ({format_bytes(stats['bytes_downloaded'])})")
    print(f"Local files deleted: {stats['files_deleted']} ({format_bytes(stats['bytes_deleted'])})")
    print(f"VCons created: {stats['vcons_created']}")
    print(f"Download errors: {stats['download_errors']}")
    print(f"VCon errors: {stats['vcon_errors']}")
    print(f"Total time: {elapsed:.1f} seconds")
    
    if elapsed > 0:
        print(f"Average rates:")
        print(f"  File processing: {stats['files_processed']/elapsed:.1f} files/sec")
        if stats['bytes_downloaded'] > 0:
            print(f"  Download rate: {(stats['bytes_downloaded']/(1024*1024))/elapsed:.1f} MB/sec")
        print(f"  VCon creation: {stats['vcons_created']/elapsed:.1f} vcons/sec")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        print_progress_summary()
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 