#!/usr/bin/env python3

import os
import sys
import subprocess
import tempfile
import time
import threading
from typing import List, Dict, Tuple

sys.path.append('.')
from mongo_utils import db
from vcon_class import Vcon
from vcon_utils import insert_one, get_by_basename

# Configuration settings
S3_ENDPOINT = "https://nyc3.digitaloceanspaces.com"
BUCKETS = [f"vol{i}-eon" for i in range(1, 9)]
LOCAL_BASE = "/media/10900-hdd-0"
BATCH_SIZE = 6000
NUM_WORKERS = 64
TIMEOUT_SECONDS = 300

# Global statistics
stats = {
    'start_time': None,
    'total_files_processed': 0,
    'total_bytes_downloaded': 0,
    'lock': threading.Lock()
}

def extract_basename(s3_key: str) -> str:
    return os.path.splitext(os.path.basename(s3_key))[0]

def get_local_path(s3_key: str) -> str:
    # Strip bucket name from path: vol1-eon/Freeswitch1/... -> Freeswitch1/...
    path_parts = s3_key.split('/', 1)
    if len(path_parts) > 1:
        # Remove bucket prefix (vol1-eon, vol2-eon, etc.)
        local_path = path_parts[1]
    else:
        local_path = s3_key
    return os.path.join(LOCAL_BASE, local_path)

def get_sftp_url(s3_key: str) -> str:
    return f"sftp://bantaim@banidk0:22{get_local_path(s3_key)}"

def get_rate_stats() -> str:
    """Calculate current rates and return formatted string"""
    with stats['lock']:
        if not stats['start_time']:
            return ""
        
        elapsed = time.time() - stats['start_time']
        if elapsed < 1:
            return ""
        
        files_per_sec = stats['total_files_processed'] / elapsed
        mb_per_sec = (stats['total_bytes_downloaded'] / (1024 * 1024)) / elapsed
        
        return f"({files_per_sec:.1f} files/s, {mb_per_sec:.1f} MB/s)"

def update_stats(files_processed: int = 0, bytes_downloaded: int = 0):
    """Update global statistics"""
    with stats['lock']:
        stats['total_files_processed'] += files_processed
        stats['total_bytes_downloaded'] += bytes_downloaded

def run_s5cmd(args: List[str], timeout: int = 60) -> str:
    cmd = ['s5cmd', '--endpoint-url', S3_ENDPOINT, '--no-sign-request'] + args
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return result.stdout if result.returncode == 0 else ""

def list_s3_dirs(bucket: str, path: str) -> List[str]:
    output = run_s5cmd(['ls', f's3://{bucket}/{path}'])
    dirs = []
    for line in output.strip().split('\n'):
        if line.endswith('/') and 'DIR' in line:
            dirs.append(line.split()[-1].rstrip('/'))
    return dirs

def list_s3_files(bucket: str, path: str) -> List[Tuple[str, int]]:
    output = run_s5cmd(['ls', f's3://{bucket}/{path}*'], 120)
    files = []
    for line in output.strip().split('\n'):
        if not line.strip() or line.endswith('/'):
            continue
        parts = line.split()
        if len(parts) >= 4:
            try:
                size = int(parts[-2])
                filename = parts[-1]
                files.append((filename, size))
            except ValueError:
                continue
    return files

def fix_filename_if_needed(filename: str, s3_key: str) -> str:
    """Fix filename if it starts with sftp:// instead of being a local path"""
    if filename and filename.startswith('sftp://'):
        # Extract just the path from the SFTP URL
        from urllib.parse import urlparse
        parsed = urlparse(filename)
        return parsed.path
    return filename

def get_vcon_status(basename: str) -> Dict:
    vcon = get_by_basename(basename)
    if not vcon:
        return {'exists': False}
    
    filename = vcon.get('filename')
    needs_filename_fix = filename and filename.startswith('sftp://')
    
    return {
        'exists': True,
        'done': vcon.get('done', False),
        'corrupt': vcon.get('corrupt', False),
        'filename': filename,
        'needs_filename_fix': needs_filename_fix
    }

def get_vcon_statuses_batch(basenames: List[str]) -> Dict[str, Dict]:
    """Get vcon statuses for multiple basenames in a single query"""
    vcons = db.find({"basename": {"$in": basenames}})
    
    status_map = {}
    for vcon in vcons:
        basename = vcon.get('basename')
        if basename:
            filename = vcon.get('filename')
            needs_filename_fix = filename and filename.startswith('sftp://')
            
            status_map[basename] = {
                'exists': True,
                'done': vcon.get('done', False),
                'corrupt': vcon.get('corrupt', False),
                'filename': filename,
                'needs_filename_fix': needs_filename_fix
            }
    
    # Fill in missing ones
    for basename in basenames:
        if basename not in status_map:
            status_map[basename] = {'exists': False}
    
    return status_map

def download_batch(downloads: List[Tuple[str, str, str]]) -> List[str]:
    if not downloads:
        return []
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for bucket, s3_key, local_path in downloads:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            f.write(f"cp s3://{bucket}/{s3_key} {local_path}\n")
        
        cmd_file = f.name
    
    try:
        run_s5cmd(['--numworkers', str(NUM_WORKERS), 'run', cmd_file], TIMEOUT_SECONDS)
        successful = []
        total_bytes = 0
        for bucket, s3_key, local_path in downloads:
            if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                successful.append(extract_basename(s3_key))
                total_bytes += os.path.getsize(local_path)
        
        # Update download stats
        update_stats(bytes_downloaded=total_bytes)
        return successful
    finally:
        os.unlink(cmd_file)

def batch_update_vcons(updates: List[Tuple[str, Dict]]):
    """Batch update multiple vcons"""
    if not updates:
        return
    from pymongo import UpdateOne
    operations = [
        UpdateOne({"basename": basename}, {"$set": update_data}) 
        for basename, update_data in updates
    ]
    if operations:
        db.bulk_write(operations, ordered=False)

def batch_insert_vcons(vcons: List[Dict]):
    """Batch insert multiple vcons"""
    if not vcons:
        return
    db.insert_many(vcons, ordered=False)

def prepare_new_vcon(basename: str, s3_key: str, local_path: str) -> Dict:
    """Prepare vcon data for batch insertion"""
    try:
        filename = os.path.basename(s3_key)
        vcon = Vcon.create_from_url(filename)
        vcon.basename = basename
        vcon.filename = local_path  # Just the local path, not SFTP URL
        vcon.size = os.path.getsize(local_path)
        return vcon.__dict__
    except Exception:
        return None


def process_files(bucket: str, hour_path: str, files: List[Tuple[str, int]]):
    for i in range(0, len(files), BATCH_SIZE):
        batch = files[i:i + BATCH_SIZE]
        
        # Get all basenames for this batch
        basenames = [extract_basename(f"{hour_path}{filename}") for filename, size in batch]
        
        # Single DB query for all vcon statuses
        status_map = get_vcon_statuses_batch(basenames)
        
        # Collect operations for batching
        downloads_needed = []
        files_to_delete = []
        db_updates = []
        new_vcons = []
        file_actions = {}  # Track what action each file needs
        
        # Phase 1: Analyze what needs to be done
        for filename, size in batch:
            s3_key = f"{hour_path}{filename}"
            basename = extract_basename(s3_key)
            local_path = get_local_path(s3_key)
            status = status_map[basename]
            
            try:
                if status['exists'] and status['done'] and not status['corrupt']:
                    # Always fix filename if needed, then decide about file deletion
                    if status.get('needs_filename_fix', False):
                        db_updates.append((basename, {"filename": local_path}))
                        file_actions[basename] = 'fixed_filename'
                    
                    if os.path.exists(local_path):
                        files_to_delete.append((basename, local_path))
                        if file_actions.get(basename) == 'fixed_filename':
                            file_actions[basename] = 'fixed_filename_and_deleted'
                        else:
                            file_actions[basename] = 'delete_local'
                    else:
                        if basename not in file_actions:
                            file_actions[basename] = 'already_processed'
                        
                elif status['exists'] and status['corrupt']:
                    # Need to download and fix
                    downloads_needed.append((bucket, s3_key, local_path, basename, 'fix_corrupt'))
                    file_actions[basename] = 'fix_corrupt'
                    
                elif status['exists'] and not status['done']:
                    # Always fix filename if needed
                    if status.get('needs_filename_fix', False):
                        db_updates.append((basename, {"filename": local_path}))
                        file_actions[basename] = 'fixed_filename'
                    
                    # Check if download needed
                    local_size = os.path.getsize(local_path) if os.path.exists(local_path) else 0
                    if not os.path.exists(local_path) or (size > 0 and local_size != size):
                        downloads_needed.append((bucket, s3_key, local_path, basename, 'update_existing'))
                        if file_actions.get(basename) == 'fixed_filename':
                            file_actions[basename] = 'fixed_filename_and_download'
                        else:
                            file_actions[basename] = 'download_existing'
                    else:
                        if basename not in file_actions:
                            file_actions[basename] = 'exists_correct_size'
                        
                else:  # New file
                    local_size = os.path.getsize(local_path) if os.path.exists(local_path) else 0
                    need_download = not os.path.exists(local_path) or (size > 0 and local_size != size) or local_size == 0
                    if need_download:
                        downloads_needed.append((bucket, s3_key, local_path, basename, 'new_file'))
                        file_actions[basename] = 'download_and_create'
                    else:
                        # File exists, just create vcon
                        vcon_data = prepare_new_vcon(basename, s3_key, local_path)
                        if vcon_data:
                            new_vcons.append((basename, vcon_data))
                            file_actions[basename] = 'create_vcon_only'
                        else:
                            file_actions[basename] = 'create_vcon_failed'
                            
            except Exception as e:
                file_actions[basename] = f'error_{e}'
        
        # Phase 2: Execute downloads
        download_success = {}
        if downloads_needed:
            download_list = [(bucket, s3_key, local_path) for bucket, s3_key, local_path, _, _ in downloads_needed]
            successful_downloads = download_batch(download_list)
            
            # Track download results and update data structures
            for bucket, s3_key, local_path, basename, action_type in downloads_needed:
                download_success[basename] = basename in successful_downloads
                
                # Handle post-download actions
                if basename in successful_downloads:
                    if action_type == 'fix_corrupt':
                        # Add updates for fixed corrupt files
                        db_updates.append((basename, {
                            "filename": local_path,
                            "corrupt": False,
                            "done": False
                        }))
                    elif action_type == 'new_file':
                        # Prepare vcon data for new files
                        vcon_data = prepare_new_vcon(basename, s3_key, local_path)
                        if vcon_data:
                            new_vcons.append((basename, vcon_data))
        
        # Phase 3: Process all files and print results
        for filename, size in batch:
            basename = extract_basename(f"{hour_path}{filename}")
            action = file_actions.get(basename, 'unknown')
            
            try:
                if action == 'delete_local':
                    print(f"DELETED {basename} {get_rate_stats()}")
                    
                elif action == 'fixed_filename':
                    print(f"FIXED_FILENAME {basename} {get_rate_stats()}")
                    
                elif action == 'fixed_filename_and_deleted':
                    print(f"FIXED_FILENAME_AND_DELETED {basename} {get_rate_stats()}")
                    
                elif action == 'fixed_filename_and_download':
                    if download_success.get(basename, False):
                        print(f"FIXED_FILENAME_AND_DOWNLOADED {basename} {get_rate_stats()}")
                    else:
                        print(f"FIXED_FILENAME_DOWNLOAD_FAILED {basename} {get_rate_stats()}")
                    
                elif action == 'already_processed':
                    print(f"ALREADY_PROCESSED {basename} {get_rate_stats()}")
                    
                elif action == 'fix_corrupt':
                    if download_success.get(basename, False):
                        print(f"FIXED {basename} {get_rate_stats()}")
                    else:
                        print(f"FIX_FAILED {basename} {get_rate_stats()}")
                        
                elif action == 'download_existing':
                    if download_success.get(basename, False):
                        print(f"DOWNLOADED {basename} {get_rate_stats()}")
                    else:
                        print(f"DOWNLOAD_FAILED {basename} {get_rate_stats()}")
                        
                elif action == 'exists_correct_size':
                    print(f"EXISTS {basename} {get_rate_stats()}")
                    
                elif action == 'download_and_create':
                    if download_success.get(basename, False):
                        print(f"CREATED {basename} {get_rate_stats()}")
                    else:
                        print(f"CREATE_FAILED {basename} {get_rate_stats()}")
                        
                elif action == 'create_vcon_only':
                    print(f"VCON_CREATED {basename} {get_rate_stats()}")
                    
                elif action == 'create_vcon_failed':
                    print(f"VCON_FAILED {basename} {get_rate_stats()}")
                    
                elif action.startswith('error_'):
                    print(f"ERROR {basename}: {action[6:]} {get_rate_stats()}")
                    
                else:
                    print(f"UNKNOWN {basename} {get_rate_stats()}")
                
                update_stats(files_processed=1)
                
            except Exception as e:
                print(f"PRINT_ERROR {basename}: {e} {get_rate_stats()}")
                update_stats(files_processed=1)
        
        # Phase 4: Batch database operations (after printing)
        if db_updates:
            try:
                batch_update_vcons(db_updates)
            except Exception as e:
                print(f"BATCH_UPDATE_FAILED: {e} {get_rate_stats()}")
        
        if new_vcons:
            try:
                vcon_docs = [vcon_data for basename, vcon_data in new_vcons]
                batch_insert_vcons(vcon_docs)
            except Exception as e:
                print(f"BATCH_INSERT_FAILED: {e} {get_rate_stats()}")
                
        # Phase 5: Execute file deletions (after printing and DB operations)
        for basename, local_path in files_to_delete:
            try:
                os.remove(local_path)
            except Exception:
                pass  # Already printed the action, ignore delete errors

def process_hour(bucket: str, freeswitch: str, date: str, hour: str):
    hour_path = f"{bucket}/{freeswitch}/{date}/{hour}/"
    files = list_s3_files(bucket, hour_path)
    if files:
        print(f"Processing {hour_path} ({len(files)} files) {get_rate_stats()}")
        process_files(bucket, hour_path, files)

def process_bucket(bucket: str):
    print(f"Processing bucket: {bucket}")
    freeswitch_dirs = list_s3_dirs(bucket, f"{bucket}/")
    
    for freeswitch in freeswitch_dirs:
        date_dirs = list_s3_dirs(bucket, f"{bucket}/{freeswitch}/")
        for date in date_dirs:
            hour_dirs = list_s3_dirs(bucket, f"{bucket}/{freeswitch}/{date}/")
            for hour in hour_dirs:
                try:
                    process_hour(bucket, freeswitch, date, hour)
                except Exception as e:
                    print(f"Error processing {bucket}/{freeswitch}/{date}/{hour}: {e}")

def main():
    # Initialize statistics
    stats['start_time'] = time.time()
    
    os.makedirs(LOCAL_BASE, exist_ok=True)
    print(f"Starting S3 processing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    for bucket in BUCKETS:
        try:
            process_bucket(bucket)
        except Exception as e:
            print(f"Error processing bucket {bucket}: {e}")
    
    # Final statistics
    elapsed = time.time() - stats['start_time']
    print(f"\nCompleted in {elapsed:.1f}s - {get_rate_stats()}")

if __name__ == "__main__":
    main() 




