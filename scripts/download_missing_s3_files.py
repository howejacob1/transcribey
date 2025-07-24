#!/usr/bin/env python3
"""
Script to download S3 objects from Digital Ocean Spaces that don't exist as vcons in MongoDB.

This script:
1. Lists all objects from vol1-eon through vol8-eon buckets 
2. Extracts basenames from S3 object keys
3. Checks which basenames don't exist in MongoDB
4. Downloads missing files to /media/10900-hdd-0/ with proper directory structure
5. Handles checksums and skips existing files with matching checksums
"""

import os
import sys
import subprocess
import tempfile
import hashlib
import concurrent.futures
import time
from pathlib import Path
from typing import Set, List, Dict, Tuple
import json

# Add current directory to path for imports
sys.path.append('.')

from mongo_utils import db

# Configuration
S3_ENDPOINT = "https://nyc3.digitaloceanspaces.com"
BUCKET_PREFIXES = [f"vol{i}-eon" for i in range(1, 9)]  # vol1-eon through vol8-eon
LOCAL_BASE_DIR = "/media/10900-hdd-0"
BATCH_SIZE = 10000  # Number of files to process in each batch
MAX_WORKERS = 8  # Number of download threads
CHECKSUM_CACHE_FILE = "/tmp/s3_checksums.json"

# s5cmd configuration
S5CMD_CONFIG = {
    'endpoint-url': S3_ENDPOINT,
    'no-verify-ssl': False,
    'retry-count': '3'
}

def setup_s5cmd():
    """Install s5cmd if not available"""
    try:
        result = subprocess.run(['s5cmd', 'version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ s5cmd already available: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    # Check if .deb file exists and install
    deb_file = "s5cmd_2.3.0_linux_amd64.deb"
    if os.path.exists(deb_file):
        print(f"Installing s5cmd from {deb_file}...")
        try:
            subprocess.run(['sudo', 'dpkg', '-i', deb_file], check=True)
            print("✓ s5cmd installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to install s5cmd: {e}")
            return False
    else:
        print(f"s5cmd not found and {deb_file} not available")
        return False

def extract_basename_from_s3_key(s3_key: str) -> str:
    """
    Extract basename from S3 key.
    Example: vol1-eon/Freeswitch1/2025-07-03/15/4272_15028797517_993315038160020_2025-07-03_15:55:45.wav
    Returns: 4272_15028797517_993315038160020_2025-07-03_15:55:45
    """
    filename = os.path.basename(s3_key)
    basename = os.path.splitext(filename)[0]
    return basename

def get_s3_key_to_local_path(s3_key: str) -> str:
    """
    Convert S3 key to local path.
    Example: vol1-eon/Freeswitch1/2025-07-03/15/file.wav -> /media/10900-hdd-0/Freeswitch1/2025-07-03/15/file.wav
    """
    # Remove bucket prefix (vol1-eon/, vol2-eon/, etc.)
    parts = s3_key.split('/', 1)
    if len(parts) == 2:
        relative_path = parts[1]
    else:
        relative_path = s3_key
    
    return os.path.join(LOCAL_BASE_DIR, relative_path)

def list_s3_objects() -> List[Tuple[str, str, int]]:
    """
    List all audio objects from all buckets.
    Returns list of tuples: (bucket, key, size)
    """
    print("Listing S3 objects from all buckets...")
    all_objects = []
    
    for bucket in BUCKET_PREFIXES:
        print(f"  Listing objects from {bucket}... (this may take a while for large buckets)")
        
        cmd = [
            's5cmd',
            '--endpoint-url', S3_ENDPOINT,
            '--no-sign-request',
            'ls',
            '--show-fullpath',
            f's3://{bucket}/*'
        ]
        
        try:
            # No timeout - let it run as long as needed for huge buckets
            print(f"    Starting listing for {bucket}...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"    Error listing {bucket}: {result.stderr}")
                continue
            
            bucket_count = 0
            lines = result.stdout.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Parse --show-fullpath output: "s3://bucket/path/file.wav"
                if line.startswith('s3://'):
                    try:
                        # Extract bucket and key from s3://bucket/key
                        s3_path = line[5:]  # Remove 's3://'
                        if '/' in s3_path:
                            bucket_name, key = s3_path.split('/', 1)
                            
                            # Only include audio files
                            if key and key.lower().endswith(('.wav', '.mp3', '.ogg', '.m4a', '.flac')):
                                all_objects.append((bucket_name, key, 0))  # Size unknown with --show-fullpath
                                bucket_count += 1
                    except (ValueError, IndexError):
                        # Skip malformed lines
                        continue
            
            print(f"    Found {bucket_count} audio files in {bucket}")
                    
        except Exception as e:
            print(f"    Error processing {bucket}: {e}")
    
    print(f"Total audio files found: {len(all_objects)}")
    return all_objects

def get_vcon_status_batch(basenames: List[str]) -> Dict[str, Dict]:
    """Get vcon status for basenames from MongoDB in batch"""
    if not basenames:
        return {}
    
    try:
        status_map = {}
        
        # Due to MongoDB query limitations with nested array fields, 
        # query each basename individually for reliability
        for basename in basenames:
            doc = db.find_one(
                {"dialog.0.basename": basename},
                {"done": 1, "corrupt": 1, "_id": 1}
            )
            
            if doc:
                status_map[basename] = {
                    'exists': True,
                    'done': doc.get('done', False),
                    'corrupt': doc.get('corrupt', False),
                    '_id': doc.get('_id')
                }
            else:
                status_map[basename] = {
                    'exists': False,
                    'done': False,
                    'corrupt': False,
                    '_id': None
                }
                    
            return status_map
            
    except Exception as e:
        print(f"Database error checking basenames: {e}")
        # Return default "not exists" for all basenames on error
        return {basename: {'exists': False, 'done': False, 'corrupt': False, '_id': None} for basename in basenames}

def calculate_file_checksum(filepath: str) -> str:
    """Calculate MD5 checksum of a file"""
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception:
        return ""

def load_checksum_cache() -> Dict[str, str]:
    """Load cached checksums from file"""
    try:
        if os.path.exists(CHECKSUM_CACHE_FILE):
            with open(CHECKSUM_CACHE_FILE, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def save_checksum_cache(cache: Dict[str, str]):
    """Save checksums to cache file"""
    try:
        with open(CHECKSUM_CACHE_FILE, 'w') as f:
            json.dump(cache, f)
    except Exception:
        pass

def unmark_vcon_as_corrupt(vcon_id: str):
    """Remove corrupt flag from a vcon in MongoDB"""
    try:
        result = db.update_one(
            {"_id": vcon_id},
            {"$unset": {"corrupt": ""}}
        )
        return result.modified_count > 0
    except Exception as e:
        print(f"Error unmarking vcon {vcon_id} as corrupt: {e}")
        return False

def check_local_file_matches_s3(local_path: str, s3_size: int, s3_url: str, checksum_cache: Dict[str, str]) -> bool:
    """Check if local file exists and matches S3 file (size and checksum if available)"""
    if not os.path.exists(local_path):
        return False
    
    local_size = os.path.getsize(local_path)
    if local_size != s3_size:
        return False
    
    # Check checksum if we have it cached
    cached_checksum = checksum_cache.get(s3_url)
    if cached_checksum:
        local_checksum = calculate_file_checksum(local_path)
        return local_checksum == cached_checksum
    
    # If no cached checksum, assume match based on size
    return True

def download_file(bucket: str, key: str, size: int, checksum_cache: Dict[str, str], vcon_id: str = None, is_corrupt_vcon: bool = False) -> bool:
    """
    Download a single file from S3 to local filesystem.
    Returns True if downloaded, False if skipped.
    """
    local_path = get_s3_key_to_local_path(key)
    s3_url = f"s3://{bucket}/{key}"
    
    # Create directory structure
    local_dir = os.path.dirname(local_path)
    os.makedirs(local_dir, exist_ok=True)
    
    # For corrupt vcons, check if local file exists
    if is_corrupt_vcon and os.path.exists(local_path):
        # Local file exists - skip download and keep vcon marked as corrupt
        return False
    
    # Check if file already exists (for non-corrupt cases)
    if not is_corrupt_vcon and os.path.exists(local_path):
        # File exists - check checksum if we have it cached
        cached_checksum = checksum_cache.get(s3_url)
        if cached_checksum:
            local_checksum = calculate_file_checksum(local_path)
            if local_checksum == cached_checksum:
                return False  # Skip - file exists and checksums match
        else:
            return False  # Skip - file exists (no size check since size=0)
    
    # Download the file
    cmd = [
        's5cmd',
        '--endpoint-url', S3_ENDPOINT,
        '--no-sign-request',
        'cp',
        s3_url,
        local_path
    ]
    
    try:
        # Increase timeout for individual file downloads (5 minutes -> 15 minutes)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if result.returncode == 0:
            # Verify download
            if os.path.exists(local_path):
                # Calculate and cache checksum
                checksum = calculate_file_checksum(local_path)
                if checksum:
                    checksum_cache[s3_url] = checksum
                
                # If this was a corrupt vcon, unmark it as corrupt
                if is_corrupt_vcon and vcon_id:
                    if unmark_vcon_as_corrupt(vcon_id):
                        print(f"Unmarked vcon {vcon_id} as corrupt after successful download")
                
                return True
            return False
        else:
            print(f"Download failed for {s3_url}: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"Timeout downloading {s3_url}")
        return False
    except Exception as e:
        print(f"Error downloading {s3_url}: {e}")
        return False

def main():
    print("S3 Missing Files Downloader")
    print("=" * 50)
    
    # Setup s5cmd
    if not setup_s5cmd():
        print("Failed to setup s5cmd. Exiting.")
        return 1
    
    # Load checksum cache
    checksum_cache = load_checksum_cache()
    
    # List all S3 objects
    all_s3_objects = list_s3_objects()
    if not all_s3_objects:
        print("No S3 objects found. Exiting.")
        return 1
    
    print(f"\nProcessing {len(all_s3_objects)} S3 objects...")
    
    # Process in batches to avoid overwhelming MongoDB
    total_processed = 0
    total_downloaded = 0
    total_skipped = 0
    start_time = time.time()
    
    for batch_start in range(0, len(all_s3_objects), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(all_s3_objects))
        batch = all_s3_objects[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_start//BATCH_SIZE + 1}/{(len(all_s3_objects)-1)//BATCH_SIZE + 1} ({len(batch)} files)...")
        
        # Extract basenames for this batch
        batch_basenames = []
        basename_to_object = {}
        
        for bucket, key, size in batch:
            basename = extract_basename_from_s3_key(key)
            batch_basenames.append(basename)
            basename_to_object[basename] = (bucket, key, size)
        
        # Get vcon status for all basenames in this batch
        vcon_status_map = get_vcon_status_batch(batch_basenames)
        
        # Apply download logic based on vcon status
        files_to_download = []
        skip_no_vcon = 0
        skip_corrupt = 0
        skip_done_clean = 0
        
        for basename in batch_basenames:
            bucket, key, size = basename_to_object[basename]
            status = vcon_status_map.get(basename, {'exists': False, 'done': False, 'corrupt': False, '_id': None})
            
            should_download = False
            vcon_id = None
            is_corrupt_vcon = False
            
            if not status['exists']:
                # Case 1: No vcon exists - download if file missing/different checksum
                should_download = True
                skip_no_vcon += 1
            elif status['corrupt']:
                # Case 2: Vcon exists but marked corrupt - check if local file matches S3
                should_download = True  # Will be filtered in download_file based on local/S3 match
                vcon_id = status['_id']
                is_corrupt_vcon = True
                skip_corrupt += 1
            elif status['done'] and not status['corrupt']:
                # Case 3: Vcon exists, done, not corrupt - skip
                should_download = False
                skip_done_clean += 1
            else:
                # Case 4: Vcon exists but not done and not corrupt - download
                should_download = True
                vcon_id = status['_id']
            
            if should_download:
                files_to_download.append((bucket, key, size, vcon_id, is_corrupt_vcon))
        
        print(f"  Files to download: {len(files_to_download)}")
        print(f"    - No vcon in DB: {skip_no_vcon}")
        print(f"    - Corrupt vcons: {skip_corrupt}")
        print(f"  Skipping (done + clean): {skip_done_clean}")
        
        # Download files in parallel
        if files_to_download:
            downloaded_count = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_file = {
                    executor.submit(download_file, bucket, key, size, checksum_cache, vcon_id, is_corrupt_vcon): 
                    (bucket, key, size, vcon_id, is_corrupt_vcon)
                    for bucket, key, size, vcon_id, is_corrupt_vcon in files_to_download
                }
                
                for future in concurrent.futures.as_completed(future_to_file):
                    bucket, key, size, vcon_id, is_corrupt_vcon = future_to_file[future]
                    try:
                        downloaded = future.result()
                        if downloaded:
                            downloaded_count += 1
                            if downloaded_count % 10 == 0:
                                print(f"    Downloaded {downloaded_count}/{len(files_to_download)} files...")
                    except Exception as e:
                        print(f"    Error downloading {bucket}/{key}: {e}")
            
            total_downloaded += downloaded_count
            total_skipped += len(files_to_download) - downloaded_count
        
        total_processed += len(batch)
        
        # Save checksum cache periodically
        if batch_start % (BATCH_SIZE * 5) == 0:
            save_checksum_cache(checksum_cache)
        
        # Progress report
        elapsed = time.time() - start_time
        rate = total_processed / elapsed if elapsed > 0 else 0
        print(f"  Progress: {total_processed}/{len(all_s3_objects)} ({total_processed/len(all_s3_objects)*100:.1f}%) - {rate:.1f} files/sec")
    
    # Final save of checksum cache
    save_checksum_cache(checksum_cache)
    
    # Summary
    elapsed = time.time() - start_time
    print(f"\n" + "=" * 50)
    print(f"Summary:")
    print(f"  Total files processed: {total_processed}")
    print(f"  Files downloaded: {total_downloaded}")
    print(f"  Files skipped (already exist): {total_skipped}")
    print(f"  Total time: {elapsed:.1f} seconds")
    print(f"  Average rate: {total_processed/elapsed:.1f} files/sec")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1) 