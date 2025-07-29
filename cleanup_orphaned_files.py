#!/usr/bin/env python3

import os
import sys
import time
from pathlib import Path

sys.path.append('.')
from vcon_utils import get_by_basename

# Configuration
BASE_DIR = "/media/10900-hdd-0"
DRY_RUN = True  # Set to False to actually delete files

def extract_basename_from_file(filepath: str) -> str:
    """Extract basename from file path (filename without extension)"""
    return os.path.splitext(os.path.basename(filepath))[0]

def should_keep_file(basename: str) -> bool:
    """Check if file should be kept (has completed vcon)"""
    try:
        vcon = get_by_basename(basename)
        if not vcon:
            return False
        
        # Keep file only if vcon is done and not corrupt
        return vcon.get('done', False) and not vcon.get('corrupt', False)
    except Exception:
        return False

def scan_and_cleanup():
    """Scan directory and remove orphaned files"""
    if not os.path.exists(BASE_DIR):
        print(f"Directory {BASE_DIR} does not exist")
        return
    
    stats = {
        'files_scanned': 0,
        'files_deleted': 0,
        'bytes_freed': 0,
        'start_time': time.time()
    }
    
    print(f"{'DRY RUN - ' if DRY_RUN else ''}Scanning {BASE_DIR} for orphaned files...")
    print("Looking for files without completed vcons (done=True, corrupt=False)")
    print()
    
    # Walk through all files
    for root, dirs, files in os.walk(BASE_DIR):
        for filename in files:
            filepath = os.path.join(root, filename)
            basename = extract_basename_from_file(filename)
            
            stats['files_scanned'] += 1
            
            # Show progress every 1000 files
            if stats['files_scanned'] % 1000 == 0:
                elapsed = time.time() - stats['start_time']
                rate = stats['files_scanned'] / elapsed if elapsed > 0 else 0
                print(f"Scanned {stats['files_scanned']} files ({rate:.1f} files/s)")
            
            try:
                if not should_keep_file(basename):
                    file_size = os.path.getsize(filepath)
                    
                    if DRY_RUN:
                        print(f"WOULD DELETE: {filepath} ({file_size} bytes)")
                    else:
                        os.remove(filepath)
                        print(f"DELETED: {filepath} ({file_size} bytes)")
                    
                    stats['files_deleted'] += 1
                    stats['bytes_freed'] += file_size
                    
            except Exception as e:
                print(f"ERROR processing {filepath}: {e}")
    
    # Final stats
    elapsed = time.time() - stats['start_time']
    mb_freed = stats['bytes_freed'] / (1024 * 1024)
    
    print()
    print(f"=== CLEANUP SUMMARY ===")
    print(f"Files scanned: {stats['files_scanned']}")
    print(f"Files {'would be ' if DRY_RUN else ''}deleted: {stats['files_deleted']}")
    print(f"Space {'would be ' if DRY_RUN else ''}freed: {mb_freed:.1f} MB")
    print(f"Time taken: {elapsed:.1f} seconds")
    
    if DRY_RUN:
        print()
        print("This was a DRY RUN. Set DRY_RUN = False to actually delete files.")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--execute":
        global DRY_RUN
        DRY_RUN = False
        print("EXECUTE MODE: Files will be actually deleted!")
        print()
    
    scan_and_cleanup()

if __name__ == "__main__":
    main() 