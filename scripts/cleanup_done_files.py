#!/usr/bin/env python3
"""
Clean up files for done vcons by scanning the filesystem first.
This script walks through /media/10900-hdd-0/ and deletes files that correspond to done vcons.
"""

import os
import sys
import time
from typing import List
sys.path.append('.')

from mongo_utils import db
from vcon_class import Vcon

def cleanup_done_files():
    """Clean up files by scanning filesystem and checking against done vcons"""
    
    print("Starting filesystem-based cleanup of files for done vcons...")
    
    # Multiple directories to scan based on actual file locations
    target_directories = [
        "/media/10900-hdd-0/",
        "/media/500-hdd-0/",
        "/media/500-ssd-0/", 
        "/media/1800-hdd-0/"
    ]
    
    # Check which directories exist
    existing_directories = []
    for directory in target_directories:
        if os.path.exists(directory):
            existing_directories.append(directory)
            print(f"✓ Found directory: {directory}")
        else:
            print(f"✗ Directory not found: {directory}")
    
    if not existing_directories:
        print("Error: No target directories exist")
        return 0
    
    print(f"\nScanning {len(existing_directories)} directories...")
    print()
    
    # Statistics
    total_files_scanned = 0
    files_deleted = 0
    files_skipped_not_done = 0
    files_skipped_no_vcon = 0
    files_error = 0
    bytes_freed = 0
    
    start_time = time.time()
    
    try:
        # Walk through all files in each target directory
        for directory in existing_directories:
            print(f"Scanning directory: {directory}")
            
            for root, dirs, files in os.walk(directory):
                for filename in files:
                    total_files_scanned += 1
                    file_path = os.path.join(root, filename)
                    
                    # Extract basename (filename without extension)
                    basename = os.path.splitext(filename)[0]
                    
                    if not basename:  # Skip files with no basename
                        continue
                    
                    # Check if there's a vcon with this basename that is done
                    try:
                        vcon_doc = db.find_one(
                            {"basename": basename, "done": True},
                            {"_id": 1, "done": 1, "corrupt": 1}
                        )
                        
                        if vcon_doc:
                            # Found a done vcon with this basename - delete the file
                            try:
                                file_size = os.path.getsize(file_path)
                                os.remove(file_path)
                                files_deleted += 1
                                bytes_freed += file_size
                                
                                if files_deleted <= 10:  # Show first few deletions
                                    print(f"Deleted: {file_path}")
                                elif files_deleted % 1000 == 0:  # Show progress every 1000 deletions
                                    elapsed = time.time() - start_time
                                    rate = total_files_scanned / elapsed if elapsed > 0 else 0
                                    print(f"Progress: Scanned {total_files_scanned:,} files | "
                                          f"Deleted {files_deleted:,} | "
                                          f"Rate: {rate:.1f} files/sec | "
                                          f"Freed: {format_bytes(bytes_freed)}")
                                    
                            except OSError as e:
                                files_error += 1
                                print(f"Error deleting {file_path}: {e}")
                        else:
                            # Check if there's any vcon with this basename (regardless of done status)
                            any_vcon = db.find_one(
                                {"basename": basename},
                                {"_id": 1, "done": 1}
                            )
                            
                            if any_vcon:
                                # Vcon exists but not done
                                files_skipped_not_done += 1
                            else:
                                # No vcon found with this basename
                                files_skipped_no_vcon += 1
                                
                    except Exception as e:
                        files_error += 1
                        print(f"Database error for {file_path}: {e}")
                    
                    # Show periodic progress updates
                    if total_files_scanned % 10000 == 0:
                        elapsed = time.time() - start_time
                        rate = total_files_scanned / elapsed if elapsed > 0 else 0
                        print(f"Scanned {total_files_scanned:,} files | "
                              f"Deleted {files_deleted:,} | "
                              f"Rate: {rate:.1f} files/sec | "
                              f"Freed: {format_bytes(bytes_freed)}")
    
    except KeyboardInterrupt:
        print("\nOperation interrupted by user.")
    except Exception as e:
        print(f"Error during cleanup: {e}")
    
    # Final statistics
    elapsed = time.time() - start_time
    print(f"\nCleanup completed in {elapsed:.1f} seconds")
    print(f"Total files scanned: {total_files_scanned:,}")
    print(f"Files deleted (done vcons): {files_deleted:,}")
    print(f"Files skipped (vcon not done): {files_skipped_not_done:,}")
    print(f"Files skipped (no vcon found): {files_skipped_no_vcon:,}")
    print(f"Files with errors: {files_error:,}")
    print(f"Disk space freed: {format_bytes(bytes_freed)}")
    
    if total_files_scanned > 0:
        delete_percentage = (files_deleted / total_files_scanned) * 100
        print(f"Deletion rate: {delete_percentage:.1f}% of scanned files")
    
    return files_deleted

def format_bytes(bytes_count):
    """Format bytes in human readable format"""
    if bytes_count == 0:
        return "0 B"
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_count < 1024.0:
            return f"{bytes_count:.1f} {unit}"
        bytes_count /= 1024.0
    
    return f"{bytes_count:.1f} PB"

if __name__ == "__main__":
    print("WARNING: This will delete files in /media/10900-hdd-0/ that correspond to done vcons!")
    print("This operation cannot be undone!")
    print()
    
    # Ask for confirmation
    confirm = input("Are you sure you want to proceed? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Operation cancelled.")
        sys.exit(0)
    
    deleted_count = cleanup_done_files()
    print(f"\nOperation completed. Deleted {deleted_count:,} files.") 