#!/usr/bin/env python3
"""
WAV to OGG Vorbis converter using ffmpeg-python library.

Requirements:
- ffmpeg binary: sudo apt update && sudo apt install ffmpeg  
- ffmpeg-python library: pip install ffmpeg-python
"""
import random

import os
import subprocess
import hashlib
import shutil
import threading
import time
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
try:
    import ffmpeg
except ImportError:
    print("❌ ffmpeg-python library not found!")
    print("Please install it with: pip install ffmpeg-python")
    exit(1)

from mongo_utils import db, _db_semaphore
from vcon_class import Vcon

# Target directories to save converted files
TARGET_DRIVES = [
    "/media/500-ssd-0",
    "/media/500-hdd-0", 
    "/media/1800-hdd-0"
]

SOURCE_DRIVE = "/media/10900-hdd-0"

# Minimum free space required (in bytes) - 1GB safety margin
MIN_FREE_SPACE = 1 * 1024 * 1024 * 1024  # 1GB

# Global variable to store available drives (checked once at startup)
AVAILABLE_DRIVES = []

# Thread-safe counter for progress tracking
progress_lock = threading.Lock()
total_processed = 0

# Work queue for batch processing
work_queue = queue.Queue()
BATCH_SIZE = 3000  # Reserve 100 vcons at a time

def get_available_space(drive_path):
    """Get available space on a drive in bytes"""
    try:
        _, _, free_bytes = shutil.disk_usage(drive_path)
        return free_bytes
    except Exception as e:
        print(f"⚠️ Could not check space for {drive_path}: {e}")
        return 0

def initialize_available_drives():
    """Check drive space once at startup and cache available drives"""
    global AVAILABLE_DRIVES
    
    print("🔍 Checking target drives (one-time check)...")
    AVAILABLE_DRIVES = []
    
    for drive in TARGET_DRIVES:
        if not os.path.exists(drive):
            print(f"❌ {drive} does not exist")
            continue
            
        free_space = get_available_space(drive)
        
        if free_space >= MIN_FREE_SPACE:
            print(f"✅ {drive}: {free_space / 1024 / 1024 / 1024:.2f} GB free")
            AVAILABLE_DRIVES.append(drive)
        else:
            print(f"❌ {drive}: {free_space / 1024 / 1024 / 1024:.2f} GB free (insufficient)")
    
    if not AVAILABLE_DRIVES:
        print("❌ No drives have sufficient free space!")
        return False
    
    print(f"✅ Will use {len(AVAILABLE_DRIVES)} drive(s) for conversions")
    return True

def get_target_drive_by_hash(filename):
    """Distribute files across available target drives using filename hash"""
    if not AVAILABLE_DRIVES:
        return None

    return AVAILABLE_DRIVES[random.randint(0, len(AVAILABLE_DRIVES) - 1)]

def ensure_target_directory_exists(target_path):
    """Create target directory if it doesn't exist"""
    target_dir = os.path.dirname(target_path)
    os.makedirs(target_dir, exist_ok=True)

def convert_wav_to_ogg(source_path, target_path):
    """Convert WAV file to OGG Vorbis using ffmpeg-python library"""
    try:
        # Ensure target directory exists
        ensure_target_directory_exists(target_path)
        
        print(f"Converting {target_path}")
        
        # Use ffmpeg-python to convert WAV to OGG Vorbis
        stream = ffmpeg.input(source_path)
        stream = ffmpeg.output(
            stream, 
            target_path,
            acodec='libvorbis',
            **{'qscale:a': '5'}  # Quality level 5 (good balance of size/quality)
        )
        
        # Run the conversion silently
        ffmpeg.run(stream, overwrite_output=True, quiet=True)
        
        return True, None
            
    except ffmpeg.Error as e:
        error_msg = f"FFmpeg conversion error: {e}"
        return False, error_msg
    except Exception as e:
        error_msg = f"Error during conversion: {e}"
        return False, error_msg

def reserve_batch_vcons():
    """Reserve a batch of vcons atomically and add them to work queue"""
    query = {
        "done": True,
        "corrupt": {"$ne": True},
        "converting": {"$ne": True},  # Not already being converted
        "dialog.0.filename": {
            "$regex": f"^{SOURCE_DRIVE}/.*\\.wav$",
            "$options": "i"
        }
    }
    
    update = {"$set": {"converting": True}}
    
    reserved_count = 0
    
    with _db_semaphore:
        # Find and mark vcons as being converted atomically
        for _ in range(BATCH_SIZE):
            vcon_dict = db.find_one_and_update(
                query,
                update,
                return_document=True
            )
            
            if vcon_dict:
                vcon = Vcon.from_dict(vcon_dict)
                # Check if file exists before adding to queue
                if os.path.exists(vcon.filename):
                    work_queue.put(vcon)
                    reserved_count += 1
                else:
                    # Mark as corrupt if file doesn't exist
                    db.update_one(
                        {"_id": vcon.uuid},
                        {"$set": {"corrupt": True, "converting": False}}
                    )
            else:
                # No more vcons available
                break
    
    if reserved_count > 0:
        print(f"📦 Reserved {reserved_count} vcons for processing")
    
    return reserved_count

def get_vcon_from_queue():
    """Get a vcon from the work queue (non-blocking)"""
    try:
        return work_queue.get_nowait()
    except queue.Empty:
        return None

def update_vcon_filename(vcon, new_filename):
    """Update the vcon filename in the database and clear converting flag"""
    try:
        with _db_semaphore:
            result = db.update_one(
                {"_id": vcon.uuid},
                {"$set": {"dialog.0.filename": new_filename}, "$unset": {"converting": ""}}
            )
        
        if result.modified_count > 0:
            return True
        else:
            print(f"❌ Failed to update database (no documents modified)")
            # Clear converting flag even if update failed
            db.update_one({"_id": vcon.uuid}, {"$unset": {"converting": ""}})
            return False
            
    except Exception as e:
        print(f"❌ Error updating database: {e}")
        # Clear converting flag on error
        try:
            db.update_one({"_id": vcon.uuid}, {"$unset": {"converting": ""}})
        except:
            pass
        return False

def process_one_vcon():
    """Process one vcon: convert WAV to OGG, update database, remove original"""
    
    vcon = get_vcon_from_queue()
    
    if not vcon:
        # No work in queue
        return False
    
    original_filename = vcon.filename
    #print(f"🎯 Found vcon: {vcon.uuid}")
    #print(f"📁 Original file: {original_filename}")
    
    # Check if original file exists
    if not os.path.exists(original_filename):
        print(f"❌ Original file does not exist: {original_filename}")
        # Clear converting flag and mark as corrupt
        try:
            with _db_semaphore:
                db.update_one(
                    {"_id": vcon.uuid},
                    {"$set": {"corrupt": True}, "$unset": {"converting": ""}}
                )
        except:
            pass
        return False
    
    # Generate new filename path
    original_path = Path(original_filename)
    relative_path = original_path.relative_to(SOURCE_DRIVE)
    
    # Change extension to .ogg
    new_relative_path = relative_path.with_suffix('.ogg')
    
    # Choose target drive from cached available drives
    target_drive = get_target_drive_by_hash(str(relative_path))
    if not target_drive:
        print("❌ No available drives for conversion")
        return False
        
    new_filename = os.path.join(target_drive, str(new_relative_path))
    
    # Convert the file
    success, error_msg = convert_wav_to_ogg(original_filename, new_filename)
    
    if not success:
        print(f"❌ Conversion failed: {error_msg}")
        # Clear converting flag and mark as corrupt
        try:
            with _db_semaphore:
                db.update_one(
                    {"_id": vcon.uuid},
                    {"$set": {"corrupt": True}, "$unset": {"converting": ""}}
                )
        except:
            pass
        return False
    
    # Check new file exists
    if not os.path.exists(new_filename):
        print(f"❌ Converted file not found: {new_filename}")
        return False
    
    # Update database
    if not update_vcon_filename(vcon, new_filename):
        print("❌ Database update failed, keeping original file")
        try:
            os.remove(new_filename)
            print("🗑️ Removed converted file due to database failure")
        except:
            pass
        return False
    
    # Remove original file
    try:
        os.remove(original_filename)
    except Exception as e:
        pass  # Ignore removal errors
    
    return True

def increment_progress():
    """Thread-safe progress counter increment"""
    global total_processed
    with progress_lock:
        total_processed += 1
        if total_processed % 100 == 0:  # Show progress every 100 files
            print(f"✅ Processed {total_processed} files...")
        return total_processed

def process_vcon_worker():
    """Worker function for thread pool - processes one vcon"""
    success = process_one_vcon()
    if success:
        return increment_progress()
    return None

def cleanup_converting_flags():
    """Clear any remaining converting flags on exit"""
    try:
        with _db_semaphore:
            result = db.update_many(
                {"converting": True},
                {"$unset": {"converting": ""}}
            )
            if result.modified_count > 0:
                print(f"🧹 Cleaned up {result.modified_count} converting flags")
    except Exception as e:
        print(f"⚠️ Error cleaning up: {e}")

def main():
    global total_processed
    total_processed = 0  # Reset counter
    
    print("🎵 WAV to OGG Vorbis Converter for VCONs (14 Threads)")
    print("=" * 50)
    print("This script will continuously:")
    print("1. Find done vcons with WAV files")
    print("2. Check available disk space on target drives") 
    print("3. Convert WAV to OGG Vorbis format (14 parallel threads)")
    print("4. Save to a drive with sufficient free space")
    print("5. Update the database")
    print("6. Remove the original WAV file")
    print("7. Repeat until no more WAV files found")
    print()
    
    # Check if ffmpeg is available
    try:
        # Test that ffmpeg-python can access ffmpeg by checking version
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("✅ ffmpeg binary is available")
        print("✅ ffmpeg-python library is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ ffmpeg binary is not available. Please install it first:")
        print("   sudo apt update && sudo apt install ffmpeg")
        return False
    
    # Initialize available drives (one-time check)
    print()
    if not initialize_available_drives():
        return False
    
    # Check source drive exists
    if os.path.exists(SOURCE_DRIVE):
        print(f"✅ Source drive {SOURCE_DRIVE} exists")
    else:
        print(f"❌ Source drive {SOURCE_DRIVE} does not exist")
        return False
    
    print()
    print("🚀 Starting parallel conversion process with 14 threads...")
    print("   Press Ctrl+C to stop at any time")
    print()
    
    # Process files with thread pool
    MAX_WORKERS = 14
    
    try:
        # Reserve initial batch of work
        print("📦 Reserving initial batch of vcons...")
        reserve_batch_vcons()
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit initial batch of work
            futures = []
            for _ in range(MAX_WORKERS):
                future = executor.submit(process_vcon_worker)
                futures.append(future)
            
            while True:
                # Check if we need to reserve more work
                if work_queue.qsize() < MAX_WORKERS:
                    reserved = reserve_batch_vcons()
                    if reserved == 0:
                        print("💡 No more vcons available to reserve")
                
                # Check completed futures and submit new work
                new_futures = []
                active_work = False
                
                for future in futures:
                    if future.done():
                        try:
                            result = future.result()
                            # Always submit new work regardless of result
                            new_future = executor.submit(process_vcon_worker)
                            new_futures.append(new_future)
                        except Exception as e:
                            print(f"⚠️ Thread error: {e}")
                            # Submit replacement work
                            new_future = executor.submit(process_vcon_worker)
                            new_futures.append(new_future)
                    else:
                        # Keep the future that's still running
                        new_futures.append(future)
                        active_work = True
                
                futures = new_futures
                
                # If no active work and no work in queue, we're done
                if not active_work and work_queue.empty():
                    # Wait a bit for any remaining futures to complete
                    all_done = True
                    for future in futures:
                        if not future.done():
                            all_done = False
                            break
                    
                    if all_done:
                        break
                
                # Brief pause
                time.sleep(0.5)
                    
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user (Ctrl+C)")
        print(f"📊 Processed {total_processed} files before interruption")
    
    # Clean up any remaining converting flags
    print("\n🧹 Cleaning up...")
    cleanup_converting_flags()
    
    print("\n" + "=" * 60)
    print("🎉 CONVERSION COMPLETE!")
    print(f"📊 Total files processed: {total_processed}")
    
    if total_processed == 0:
        print("💡 No WAV files found to convert. All done!")
    else:
        print(f"🎵 Successfully converted {total_processed} WAV files to OGG format")
        print("💾 Database updated and original WAV files removed")
    
    print("=" * 60)
    return total_processed > 0

if __name__ == "__main__":
    main() 