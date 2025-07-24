#!/usr/bin/env python3
"""
Pipeline debugging script to help identify where processing stops after "returning 512 vcons"
"""
import sys
import time
import threading
from multiprocessing import Queue
sys.path.append('.')

import settings
from mongo_utils import db, _db_semaphore

def check_queue_sizes():
    """Check if any processes are alive and working"""
    print("\n=== QUEUE SIZE DEBUG ===")
    
    # Count reserved but not processed vcons
    with _db_semaphore:
        reserved_count = db.count_documents({"processed_by": settings.hostname, "done": {"$ne": True}})
        total_undone = db.count_documents({"done": {"$ne": True}, "corrupt": {"$ne": True}})
        
    print(f"Reserved by this host: {reserved_count}")
    print(f"Total undone vcons: {total_undone}")
    
    return reserved_count, total_undone

def check_disk_space():
    """Check if we're running out of disk space in cache"""
    import shutil
    
    print("\n=== DISK SPACE DEBUG ===")
    try:
        cache_usage = shutil.disk_usage(settings.cache_dir)
        cache_free_gb = cache_usage.free / (1024**3)
        print(f"Cache dir ({settings.cache_dir}) free space: {cache_free_gb:.1f} GB")
        
        if cache_free_gb < 1.0:
            print("⚠️ WARNING: Less than 1GB free in cache directory!")
            return False
        return True
    except Exception as e:
        print(f"Error checking disk space: {e}")
        return False

def check_gpu_processes():
    """Check what processes are using GPU"""
    import subprocess
    
    print("\n=== GPU PROCESS DEBUG ===")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if lines and lines[0]:
                print("GPU processes:")
                for line in lines:
                    if line.strip():
                        print(f"  {line}")
            else:
                print("No GPU processes found")
        else:
            print("Could not query GPU processes")
    except Exception as e:
        print(f"Error checking GPU: {e}")

def check_mongodb_connections():
    """Check MongoDB connection health"""
    print("\n=== MONGODB DEBUG ===")
    try:
        # Test a simple query
        start_time = time.time()
        with _db_semaphore:
            result = db.find_one({}, {"_id": 1})
        query_time = time.time() - start_time
        
        if result:
            print(f"MongoDB connection OK (query took {query_time:.3f}s)")
        else:
            print("MongoDB connection OK but no documents found")
            
        # Check semaphore
        print(f"DB semaphore available permits: {_db_semaphore._value}")
        
        return True
    except Exception as e:
        print(f"MongoDB connection error: {e}")
        return False

def monitor_cache_dir():
    """Monitor cache directory for stuck files"""
    import os
    import glob
    
    print("\n=== CACHE DIR DEBUG ===")
    
    try:
        # Check downloading directory
        downloading_files = glob.glob(os.path.join(settings.downloading_dir, "*"))
        processing_files = glob.glob(os.path.join(settings.processing_dir, "*"))
        
        print(f"Files in downloading dir: {len(downloading_files)}")
        print(f"Files in processing dir: {len(processing_files)}")
        
        # Check for stuck files (older than 5 minutes)
        stuck_files = []
        current_time = time.time()
        
        for file_path in downloading_files + processing_files:
            try:
                mtime = os.path.getmtime(file_path)
                age_minutes = (current_time - mtime) / 60
                if age_minutes > 5:
                    stuck_files.append((file_path, age_minutes))
            except OSError:
                pass
        
        if stuck_files:
            print("⚠️ Potentially stuck files (>5 min old):")
            for file_path, age in stuck_files[:5]:  # Show first 5
                print(f"  {os.path.basename(file_path)} ({age:.1f} min old)")
        else:
            print("No stuck files detected")
            
    except Exception as e:
        print(f"Error checking cache dir: {e}")

def main():
    """Run comprehensive pipeline debugging"""
    print("TRANSCRIBEY PIPELINE DEBUGGER")
    print("=" * 50)
    
    # Basic health checks
    mongodb_ok = check_mongodb_connections()
    disk_ok = check_disk_space()
    
    if not mongodb_ok:
        print("❌ MongoDB connection issues detected!")
        return
    
    if not disk_ok:
        print("❌ Disk space issues detected!")
        return
    
    # Check current state
    reserved, total = check_queue_sizes()
    check_gpu_processes()
    monitor_cache_dir()
    
    # Continuous monitoring
    print("\n=== CONTINUOUS MONITORING ===")
    print("Monitoring for 30 seconds... (Ctrl+C to stop)")
    
    try:
        for i in range(6):  # 30 seconds / 5 second intervals
            time.sleep(5)
            new_reserved, new_total = check_queue_sizes()
            
            reserved_change = new_reserved - reserved
            total_change = new_total - total
            
            print(f"[{i*5:2d}s] Reserved: {reserved_change:+d}, Total undone: {total_change:+d}")
            
            if reserved_change == 0 and new_reserved > 0:
                print("⚠️ No progress on reserved vcons - potential stuck process!")
            
            reserved, total = new_reserved, new_total
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
    
    print("\n=== RECOMMENDATIONS ===")
    if reserved > 0:
        print("• Check if preprocessing/lang_detect/transcribe processes are stuck")
        print("• Look for error messages in the console output")
        print("• Check GPU memory usage with 'nvidia-smi'")
        print("• Consider restarting the worker if no progress for >5 minutes")
    else:
        print("• No vcons currently reserved - reserver might be stuck")
        print("• Check SFTP connection and network connectivity")

if __name__ == "__main__":
    main() 