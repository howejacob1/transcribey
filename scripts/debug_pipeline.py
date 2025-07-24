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
from mongo_utils import db

def check_queue_sizes():
    """Check if any processes are alive and working"""
    print("\n=== QUEUE SIZE DEBUG ===")
    
    # Count reserved but not processed vcons
    reserved_count = db.count_documents({"processed_by": settings.hostname, "done": {"$ne": True}})
    total_undone = db.count_documents({"done": {"$ne": True}, "corrupt": {"$ne": True}})
    
    print(f"Reserved by this host: {reserved_count}")
    print(f"Total undone vcons: {total_undone}")
    
    return reserved_count, total_undone

def debug_processing_state():
    """Debug the processing state to understand bottlenecks"""
    print("=== PROCESSING STATE DEBUG ===")
    
    # Check vcon status distribution
    total_vcons = db.count_documents({})
    done_vcons = db.count_documents({"done": True})
    corrupt_vcons = db.count_documents({"corrupt": True})
    processing_vcons = db.count_documents({"processed_by": {"$exists": True}, "done": {"$ne": True}})
    
    print(f"Total vcons: {total_vcons:,}")
    print(f"Done vcons: {done_vcons:,}")
    print(f"Corrupt vcons: {corrupt_vcons:,}")
    print(f"Currently processing: {processing_vcons:,}")
    
    # Check by hostname
    hostname_stats = db.aggregate([
        {"$match": {"processed_by": {"$exists": True}}},
        {"$group": {"_id": "$processed_by", "count": {"$sum": 1}}}
    ])
    
    print("\nProcessing by hostname:")
    for stat in hostname_stats:
        print(f"  {stat['_id']}: {stat['count']:,} vcons")
    
    return total_vcons, done_vcons, corrupt_vcons, processing_vcons

if __name__ == "__main__":
    debug_processing_state()
    check_queue_sizes() 