#!/usr/bin/env python3
"""
Delete all vcons NOT marked as done from the vcons collection.
Much faster than moving them to another collection.
"""

import sys
import time
import concurrent.futures
import threading
from datetime import datetime
sys.path.append('.')

from mongo_utils import db, _db_semaphore

BATCH_SIZE = 30000  # Larger batches for deletion efficiency
PROGRESS_INTERVAL = 30000  # Show progress every N documents
MAX_WORKERS = 20  # Number of parallel threads

def delete_batch(batch_ids):
    """Delete a batch of vcons by their IDs"""
    if not batch_ids:
        return 0
    
    try:
        with _db_semaphore:
            result = db.delete_many({"_id": {"$in": batch_ids}})
            return result.deleted_count
    except Exception as e:
        print(f"Error deleting batch: {e}")
        return 0

def delete_undone_vcons():
    """Delete all vcons not marked as done using parallel processing"""
    
    print("Starting deletion of vcons not marked as done...")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max workers: {MAX_WORKERS}")
    print(f"Progress updates every {PROGRESS_INTERVAL:,} deletions")
    print()
    
    total_deleted = 0
    start_time = time.time()
    last_progress_time = start_time
    last_progress_count = 0
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            
            while True:
                with _db_semaphore:
                    # Find a batch of vcons that are not done (just get IDs for efficiency)
                    cursor = db.find(
                        {"$or": [{"done": {"$ne": True}}, {"done": {"$exists": False}}]},
                        {"_id": 1},  # Only return _id field
                        limit=BATCH_SIZE
                    )
                    
                    batch_ids = [doc["_id"] for doc in cursor]
                
                if not batch_ids:
                    print(f"{datetime.now().strftime('%H:%M:%S')} - No more vcons to delete!")
                    break
                
                # Submit batch for deletion
                future = executor.submit(delete_batch, batch_ids)
                futures.append(future)
                
                # Collect completed futures to avoid memory buildup
                completed_futures = []
                for future in futures:
                    if future.done():
                        try:
                            deleted = future.result()
                            total_deleted += deleted
                            completed_futures.append(future)
                        except Exception as e:
                            print(f"Batch deletion error: {e}")
                            completed_futures.append(future)
                
                # Remove completed futures
                for future in completed_futures:
                    futures.remove(future)
                
                # Progress reporting
                current_time = time.time()
                if total_deleted > 0 and (total_deleted % PROGRESS_INTERVAL == 0 or 
                                        current_time - last_progress_time >= 5):  # At least every 5 seconds
                    
                    elapsed_total = current_time - start_time
                    elapsed_since_last = current_time - last_progress_time
                    
                    # Overall rate
                    overall_rate = total_deleted / elapsed_total if elapsed_total > 0 else 0
                    
                    # Recent rate (since last progress update)
                    docs_since_last = total_deleted - last_progress_count
                    recent_rate = docs_since_last / elapsed_since_last if elapsed_since_last > 0 else 0
                    
                    # Estimate remaining time
                    if overall_rate > 0:
                        # Get current count to estimate remaining
                        try:
                            with _db_semaphore:
                                remaining_estimate = db.count_documents(
                                    {"$or": [{"done": {"$ne": True}}, {"done": {"$exists": False}}]}, 
                                    limit=100000  # Limit to avoid slow queries
                                )
                            if remaining_estimate >= 100000:
                                remaining_str = ">100k remaining"
                            else:
                                eta_seconds = remaining_estimate / overall_rate
                                eta_hours = eta_seconds / 3600
                                remaining_str = f"~{remaining_estimate:,} remaining (~{eta_hours:.1f}h left)"
                        except:
                            remaining_str = "remaining unknown"
                    else:
                        remaining_str = "calculating..."
                    
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    print(f"{timestamp} - Deleted {total_deleted:,} vcons | "
                          f"Overall: {overall_rate:.1f}/sec | Recent: {recent_rate:.1f}/sec | "
                          f"{remaining_str}")
                    
                    last_progress_time = current_time
                    last_progress_count = total_deleted
                
                # Small delay to prevent overwhelming the database
                time.sleep(0.01)
            
            # Wait for remaining futures to complete
            print("Waiting for final batches to complete...")
            for future in concurrent.futures.as_completed(futures):
                try:
                    deleted = future.result()
                    total_deleted += deleted
                except Exception as e:
                    print(f"Final batch deletion error: {e}")
    
    except KeyboardInterrupt:
        print(f"\nInterrupted! Deleted {total_deleted:,} vcons so far.")
        return total_deleted
    except Exception as e:
        print(f"Error during deletion operation: {e}")
        return total_deleted
    
    elapsed = time.time() - start_time
    print(f"\nCompleted! Deleted {total_deleted:,} vcons in {elapsed:.1f} seconds")
    print(f"Average rate: {total_deleted/elapsed:.1f} docs/sec")
    
    return total_deleted

def verify_counts():
    """Verify the counts after deletion"""
    try:
        with _db_semaphore:
            total_vcons = db.estimated_document_count()
            done_vcons = db.count_documents({"done": True})
            
            print(f"\nFinal counts:")
            print(f"  Total vcons remaining: {total_vcons:,}")
            print(f"  Done vcons: {done_vcons:,}")
            print(f"  Undone vcons: {total_vcons - done_vcons:,}")
            
    except Exception as e:
        print(f"Error getting final counts: {e}")

if __name__ == "__main__":
    print("WARNING: This will DELETE all vcons not marked as done!")
    print("This operation CANNOT be undone!")
    
    # Get initial counts
    try:
        with _db_semaphore:
            initial_vcons = db.estimated_document_count()
            done_vcons = db.count_documents({"done": True})
            to_delete = initial_vcons - done_vcons
            
        print(f"\nInitial state:")
        print(f"  Total vcons: {initial_vcons:,}")
        print(f"  Done vcons: {done_vcons:,}")
        print(f"  Vcons to DELETE: {to_delete:,}")
        print()
        
    except Exception as e:
        print(f"Error getting initial counts: {e}")
        sys.exit(1)
    
    # Start the deletion operation
    deleted = delete_undone_vcons()
    
    # Verify final state
    verify_counts()
    
    print(f"\nOperation completed. Deleted {deleted:,} vcons.") 