#!/usr/bin/env python3
"""
Move all vcons NOT marked as done to a new collection called 'trashio'.
Uses parallel batch processing to handle the massive number of documents efficiently.
"""

import sys
import time
import concurrent.futures
import threading
from datetime import datetime
from pymongo import UpdateOne
sys.path.append('.')

from mongo_utils import db, _db_semaphore, something_db

BATCH_SIZE = 5000  # Larger batches for efficiency
PROGRESS_INTERVAL = 5000  # Show progress every N documents (more frequent)
MAX_WORKERS = 10  # Number of parallel threads
PREFETCH_BATCHES = 20  # Number of batches to prefetch

def process_batch(batch_data):
    """Process a single batch of vcons - move from vcons to trashio"""
    batch, trashio = batch_data
    
    if not batch:
        return 0
    
    try:
        batch_ids = [doc["_id"] for doc in batch]
        
        # Check which documents already exist in trashio to avoid duplicates
        existing_ids = set()
        for doc_id in batch_ids:
            if trashio.find_one({"_id": doc_id}, {"_id": 1}):
                existing_ids.add(doc_id)
        
        # Filter out documents that already exist in trashio
        docs_to_insert = [doc for doc in batch if doc["_id"] not in existing_ids]
        ids_to_delete = [doc["_id"] for doc in docs_to_insert]
        
        if docs_to_insert:
            # Insert new documents into trashio collection
            trashio.insert_many(docs_to_insert)
            
            # Remove inserted documents from vcons collection
            db.delete_many({"_id": {"$in": ids_to_delete}})
        
        # Also delete any documents that already existed in trashio
        if existing_ids:
            db.delete_many({"_id": {"$in": list(existing_ids)}})
        
        return len(batch)
        
    except Exception as e:
        print(f"Error processing batch: {e}")
        return 0

def fetch_batches(trashio, batch_queue, stop_event):
    """Fetch batches of vcons and put them in queue for processing"""
    try:
        while not stop_event.is_set():
            # Find a batch of vcons that are not done
            cursor = db.find(
                {"$or": [{"done": {"$ne": True}}, {"done": {"$exists": False}}]},
                limit=BATCH_SIZE
            )
            
            batch = list(cursor)
            
            if not batch:
                break
            
            # Put batch in queue for processing
            batch_queue.put((batch, trashio))
            
    except Exception as e:
        print(f"Error fetching batches: {e}")
    finally:
        # Signal that we're done fetching
        batch_queue.put(None)

def move_vcons_to_trashio():
    """Move all vcons not marked as done to trashio collection using parallel processing"""
    
    # Get reference to trashio collection
    trashio = something_db["trashio"]
    
    print("Starting parallel move of vcons not marked as done to trashio collection...")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max workers: {MAX_WORKERS}")
    
    total_moved = 0
    start_time = time.time()
    
    # Use a queue to manage batches
    import queue
    batch_queue = queue.Queue(maxsize=PREFETCH_BATCHES)
    stop_event = threading.Event()
    
    try:
        # Start fetcher thread
        fetcher_thread = threading.Thread(
            target=fetch_batches, 
            args=(trashio, batch_queue, stop_event)
        )
        fetcher_thread.start()
        
        # Process batches in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            
            while True:
                # Get next batch from queue
                batch_data = batch_queue.get()
                
                if batch_data is None:  # No more batches
                    break
                
                # Submit batch for processing
                future = executor.submit(process_batch, batch_data)
                futures.append(future)
                
                # Collect completed futures periodically
                if len(futures) >= MAX_WORKERS * 2:
                    completed_futures = []
                    for future in futures:
                        if future.done():
                            try:
                                moved = future.result()
                                total_moved += moved
                                completed_futures.append(future)
                            except Exception as e:
                                print(f"Batch processing error: {e}")
                                completed_futures.append(future)
                    
                    # Remove completed futures
                    for future in completed_futures:
                        futures.remove(future)
                    
                    # Progress reporting (more frequent and detailed)
                    if total_moved > 0 and total_moved % PROGRESS_INTERVAL == 0:
                        elapsed = time.time() - start_time
                        rate = total_moved / elapsed if elapsed > 0 else 0
                        current_time = datetime.now().strftime("%H:%M:%S")
                        remaining = 389_866_971 - total_moved
                        eta_seconds = remaining / rate if rate > 0 else 0
                        eta_hours = eta_seconds / 3600
                        print(f"[{current_time}] Moved {total_moved:,} vcons to trashio ({rate:.1f} docs/sec) - ETA: {eta_hours:.1f}h")
            
            # Wait for remaining futures to complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    moved = future.result()
                    total_moved += moved
                except Exception as e:
                    print(f"Final batch processing error: {e}")
        
        # Wait for fetcher thread to complete
        stop_event.set()
        fetcher_thread.join()
    
    except KeyboardInterrupt:
        print(f"\nInterrupted! Moved {total_moved:,} vcons so far.")
        stop_event.set()
        return total_moved
    except Exception as e:
        print(f"Error during move operation: {e}")
        stop_event.set()
        return total_moved
    
    elapsed = time.time() - start_time
    print(f"\nCompleted! Moved {total_moved:,} vcons to trashio in {elapsed:.1f} seconds")
    print(f"Average rate: {total_moved/elapsed:.1f} docs/sec")
    
    return total_moved

def verify_counts():
    """Verify the counts after the move operation"""
    try:
        vcons_count = db.estimated_document_count()
        trashio_count = something_db["trashio"].estimated_document_count()
        
        print(f"\nFinal counts:")
        print(f"  vcons collection: {vcons_count:,}")
        print(f"  trashio collection: {trashio_count:,}")
        
    except Exception as e:
        print(f"Error getting final counts: {e}")

if __name__ == "__main__":
    print("WARNING: This will move ~390 million vcons to a new collection!")
    print("This operation will take a very long time and cannot be easily undone.")
    
    # Get initial counts
    try:
        initial_vcons = db.estimated_document_count()
        done_vcons = db.count_documents({"done": True})
        to_move = initial_vcons - done_vcons
        
        print(f"\nInitial state:")
        print(f"  Total vcons: {initial_vcons:,}")
        print(f"  Done vcons: {done_vcons:,}")
        print(f"  Vcons to move: {to_move:,}")
        
    except Exception as e:
        print(f"Error getting initial counts: {e}")
        sys.exit(1)
    
    # Start the move operation
    moved = move_vcons_to_trashio()
    
    # Verify final state
    verify_counts()
    
    print(f"\nOperation completed. Moved {moved:,} vcons to trashio.") 