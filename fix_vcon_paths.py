#!/usr/bin/env python3

from mongo_utils import db
import mongo_utils
from pymongo import UpdateOne
import re

def fix_vcon_paths():
    """Fix VCON paths to use correct /media/10900-hdd-0/ prefix"""
    
    print("Fixing VCON paths to use /media/10900-hdd-0/ prefix...")
    
    # Find VCONs that need path updates
    # Skip those that already start with /media/10900-hdd-0/
    # Look for filenames with Freeswitch<N> pattern
    query = {
        "dialog.0.filename": {
            "$regex": r"Freeswitch\d+/",
            "$not": {"$regex": r"^/media/10900-hdd-0/"}
        }
    }
    
    # First, check how many need updating
    with mongo_utils._db_semaphore:
        total_to_update = db.count_documents(query)
    
    print(f"Found {total_to_update:,} VCONs that need path updates")
    
    if total_to_update == 0:
        print("No VCONs need path updates. All done!")
        return 0
    
    batch_size = 1000
    total_processed = 0
    total_updates = 0
    
    # Use pagination to avoid loading all data at once
    skip = 0
    
    while True:
        # Get a batch of VCONs that need updates (without problematic projection)
        with mongo_utils._db_semaphore:
            batch = list(db.find(query).skip(skip).limit(batch_size))
        
        if not batch:
            break
        
        total_processed += len(batch)
        print(f"Processing batch {skip//batch_size + 1}, VCONs {skip + 1} to {skip + len(batch)}")
        
        # Prepare bulk operations for this batch
        bulk_operations = []
        examples_shown = 0
        
        for vcon in batch:
            try:
                # Get the current filename
                current_filename = vcon['dialog'][0]['filename']
                
                # Find the Freeswitch<N> pattern and extract everything from there
                match = re.search(r'(Freeswitch\d+/.*)$', current_filename)
                if match:
                    freeswitch_path = match.group(1)
                    new_filename = f"/media/10900-hdd-0/{freeswitch_path}"
                    
                    # Show some examples
                    if examples_shown < 3:
                        print(f"  Example: {current_filename} -> {new_filename}")
                        examples_shown += 1
                    
                    # Add to bulk operations
                    bulk_operations.append(
                        UpdateOne(
                            {'_id': vcon['_id']},
                            {'$set': {'dialog.0.filename': new_filename}}
                        )
                    )
                    
            except (KeyError, IndexError) as e:
                print(f"  Skipping malformed VCON {vcon.get('_id', 'unknown')}: {e}")
                continue
        
        # Execute bulk operations for this batch
        if bulk_operations:
            try:
                with mongo_utils._db_semaphore:
                    result = db.bulk_write(bulk_operations, ordered=False)
                batch_updates = result.modified_count
                total_updates += batch_updates
                print(f"  Updated {batch_updates} VCONs in this batch")
            except Exception as e:
                print(f"Error executing bulk operations for batch: {e}")
        
        skip += batch_size
        
        # Progress update
        if total_processed % 5000 == 0:
            print(f"Progress: {total_processed:,} VCONs processed, {total_updates:,} updated so far")
    
    print(f"\nCompleted processing {total_processed:,} VCONs")
    print(f"Successfully updated {total_updates:,} VCON paths")
    return total_updates

if __name__ == "__main__":
    fix_vcon_paths() 