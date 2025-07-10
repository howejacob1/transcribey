#!/usr/bin/env python3

from mongo_utils import db
import mongo_utils
from vcon_class import Vcon
from pymongo import UpdateOne

def update_vcon_paths():
    """Update VCON paths to move them to old-thing subdirectory"""
    
    print("Fetching VCONs with transcripts to check paths...")
    
    # Fetch all VCONs with transcripts in batches
    batch_size = 1000
    total_processed = 0
    total_updates = 0
    
    # Use pagination to avoid loading all data at once
    skip = 0
    
    while True:
        # Get a batch of VCONs with transcripts
        with mongo_utils._db_semaphore:
            batch = list(db.find(
                {'analysis.type': 'transcript'}, 
                {'_id': 1, 'dialog.0.filename': 1}
            ).skip(skip).limit(batch_size))
        
        if not batch:
            break
        
        total_processed += len(batch)
        print(f"Processing batch {skip//batch_size + 1}, VCONs {skip + 1} to {skip + len(batch)}")
        
        # Prepare bulk operations for this batch
        bulk_operations = []
        examples_shown = 0
        
        for vcon_dict in batch:
            try:
                if not vcon_dict.get('dialog') or not vcon_dict['dialog']:
                    continue
                    
                current_filename = vcon_dict['dialog'][0].get('filename', '')
                if not current_filename:
                    continue
                    
                new_filename = None
                
                # Handle /media/1800-hdd-0/ -> /media/10900-hdd-0/old-thing/
                if current_filename.startswith('/media/1800-hdd-0/'):
                    remaining_path = current_filename[len('/media/1800-hdd-0/'):]
                    new_filename = f'/media/10900-hdd-0/old-thing/{remaining_path}'
                
                # Handle /media/10900-hdd-0/ -> /media/10900-hdd-0/old-thing/
                elif current_filename.startswith('/media/10900-hdd-0/'):
                    # Skip if it already starts with old-thing to avoid double-processing
                    if current_filename.startswith('/media/10900-hdd-0/old-thing/'):
                        continue
                    remaining_path = current_filename[len('/media/10900-hdd-0/'):]
                    new_filename = f'/media/10900-hdd-0/old-thing/{remaining_path}'
                
                if new_filename:
                    bulk_operations.append(
                        UpdateOne(
                            {'_id': vcon_dict['_id']},
                            {'$set': {'dialog.0.filename': new_filename}}
                        )
                    )
                    
                    # Show first few examples across all batches
                    if total_updates < 5:
                        print(f"  Will update: {current_filename}")
                        print(f"             -> {new_filename}")
                        
            except Exception as e:
                print(f"Error processing VCON {vcon_dict.get('_id', 'unknown')}: {e}")
        
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
    
    print(f"\nCompleted processing {total_processed:,} VCONs with transcripts")
    print(f"Successfully updated {total_updates:,} VCON paths")
    return total_updates

def delete_vcons_without_transcript():
    """Delete all VCONs that don't have a transcript"""
    
    # Find VCONs without transcript analysis
    query = {
        "$nor": [
            {"analysis.type": "transcript"}
        ]
    }
    
    with mongo_utils._db_semaphore:
        vcons_to_delete = list(db.find(query))
    
    if not vcons_to_delete:
        print("No VCONs found without transcripts.")
        return 0
    
    print(f"Found {len(vcons_to_delete)} VCONs without transcripts...")
    
    # Show some examples before deletion
    print("\nExamples of VCONs to be deleted:")
    for i, vcon_dict in enumerate(vcons_to_delete[:5]):
        try:
            dialog = vcon_dict.get("dialog", [])
            filename = "Unknown"
            if dialog and len(dialog) > 0:
                filename = dialog[0].get("filename", "Unknown")
            uuid = vcon_dict.get("uuid", "Unknown")
            print(f"  {i+1}. UUID: {uuid}, File: {filename}")
        except:
            print(f"  {i+1}. UUID: {vcon_dict.get('uuid', 'Unknown')}")
    
    if len(vcons_to_delete) > 5:
        print(f"  ... and {len(vcons_to_delete) - 5} more")
    
    # Ask for confirmation
    confirm = input(f"\nAre you sure you want to delete {len(vcons_to_delete)} VCONs? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Operation cancelled.")
        return 0
    
    # Delete the VCONs
    with mongo_utils._db_semaphore:
        result = db.delete_many(query)
    
    print(f"Successfully deleted {result.deleted_count} VCONs without transcripts.")
    return result.deleted_count

def main():
    print("VCON Database Update Script")
    print("=" * 50)
    
    # Step 1: Update paths
    print("\nStep 1: Moving VCON paths to old-thing subdirectory")
    updated_count = update_vcon_paths()
    
    # Step 2: Delete VCONs without transcripts
    print("\nStep 2: Deleting VCONs without transcripts")
    deleted_count = delete_vcons_without_transcript()
    
    print(f"\nSummary:")
    print(f"- Updated paths: {updated_count} VCONs")
    print(f"- Deleted: {deleted_count} VCONs")
    print("Operation completed.")

if __name__ == "__main__":
    main() 