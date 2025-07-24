#!/usr/bin/env python3

from mongo_utils import db, client, something_db
import mongo_utils
from pymongo import InsertOne
import time

def move_spanish_vcons():
    """Move all Spanish VCONs from main collection to spanish-vcons collection"""
    
    # Get reference to the spanish-vcons collection
    spanish_db = something_db["spanish-vcons"]
    
    print("Finding Spanish VCONs...")
    
    # Find all Spanish VCONs
    query = {
        "analysis": {
            "$elemMatch": {
                "type": "language_identification",
                "body.languages": "es"
            }
        }
    }
    
    # Count Spanish VCONs first
    with mongo_utils._db_semaphore:
        spanish_count = db.count_documents(query)
    
    if spanish_count == 0:
        print("No Spanish VCONs found.")
        return 0
    
    print(f"Found {spanish_count} Spanish VCONs to move...")
    
    # Show some examples before moving
    print("\nExamples of Spanish VCONs:")
    with mongo_utils._db_semaphore:
        examples = list(db.find(query).limit(5))
    
    for i, vcon_dict in enumerate(examples):
        try:
            dialog = vcon_dict.get("dialog", [])
            filename = "Unknown"
            if dialog and len(dialog) > 0 and isinstance(dialog[0], dict):
                filename = dialog[0].get("filename", "Unknown")
            uuid = vcon_dict.get("uuid", vcon_dict.get("_id", "Unknown"))
            
            # Extract language info from analysis
            analysis = vcon_dict.get("analysis", [])
            lang_info = "Unknown"
            for item in analysis:
                if item.get("type") == "language_identification":
                    lang_info = item.get("body", {}).get("languages", "Unknown")
                    break
            
            print(f"  {i+1}. UUID: {uuid}")
            print(f"      File: {filename}")
            print(f"      Language: {lang_info}")
        except Exception as e:
            print(f"  {i+1}. UUID: {vcon_dict.get('uuid', vcon_dict.get('_id', 'Unknown'))}")
    
    if spanish_count > 5:
        print(f"  ... and {spanish_count - 5} more")
    
    # Ask for confirmation
    confirm = input(f"\nAre you sure you want to move {spanish_count} Spanish VCONs to 'spanish-vcons' collection? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Operation cancelled.")
        return 0
    
    print("\nMoving Spanish VCONs...")
    
    # Process in batches to avoid memory issues
    batch_size = 1000
    total_moved = 0
    
    while True:
        # Get a batch of Spanish VCONs
        with mongo_utils._db_semaphore:
            batch = list(db.find(query).limit(batch_size))
        
        if not batch:
            break
        
        print(f"Processing batch of {len(batch)} VCONs...")
        
        # Prepare bulk operations for spanish-vcons collection
        insert_operations = []
        uuids_to_delete = []
        
        for vcon_dict in batch:
            # Prepare for insertion into spanish-vcons
            insert_operations.append(InsertOne(vcon_dict))
            
            # Track UUID for deletion from main collection
            uuid_val = vcon_dict.get("_id", vcon_dict.get("uuid"))
            if uuid_val:
                uuids_to_delete.append(uuid_val)
        
        if insert_operations and uuids_to_delete:
            try:
                # Insert into spanish-vcons collection
                with mongo_utils._db_semaphore:
                    result = spanish_db.bulk_write(insert_operations, ordered=False)
                
                print(f"  Inserted {result.inserted_count} VCONs into spanish-vcons collection")
                
                # Delete from main collection
                with mongo_utils._db_semaphore:
                    delete_result = db.delete_many({"_id": {"$in": uuids_to_delete}})
                
                print(f"  Deleted {delete_result.deleted_count} VCONs from main collection")
                
                total_moved += delete_result.deleted_count
                
            except Exception as e:
                print(f"Error processing batch: {e}")
                break
        
        # Small delay to prevent overwhelming the database
        time.sleep(0.1)
    
    print(f"\nSuccessfully moved {total_moved} Spanish VCONs to 'spanish-vcons' collection.")
    
    # Verify the move
    with mongo_utils._db_semaphore:
        remaining_spanish = db.count_documents(query)
        spanish_collection_count = spanish_db.count_documents({})
    
    print(f"Verification:")
    print(f"  Remaining Spanish VCONs in main collection: {remaining_spanish}")
    print(f"  Total VCONs in spanish-vcons collection: {spanish_collection_count}")
    
    return total_moved

def list_spanish_stats():
    """Show statistics about Spanish VCONs without moving them"""
    
    print("Spanish VCON Statistics:")
    print("=" * 50)
    
    # Count Spanish VCONs
    query = {
        "analysis": {
            "$elemMatch": {
                "type": "language_identification",
                "body.languages": "es"
            }
        }
    }
    
    with mongo_utils._db_semaphore:
        spanish_count = db.count_documents(query)
        total_count = db.count_documents({})
        done_count = db.count_documents({"done": True})
    
    print(f"Total VCONs: {total_count}")
    print(f"Done VCONs: {done_count}")
    print(f"Spanish VCONs: {spanish_count}")
    
    if total_count > 0:
        spanish_percentage = (spanish_count / total_count) * 100
        print(f"Spanish percentage of total: {spanish_percentage:.2f}%")
    
    if done_count > 0:
        spanish_done_percentage = (spanish_count / done_count) * 100
        print(f"Spanish percentage of done: {spanish_done_percentage:.2f}%")
    
    # Check if spanish-vcons collection exists and has data
    spanish_db = something_db["spanish-vcons"]
    with mongo_utils._db_semaphore:
        spanish_collection_count = spanish_db.count_documents({})
    
    if spanish_collection_count > 0:
        print(f"VCONs already in spanish-vcons collection: {spanish_collection_count}")
    
    # Also show English count for comparison
    english_query = {
        "analysis": {
            "$elemMatch": {
                "type": "language_identification",
                "body.languages": "en"
            }
        }
    }
    
    with mongo_utils._db_semaphore:
        english_count = db.count_documents(english_query)
    
    print(f"English VCONs (for comparison): {english_count}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--stats":
        list_spanish_stats()
    else:
        move_spanish_vcons() 