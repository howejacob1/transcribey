#!/usr/bin/env python3
"""
Remove all "removed" fields from vcons in the database.
This resets the cleanup tracking so vcons can be processed again.
"""

import sys
import time
sys.path.append('.')

from mongo_utils import db

def remove_removed_fields():
    """Remove all 'removed' fields from vcons"""
    
    print("Starting removal of 'removed' fields from all vcons...")
    
    # Get initial count of vcons with removed field
    try:
        total_with_removed = db.count_documents({"removed": {"$exists": True}})
        
        print(f"VCons with 'removed' field: {total_with_removed:,}")
        
        if total_with_removed == 0:
            print("No vcons have 'removed' field. Nothing to do.")
            return 0
        
    except Exception as e:
        print(f"Error getting initial count: {e}")
        return 0
    
    # Batch configuration
    batch_size = 100000
    total_processed = 0
    
    start_time = time.time()
    
    try:
        while True:
            # Process in batches to avoid overwhelming the database
            result = db.update_many(
                {"removed": {"$exists": True}},
                {"$unset": {"removed": ""}},
                # Process in batches by limiting the operation
            )
            
            modified_count = result.modified_count
            total_processed += modified_count
            
            # Show progress
            elapsed = time.time() - start_time
            rate = total_processed / elapsed if elapsed > 0 else 0
            
            print(f"Processed: {total_processed:,}/{total_with_removed:,} vcons | "
                  f"Rate: {rate:.1f} vcons/sec")
            
            # If no documents were modified, we're done
            if modified_count == 0:
                break
            
            # Small delay to prevent overwhelming the database
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\nOperation interrupted by user.")
    except Exception as e:
        print(f"Error during removal: {e}")
    
    # Final verification
    try:
        remaining_with_removed = db.count_documents({"removed": {"$exists": True}})
        
        elapsed = time.time() - start_time
        print(f"\nRemoval completed in {elapsed:.1f} seconds")
        print(f"Total processed: {total_processed:,} vcons")
        print(f"Remaining with 'removed' field: {remaining_with_removed:,}")
        
        if remaining_with_removed == 0:
            print("✓ All 'removed' fields successfully removed!")
        else:
            print(f"⚠ {remaining_with_removed:,} vcons still have 'removed' field")
        
    except Exception as e:
        print(f"Error getting final count: {e}")
    
    return total_processed

if __name__ == "__main__":
    print("WARNING: This will remove all 'removed' fields from vcons!")
    print("This will reset cleanup tracking - all vcons will be eligible for cleanup again.")
    print()
    
    # Ask for confirmation
    confirm = input("Are you sure you want to proceed? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Operation cancelled.")
        sys.exit(0)
    
    removed_count = remove_removed_fields()
    print(f"\nOperation completed. Processed {removed_count:,} vcons.") 