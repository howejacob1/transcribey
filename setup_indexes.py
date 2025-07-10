#!/usr/bin/env python3
"""
Create MongoDB indexes for optimal performance
"""

import sys
from mongo_utils import db

def setup_indexes():
    """Create performance indexes for the transcribey collection"""
    print("🔄 Setting up MongoDB indexes for optimal performance...")
    
    try:
        # Index for find_and_reserve operations (most critical)
        # Simplified - no partial filter to avoid compatibility issues
        db.create_index(
            [("done", 1), ("processed_by", 1)],
            name="idx_reservation_status",
            background=True
        )
        print("✅ Created reservation status index")
        
        # Index for filename lookups (discovery/existence checks)
        # Make it non-unique to avoid conflicts
        db.create_index(
            [("dialog.0.filename", 1)],
            name="idx_filename_lookup",
            background=True,
            sparse=True
        )
        print("✅ Created filename lookup index")
        
        # Index for processed_by hostname cleanup
        db.create_index(
            [("processed_by", 1), ("done", 1)],
            name="idx_processed_by_cleanup",
            background=True,
            sparse=True
        )
        print("✅ Created processed_by cleanup index")
        
        # Index for general status queries
        db.create_index(
            [("done", 1), ("created_at", 1)],
            name="idx_status_created",
            background=True
        )
        print("✅ Created status/created index")
        
        # Index for UUID lookups (primary key operations)
        # Make it non-unique since _id is already the primary key
        db.create_index(
            [("uuid", 1)],
            name="idx_uuid_lookup",
            background=True,
            sparse=True
        )
        print("✅ Created UUID lookup index")
        
        # Index specifically for unprocessed items (most important query)
        db.create_index(
            [("processed_by", 1)],
            name="idx_processed_by_only",
            background=True,
            sparse=True
        )
        print("✅ Created processed_by index")
        
        print("🎉 All indexes created successfully!")
        
        # Show index info
        print("\n📊 Index information:")
        indexes = db.list_indexes()
        for index in indexes:
            print(f"  - {index['name']}: {index.get('key', 'N/A')}")
            
    except Exception as e:
        print(f"❌ Error creating indexes: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("MongoDB Index Setup for Transcribey")
    print("=" * 60)
    
    success = setup_indexes()
    
    if success:
        print("\n🎉 MongoDB indexes setup completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 MongoDB indexes setup failed!")
        sys.exit(1) 