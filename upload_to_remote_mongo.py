#!/usr/bin/env python3
"""
Script to upload the local transcribey collection to remote MongoDB
"""

import os
import subprocess
import sys
from pymongo import MongoClient
import tempfile

# Database configurations
LOCAL_URI = "mongodb://bantaim:tjiSamKAAww@127.0.0.1:27017/transcribey?authSource=admin"
REMOTE_URI = "mongodb://admin:furryfr13nd@45.55.123.39:27017/"
DATABASE_NAME = "transcribey"
COLLECTION_NAME = "vcons"

def test_connection(uri, name):
    """Test connection to a MongoDB instance"""
    if "45.55.123.39" in uri:
        # Special handling for remote connection - try different auth methods
        print(f"Testing {name} connection with different auth methods...")
        
        # Try different connection approaches
        attempts = [
            ("Direct URI", uri),
            ("Separate auth", "mongodb://45.55.123.39:27017/"),
        ]
        
        for method, test_uri in attempts:
            try:
                print(f"  Trying {method}...")
                if method == "Separate auth":
                    # Try with explicit auth
                    client = MongoClient(
                        test_uri,
                        username="admin",
                        password="furryfr13nd",
                        authSource="admin",
                        serverSelectionTimeoutMS=5000
                    )
                else:
                    client = MongoClient(test_uri, serverSelectionTimeoutMS=5000)
                
                client.admin.command('ping')
                print(f"✅ {name} connection successful using {method}")
                return True
            except Exception as e:
                print(f"  ❌ {method} failed: {e}")
        
        print(f"❌ All {name} connection methods failed")
        return False
    else:
        # Standard connection test for local
        try:
            print(f"Testing {name} connection...")
            client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            client.admin.command('ping')
            print(f"✅ {name} connection successful")
            return True
        except Exception as e:
            print(f"❌ {name} connection failed: {e}")
            return False

def get_collection_stats(uri, db_name, collection_name):
    """Get collection statistics"""
    try:
        client = MongoClient(uri)
        db = client[db_name]
        collection = db[collection_name]
        count = collection.count_documents({})
        return count
    except Exception as e:
        print(f"Error getting collection stats: {e}")
        return None

def export_collection():
    """Export the local collection using mongodump"""
    print("Exporting local collection...")
    
    # Create temporary directory for export
    export_dir = tempfile.mkdtemp()
    print(f"Export directory: {export_dir}")
    
    try:
        # Build mongodump command
        cmd = [
            "mongodump",
            "--uri", LOCAL_URI,
            "--db", DATABASE_NAME,
            "--collection", COLLECTION_NAME,
            "--out", export_dir
        ]
        
        # Run mongodump
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Export successful")
            return export_dir
        else:
            print(f"❌ Export failed: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Export error: {e}")
        return None

def import_collection(export_dir):
    """Import the collection to remote MongoDB using mongorestore"""
    print("Importing to remote MongoDB...")
    
    # Try different authentication methods
    attempts = [
        # Method 1: Direct URI
        {
            "name": "Direct URI",
            "cmd": [
                "mongorestore",
                "--uri", REMOTE_URI,
                "--db", DATABASE_NAME,
                "--collection", COLLECTION_NAME,
                "--drop",
                os.path.join(export_dir, DATABASE_NAME, f"{COLLECTION_NAME}.bson")
            ]
        },
        # Method 2: Separate auth parameters
        {
            "name": "Separate auth parameters",
            "cmd": [
                "mongorestore",
                "--host", "45.55.123.39:27017",
                "--username", "admin",
                "--password", "furryfr13nd",
                "--authenticationDatabase", "admin",
                "--db", DATABASE_NAME,
                "--collection", COLLECTION_NAME,
                "--drop",
                os.path.join(export_dir, DATABASE_NAME, f"{COLLECTION_NAME}.bson")
            ]
        },
        # Method 3: Without auth database specification
        {
            "name": "Without auth database",
            "cmd": [
                "mongorestore",
                "--host", "45.55.123.39:27017",
                "--username", "admin",
                "--password", "furryfr13nd",
                "--db", DATABASE_NAME,
                "--collection", COLLECTION_NAME,
                "--drop",
                os.path.join(export_dir, DATABASE_NAME, f"{COLLECTION_NAME}.bson")
            ]
        }
    ]
    
    for attempt in attempts:
        print(f"\nTrying {attempt['name']}...")
        try:
            # Run mongorestore
            result = subprocess.run(attempt["cmd"], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Import successful")
                return True
            else:
                print(f"❌ {attempt['name']} failed: {result.stderr}")
                
        except Exception as e:
            print(f"❌ {attempt['name']} error: {e}")
    
    return False

def cleanup(export_dir):
    """Clean up temporary files"""
    try:
        subprocess.run(["rm", "-rf", export_dir], check=True)
        print(f"✅ Cleaned up temporary directory: {export_dir}")
    except Exception as e:
        print(f"⚠️ Warning: Could not clean up {export_dir}: {e}")

def main():
    print("=" * 60)
    print("MongoDB Collection Upload Script")
    print("=" * 60)
    
    # Test connections
    local_ok = test_connection(LOCAL_URI, "Local MongoDB")
    remote_ok = test_connection(REMOTE_URI, "Remote MongoDB")
    
    if not local_ok:
        print("❌ Cannot connect to local MongoDB. Please check the connection.")
        sys.exit(1)
    
    if not remote_ok:
        print("❌ Cannot connect to remote MongoDB. Please check the connection.")
        sys.exit(1)
    
    # Get collection stats
    local_count = get_collection_stats(LOCAL_URI, DATABASE_NAME, COLLECTION_NAME)
    if local_count is None:
        print("❌ Could not get local collection statistics")
        sys.exit(1)
    
    print(f"Local collection has {local_count} documents")
    
    if local_count == 0:
        print("⚠️ Local collection is empty. Nothing to export.")
        sys.exit(0)
    
    # Confirm before proceeding
    print(f"\nThis will:")
    print(f"  - Export {local_count} documents from local collection")
    print(f"  - Drop the remote collection if it exists")
    print(f"  - Import all documents to remote MongoDB")
    
    confirm = input("\nProceed? (y/N): ").lower().strip()
    if confirm != 'y':
        print("Operation cancelled.")
        sys.exit(0)
    
    # Export
    export_dir = export_collection()
    if not export_dir:
        print("❌ Export failed. Cannot proceed.")
        sys.exit(1)
    
    # Import
    success = import_collection(export_dir)
    
    # Cleanup
    cleanup(export_dir)
    
    if success:
        # Verify import
        remote_count = get_collection_stats(REMOTE_URI, DATABASE_NAME, COLLECTION_NAME)
        if remote_count == local_count:
            print(f"🎉 Upload successful! {remote_count} documents imported.")
        else:
            print(f"⚠️ Upload completed but counts don't match. Local: {local_count}, Remote: {remote_count}")
    else:
        print("❌ Upload failed.")
        sys.exit(1)

if __name__ == "__main__":
    main() 