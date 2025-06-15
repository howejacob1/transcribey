#!/usr/bin/env python3
"""
Test MongoDB connection using .secrets.toml configuration
"""

import sys
import traceback
from pymongo import MongoClient
from secrets_utils import secrets

def test_mongo_connection():
    """Test MongoDB connection and basic operations"""
    print("Testing MongoDB connection...")
    
    try:
        # Get configuration from secrets
        config = secrets.get('mongo_db', {})
        uri = config.get('url')
        db_name = config.get('db')
        
        if not uri:
            print("❌ ERROR: MongoDB URL not found in .secrets.toml")
            return False
            
        if not db_name:
            print("❌ ERROR: MongoDB database name not found in .secrets.toml")
            return False
            
        print(f"📝 Connecting to: {uri}")
        print(f"📝 Database: {db_name}")
        
        # Create client
        client = MongoClient(uri)
        
        # Test connection
        print("🔄 Testing connection...")
        client.admin.command('ping')
        print("✅ MongoDB connection successful!")
        
        # Get database
        db = client[db_name]
        
        # Test database access
        print("🔄 Testing database access...")
        collections = db.list_collection_names()
        print(f"✅ Database accessible. Collections: {collections}")
        
        # Test vcons collection specifically
        vcons_collection = db["vcons"]
        count = vcons_collection.count_documents({})
        print(f"📊 Vcons collection has {count} documents")
        
        # Test write operation (insert a test document)
        print("🔄 Testing write operation...")
        test_doc = {
            "_test": True,
            "timestamp": "test_connection",
            "message": "MongoDB connection test"
        }
        
        result = vcons_collection.insert_one(test_doc)
        print(f"✅ Write test successful. Inserted document ID: {result.inserted_id}")
        
        # Clean up test document
        vcons_collection.delete_one({"_id": result.inserted_id})
        print("🧹 Cleaned up test document")
        
        # Close connection
        client.close()
        print("✅ All MongoDB tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        print(f"📋 Full error details:")
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("=" * 60)
    print("MongoDB Connection Test")
    print("=" * 60)
    
    success = test_mongo_connection()
    
    print("=" * 60)
    if success:
        print("🎉 MongoDB connection test completed successfully!")
        sys.exit(0)
    else:
        print("💥 MongoDB connection test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 