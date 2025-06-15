#!/usr/bin/env python3
"""
Simple MongoDB connection test with different approaches
"""

import sys
from pymongo import MongoClient
from secrets_utils import secrets

def test_basic_connection():
    """Test basic connection without auth"""
    try:
        print("🔄 Testing basic connection without auth...")
        client = MongoClient("mongodb://127.0.0.1:27017/")
        client.admin.command('ping')
        print("✅ Basic connection successful")
        client.close()
        return True
    except Exception as e:
        print(f"❌ Basic connection failed: {e}")
        return False

def test_auth_connection():
    """Test connection with authentication"""
    try:
        config = secrets.get('mongo_db', {})
        uri = config.get('url')
        print(f"🔄 Testing auth connection: {uri}")
        
        client = MongoClient(uri)
        client.admin.command('ping')
        print("✅ Auth connection successful")
        
        # Try to access the database
        db_name = config.get('db')
        db = client[db_name]
        
        # Try a simple operation
        try:
            collections = db.list_collection_names()
            print(f"✅ Database access successful. Collections: {collections}")
        except Exception as e:
            print(f"⚠️ Database access limited: {e}")
        
        client.close()
        return True
    except Exception as e:
        print(f"❌ Auth connection failed: {e}")
        return False

def main():
    print("=" * 60)
    print("Simple MongoDB Connection Test")
    print("=" * 60)
    
    basic_ok = test_basic_connection()
    auth_ok = test_auth_connection()
    
    if basic_ok:
        print("✅ MongoDB is accessible")
        if auth_ok:
            print("✅ Authentication is working")
        else:
            print("⚠️ Authentication needs setup")
    else:
        print("❌ MongoDB is not accessible")
    
    return auth_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 