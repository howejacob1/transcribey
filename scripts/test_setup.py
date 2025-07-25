#!/usr/bin/env python3
"""
Test script to verify all components are working before running the main download script.
"""

import sys
import subprocess
import os

# Add current directory to path for imports
sys.path.append('.')

def test_mongodb():
    """Test MongoDB connection"""
    print("Testing MongoDB connection...")
    try:
        from mongo_utils import db
        
        count = db.count_documents({})
        print(f"✓ MongoDB connected: {count:,} vcons in database")
        
        # Test basename query
        sample = db.find_one({"basename": {"$exists": True}})
        if sample:
            dialog = sample.get('dialog', [])
            if dialog and len(dialog) > 0:
                basename = dialog[0].get('basename', 'N/A')
                print(f"✓ Sample basename: {basename}")
            else:
                print("✓ Database accessible but no proper dialog structure found")
        else:
            print("⚠ No vcons with basename field found")
        
        return True
        
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        return False

def test_s5cmd():
    """Test s5cmd installation"""
    print("\nTesting s5cmd...")
    try:
        result = subprocess.run(['s5cmd', 'version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ s5cmd available: {result.stdout.strip()}")
            return True
        else:
            print("❌ s5cmd not working properly")
            return False
    except FileNotFoundError:
        print("❌ s5cmd not installed")
        return False

def test_s3_access():
    """Test public S3 access"""
    print("\nTesting public S3 access...")
    try:
        result = subprocess.run([
            's5cmd',
            '--endpoint-url', 'https://nyc3.digitaloceanspaces.com',
            '--no-sign-request',
            'ls'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✓ Public S3 access working")
            lines = result.stdout.strip().split('\n')
            bucket_count = len([line for line in lines if line.strip()])
            print(f"✓ Found {bucket_count} accessible buckets")
            return True
        else:
            print("❌ Cannot access Digital Ocean Spaces publicly")
            print("  This might be a network issue or the buckets are not public")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ S3 connection timeout")
        return False
    except Exception as e:
        print(f"❌ S3 test failed: {e}")
        return False

def test_target_directory():
    """Test target directory accessibility"""
    print("\nTesting target directory...")
    target_dir = "/media/10900-hdd-0"
    
    if not os.path.exists(target_dir):
        print(f"❌ Target directory does not exist: {target_dir}")
        return False
    
    if not os.access(target_dir, os.W_OK):
        print(f"❌ Target directory not writable: {target_dir}")
        return False
    
    # Check available space
    stat = os.statvfs(target_dir)
    free_bytes = stat.f_bavail * stat.f_frsize
    free_gb = free_bytes / (1024**3)
    
    print(f"✓ Target directory accessible: {target_dir}")
    print(f"✓ Available space: {free_gb:.1f} GB")
    
    if free_gb < 10:
        print("⚠ Warning: Less than 10 GB available space")
    
    return True

def test_imports():
    """Test required Python imports"""
    print("\nTesting Python imports...")
    required_modules = [
        'subprocess', 'os', 'sys', 'time', 'hashlib', 
        'concurrent.futures', 'json', 'pathlib'
    ]
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            print(f"❌ Missing module: {module}")
            return False
    
    print("✓ All required modules available")
    return True

def main():
    print("S3 Download Setup Test")
    print("=" * 40)
    
    tests = [
        ("Python imports", test_imports),
        ("MongoDB", test_mongodb),
        ("s5cmd", test_s5cmd),
        ("S3 access", test_s3_access),
        ("Target directory", test_target_directory),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 All tests passed! You can run the download script:")
        print("   python3 download_missing_s3_files.py")
    else:
        print("\n❌ Some tests failed. Please fix the issues before running the download script.")
        
        if failed == 1 and "S3 access" in [t[0] for t in tests]:
            print("\nTo test S3 access:")
            print("   python3 setup_s3_credentials.py")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1) 