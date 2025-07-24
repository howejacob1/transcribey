#!/usr/bin/env python3
"""
Helper script to test public access to Digital Ocean Spaces buckets.
"""

import os
import subprocess
import sys

def setup_credentials():
    """Verify access to public Digital Ocean Spaces buckets"""
    print("Digital Ocean Spaces Public Access Test")
    print("=" * 50)
    print()
    print("The buckets are configured as public, so no credentials are needed.")
    print("Testing public access to vol1-eon through vol8-eon buckets...")
    print()
    
    # Test the connection to public buckets
    print("Testing connection to Digital Ocean Spaces...")
    result = subprocess.run([
        's5cmd',
        '--endpoint-url', 'https://nyc3.digitaloceanspaces.com',
        '--no-sign-request',
        'ls'
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Successfully connected to Digital Ocean Spaces")
        print("Accessible buckets:")
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                print(f"  {line.strip()}")
    else:
        print(f"❌ Connection test failed: {result.stderr}")
        return False
    
    # Test access to a specific bucket
    print("\nTesting access to vol1-eon bucket...")
    result = subprocess.run([
        's5cmd',
        '--endpoint-url', 'https://nyc3.digitaloceanspaces.com',
        '--no-sign-request',
        'ls',
        's3://vol1-eon/*'
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        file_count = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
        print(f"✓ Successfully accessed vol1-eon bucket ({file_count} files visible)")
    else:
        print(f"❌ Failed to access vol1-eon bucket: {result.stderr}")
        return False
            
    print("\n✓ Public access verified!")
    print("\nYou can now run the download script:")
    print("  python3 download_missing_s3_files.py")
    
    return True

def check_existing_credentials():
    """Check if public access to Digital Ocean Spaces is working"""
    try:
        result = subprocess.run([
            's5cmd',
            '--endpoint-url', 'https://nyc3.digitaloceanspaces.com',
            '--no-sign-request',
            'ls'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✓ Public access to Digital Ocean Spaces is working")
            return True
        else:
            print("❌ Cannot access Digital Ocean Spaces")
            return False
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ Cannot access Digital Ocean Spaces")
        return False

def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--check':
        return 0 if check_existing_credentials() else 1
    
    if check_existing_credentials():
        print("Public access is working. No setup needed!")
        return 0
    
    success = setup_credentials()
    return 0 if success else 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nSetup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Setup failed: {e}")
        sys.exit(1) 