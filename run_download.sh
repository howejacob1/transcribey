#!/bin/bash
# Wrapper script to run S3 download with pre-flight checks

set -e

echo "S3 Download Script Runner"
echo "========================"
echo

# Run pre-flight checks
echo "Running pre-flight checks..."
if ! python3 test_setup.py; then
    echo
    echo "❌ Pre-flight checks failed. Please fix the issues before running the download."
    echo
    echo "Common fixes:"
    echo "  1. Test S3 access: python3 setup_s3_credentials.py"
    echo "  2. Check MongoDB connection"
    echo "  3. Ensure /media/10900-hdd-0/ is writable"
    exit 1
fi

echo
echo "✅ Pre-flight checks passed!"
echo
echo "Starting S3 download script..."
echo "Press Ctrl+C to interrupt at any time"
echo

# Give user a chance to abort
sleep 3

# Run the main download script
python3 download_missing_s3_files.py

echo
echo "✅ Download script completed!" 