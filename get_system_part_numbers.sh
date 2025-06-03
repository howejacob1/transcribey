#!/usr/bin/env bash

# Combined script to print part numbers/model info for drives, RAM, and CPU

echo "====================="
echo "DRIVE PART NUMBERS"
echo "====================="
for disk in /dev/nvme?n?; do
    echo "Drive: $disk"
    # Try to get model/part number using udevadm
    udevadm info --query=all --name="$disk" | grep -E 'ID_MODEL=|ID_PART_ENTRY_NUMBER='
    # Fallback: Try hdparm if available
    if command -v hdparm &> /dev/null; then
        sudo hdparm -I "$disk" | grep 'Model Number'
    fi
    echo "----------------------"
done

echo "====================="
echo "RAM PART NUMBERS"
echo "====================="
if ! command -v dmidecode &> /dev/null; then
    echo "dmidecode is not installed. Please install it with: sudo apt install dmidecode"
else
    sudo dmidecode --type memory | awk '
    /Memory Device$/ {in_device=1; next}
    /^\s*$/ {in_device=0}
    in_device && /Part Number:/ {
        print "RAM Part Number: " $0
    }
    '
fi

echo "====================="
echo "CPU PART NUMBER / MODEL"
echo "====================="
if command -v lscpu &> /dev/null; then
    echo "CPU Model (from lscpu):"
    lscpu | grep 'Model name'
else
    echo "lscpu not found, falling back to /proc/cpuinfo"
    grep 'model name' /proc/cpuinfo | head -1
fi

echo "====================="
echo "HARD DRIVE SPEED TEST (10GB COPY)"
echo "====================="
TEST_SRC="/tmp/dd_speed_test_src.bin"
TEST_DST="/tmp/dd_speed_test_dst.bin"

# Create a 10GB test file if it doesn't exist
if [ ! -f "$TEST_SRC" ]; then
    echo "Creating 10GB test file at $TEST_SRC ..."
    dd if=/dev/zero of="$TEST_SRC" bs=1M count=10240 status=progress
fi

# Sync to flush caches
sync

# Time the copy
START=$(date +%s)
echo "Copying 10GB file to $TEST_DST ..."
dd if="$TEST_SRC" of="$TEST_DST" bs=1M status=progress
sync
END=$(date +%s)

ELAPSED=$((END - START))
echo "Copy took $ELAPSED seconds."

# Clean up
rm -f "$TEST_SRC" "$TEST_DST"
echo "Test files removed." 