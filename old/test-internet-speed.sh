#!/bin/bash

# This script tests UDP packet loss to a specified IP address using iperf3.

# The IP address of the server
SERVER_IP="192.168.1.108"

# Test parameters
BANDWIDTH="1000M" # Bandwidth to test at (e.g., 10M for 10 Mbit/s)
DURATION=10     # Duration of the test in seconds

# --- Do not edit below this line ---

echo "--- UDP Packet Loss Test ---"
echo "Starting test to $SERVER_IP"
echo ""
echo "IMPORTANT: Make sure you are running the iperf3 server on the target machine."
echo "You can do this by running the following command on the server:"
echo "iperf3 -s"
echo ""
echo "Running test for $DURATION seconds with a target bandwidth of $BANDWIDTH..."
echo ""

# Run iperf3 client
iperf3 -c $SERVER_IP -u -b $BANDWIDTH -t $DURATION

echo ""
echo "--- Test Complete ---"
echo "Look for the 'Lost/Total Datagrams' line in the output above."
echo "This will show you how many packets were lost during the test." 