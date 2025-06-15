#!/usr/bin/env python3

import multiprocessing
import queue
import threading
import time
import random
from typing import List

# Simulate your current blocking approach
def blocking_producer(output_queue, name: str, items: int = 10):
    """Simulates discover/reserver - produces items but blocks on queue operations"""
    print(f"{name}: Starting...")
    for i in range(items):
        item = f"{name}_item_{i}"
        print(f"{name}: Producing {item}")
        output_queue.put(item)  # This can block if queue is full
        time.sleep(0.1)  # Simulate work
    print(f"{name}: Finished producing {items} items")

def blocking_consumer(input_queue, output_queue, name: str):
    """Simulates your current preprocess approach - blocks waiting for input"""
    print(f"{name}: Starting...")
    count = 0
    while True:
        try:
            item = input_queue.get(block=True, timeout=2.0)  # Blocks waiting
            print(f"{name}: Processing {item}")
            time.sleep(0.2)  # Simulate processing work
            if output_queue:
                output_queue.put(f"processed_{item}")
            count += 1
        except queue.Empty:
            print(f"{name}: Timeout - stopping after {count} items")
            break

# Better non-blocking approach
def non_blocking_producer(output_queue, name: str, items: int = 10):
    """Non-blocking producer that continues working even if queue is full"""
    print(f"{name}: Starting...")
    buffer = []
    
    for i in range(items):
        item = f"{name}_item_{i}"
        print(f"{name}: Producing {item}")
        
        # Try to put item, but don't block if queue is full
        try:
            output_queue.put(item, block=False)
        except queue.Full:
            buffer.append(item)  # Store in local buffer
            print(f"{name}: Queue full, buffering {item}")
        
        # Try to flush buffer
        while buffer:
            try:
                output_queue.put(buffer[0], block=False)
                buffer.pop(0)
            except queue.Full:
                break
        
        time.sleep(0.05)  # Less sleep = more parallelism
    
    # Final buffer flush
    while buffer:
        try:
            output_queue.put(buffer[0], block=True, timeout=1.0)
            buffer.pop(0)
        except queue.Full:
            print(f"{name}: Lost items in buffer: {len(buffer)}")
            break
    
    print(f"{name}: Finished producing {items} items")

def non_blocking_consumer(input_queue, output_queue, name: str):
    """Non-blocking consumer that doesn't wait indefinitely"""
    print(f"{name}: Starting...")
    count = 0
    consecutive_empty = 0
    
    while consecutive_empty < 5:  # Stop after 5 consecutive empty checks
        try:
            item = input_queue.get(block=False)  # Don't block
            print(f"{name}: Processing {item}")
            time.sleep(0.1)  # Less processing time
            if output_queue:
                try:
                    output_queue.put(f"processed_{item}", block=False)
                except queue.Full:
                    print(f"{name}: Output queue full, dropping processed item")
            count += 1
            consecutive_empty = 0
        except queue.Empty:
            consecutive_empty += 1
            time.sleep(0.02)  # Short sleep when no work
    
    print(f"{name}: Finished processing {count} items")

def demo_blocking_approach():
    """Demonstrates your current blocking approach"""
    print("\n" + "="*50)
    print("DEMO: BLOCKING APPROACH (Current Implementation)")
    print("="*50)
    
    # Small queues to force blocking
    q1 = queue.Queue(maxsize=2)
    q2 = queue.Queue(maxsize=2)
    
    start_time = time.time()
    
    threads = [
        threading.Thread(target=blocking_producer, args=(q1, "Producer1", 8)),
        threading.Thread(target=blocking_consumer, args=(q1, q2, "Consumer1")),
        threading.Thread(target=blocking_consumer, args=(q2, None, "Consumer2")),
    ]
    
    for t in threads:
        t.start()
    
    for t in threads:
        t.join()
    
    total_time = time.time() - start_time
    print(f"BLOCKING APPROACH Total time: {total_time:.2f} seconds")

def demo_non_blocking_approach():
    """Demonstrates improved non-blocking approach"""
    print("\n" + "="*50)
    print("DEMO: NON-BLOCKING APPROACH (Improved)")
    print("="*50)
    
    # Same small queues
    q1 = queue.Queue(maxsize=2)
    q2 = queue.Queue(maxsize=2)
    
    start_time = time.time()
    
    threads = [
        threading.Thread(target=non_blocking_producer, args=(q1, "Producer1", 8)),
        threading.Thread(target=non_blocking_consumer, args=(q1, q2, "Consumer1")),
        threading.Thread(target=non_blocking_consumer, args=(q2, None, "Consumer2")),
    ]
    
    for t in threads:
        t.start()
    
    for t in threads:
        t.join()
    
    total_time = time.time() - start_time
    print(f"NON-BLOCKING APPROACH Total time: {total_time:.2f} seconds")

if __name__ == "__main__":
    print("Parallel Processing Demo")
    print("This shows why your current approach runs sequentially")
    
    demo_blocking_approach()
    demo_non_blocking_approach()
    
    print("\n" + "="*50)
    print("ANALYSIS:")
    print("- Blocking approach: Each stage waits for the previous one")
    print("- Non-blocking approach: All stages work simultaneously")
    print("- Your code has blocking get() and put() calls causing serialization")
    print("="*50) 