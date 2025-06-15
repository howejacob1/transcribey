import multiprocessing
import time
from contextlib import contextmanager
from queue import Empty
from time import perf_counter

try:
    import GPUtil
    gpu_available = True
except ImportError:
    gpu_available = False

from process import our_program_name
import settings
from utils import seconds_to_days, seconds_to_hours

def make_stats_queue():
    return multiprocessing.Queue()

def add(stats_queue, key, value):
    stats_queue.put({"program": our_program_name(),
                     "name": key,
                     "value": value})

@contextmanager
def with_blocking_time(stats_queue):
    """Am fancy because I want perf_counter accuracy but I want stuff readable."""
    add(stats_queue, "blocking_time_start", time.time())
    blocking_time_start = perf_counter()
    yield
    blocking_time_stop = perf_counter()
    blocking_duration = blocking_time_stop - blocking_time_start
    add(stats_queue, "blocking_time_stop", time.time())
    add(stats_queue, "blocking_time_duration", blocking_duration)

def actually_print_status(program,
                         vcons_count,
                         vcons_bytes,
                         is_blocking,
                         is_alive,
                         processing_duration,
                         total_runtime,
                         percent_processing,
                         vcons_rate,
                         vcons_bytes_rate_mb, 
                         vcons_duration):
    status_char = "🔴"
    if is_blocking:
        status_char = "⚪"
    elif is_alive:
        status_char = "🟢"
    
    # Convert bytes to MB for display
    vcons_bytes_mb = vcons_bytes / 1024 / 1024
    rtf = vcons_duration / processing_duration if processing_duration > 0 else 0
    duration_days = seconds_to_days(vcons_duration)
    total_runtime_hours = seconds_to_hours(total_runtime)

    print(f"{status_char} {program:15} | "
          f"{rtf:7.1f}x {vcons_count:8,} {vcons_rate:7.1f}/s {vcons_bytes_rate_mb:7.2f}MB/s | "
          f"Totals: {vcons_bytes_mb:7.2f}MB "
          f"{duration_days:7.1f}d "
          f"| {total_runtime_hours:7.1f}h ({percent_processing:5.1%})%")

def print_gpu_stats():
    """Print GPU memory and power statistics"""
    if gpu_available:
        gpu = GPUtil.getGPUs()[0]
        memory_used = gpu.memoryUsed
        memory_total = gpu.memoryTotal
        memory_percent = (memory_used / memory_total) * 100 if memory_total > 0 else 0
        temperature = gpu.temperature
        load = gpu.load * 100
    
    print(f"Mem: {memory_used:,}MB/{memory_total:,}MB ({memory_percent:.1f}%) | "
            f"Load: {load:.1f}% | "
            f"Temp: {temperature}°C")

def print_status(status):
    # Clear screen and move cursor to top
    print("\033[2J\033[H", end="")
    
    line_number = 0
    for program, measurements in status.items():
        vcons_count = measurements["vcons_count"]
        vcons_bytes = measurements["vcons_bytes"]
        is_blocking = measurements["blocking_start"] is not None
        blocking_duration = measurements["blocking_duration"]
        start_time = measurements["start_time"]
        is_alive = measurements["stop_time"] is None
        processing_duration = time.time() - start_time - blocking_duration
        vcons_duration = measurements["vcons_duration"]
        if is_blocking:
            extra_blocking_duration = time.time() - measurements["blocking_start"]
            processing_duration -= extra_blocking_duration
        total_runtime = time.time() - start_time
        percent_processing = processing_duration / total_runtime
        vcons_rate = vcons_count / processing_duration if processing_duration > 0 else 0
        vcons_bytes_rate = vcons_bytes / processing_duration if processing_duration > 0 else 0
        vcons_bytes_rate_mb = vcons_bytes_rate / 1024 / 1024
        
        actually_print_status(program, vcons_count, vcons_bytes, is_blocking, is_alive, processing_duration, total_runtime, percent_processing, vcons_rate, vcons_bytes_rate_mb, vcons_duration)
        line_number += 1
    
    # Print GPU statistics at the end
    print_gpu_stats()

def run(stats_queue):
    status = {}
    while True:
        try: 
            print_status(status)
            time.sleep(settings.status_update_seconds)
            measurement = stats_queue.get(block=False)
            program = measurement["program"]
            name = measurement["name"]
            value = measurement["value"]
            
            if program not in status:
                status[program] = {
                    "start_time": time.time(), 
                    "vcons_count": 0, 
                    "vcons_bytes": 0, 
                    "vcons_duration": 0,
                }
            if name == "blocking_time_start":
                status[program]["blocking_start"] = value
            elif name == "blocking_time_stop":
                blocking_start = status[program]["blocking_start"]
                blocking_duration = value - blocking_start
                status[program]["blocking_duration"] += blocking_duration
                status[program]["blocking_start"] = None
            else:
                status[program][name] = value
                
            
        except Empty:
            continue
