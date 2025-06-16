import multiprocessing
import time
from contextlib import contextmanager
from queue import Empty

try:
    import GPUtil
    gpu_available = True
except ImportError:
    gpu_available = False

try:
    import psutil
    psutil_available = True
except ImportError:
    psutil_available = False

from process import our_program_name
import settings
from utils import seconds_to_days, seconds_to_hours, seconds_to_minutes

def make_stats_queue():
    return multiprocessing.Queue()

def add(stats_queue, key, value):
    stats_queue.put({"program": our_program_name(),
                     "name": key,
                     "value": value})

def start_blocking(stats_queue):
    add(stats_queue, "start_blocking", time.time())

def stop_blocking(stats_queue):
    add(stats_queue, "stop_blocking", time.time())

def bytes(stats_queue, bytes):
    add(stats_queue, "bytes", bytes)

def duration(stats_queue, duration):
    add(stats_queue, "duration", duration)

def count(stats_queue, count=1):
    add(stats_queue, "count", count)

def start(stats_queue):
    add(stats_queue, "start", time.time())

def stop(stats_queue):
    add(stats_queue, "stop", time.time())

@contextmanager
def with_blocking_time(stats_queue):
    """Am fancy because I want time.time accuracy but I want stuff readable."""
    start_blocking(stats_queue)
    try:
        yield
    finally:
        stop_blocking(stats_queue)

def actually_print_status(program,
                         count,
                         bytes,
                         is_blocking,
                         is_alive,
                         processing_duration,
                         total_runtime,
                         percent_running,
                         rate,
                         bytes_rate_mb, 
                         duration):
    status_char = "🔴"
    if is_blocking:
        status_char = "⚪"
    elif is_alive:
        status_char = "🟢"
    
    # Convert bytes to MB for display
    bytes_mb = bytes / 1024 / 1024
    rtf = duration / processing_duration if processing_duration > 0 else 0
    duration_minutes = seconds_to_minutes(duration)
    total_runtime_minutes = seconds_to_minutes(total_runtime)

    # Debug RTF calculation - uncomment to debug
    # if program == "transcribe.MainThread" and vcons_count > 0:
    #     print(f"DEBUG {program}: vcons_duration={vcons_duration:.3f}s, processing_duration={processing_duration:.3f}s, rtf={rtf:.3f}")
    print(f"{status_char} {program:28} | "
          f"{rtf:7.2f}x {count:9,} {rate:7.1f}/s {bytes_rate_mb:9.2f}MB/s | "
          f"{bytes_mb:10.2f}MB "
          f"{duration_minutes:13.1f}m | ({percent_running:6.1f}%)")

def print_gpu_stats():
    """Print GPU memory and power statistics"""
    if gpu_available:
        gpus = GPUtil.getGPUs()
        if not gpus:
            print("No GPU detected.")
            return False
        gpu = gpus[0]
        memory_used = gpu.memoryUsed
        memory_total = gpu.memoryTotal
        memory_percent = (memory_used / memory_total) * 100 if memory_total > 0 else 0
        temperature = gpu.temperature
        load = gpu.load * 100
        print(f"Mem: {memory_used:,}MB/{memory_total:,}MB ({memory_percent:.1f}%) | "
              f"Load: {load:.1f}% | "
              f"Temp: {temperature}°C")
        return True
    else:
        print("GPUtil not available.")
        return False

def print_cpu_stats():
    """Print basic CPU statistics."""
    if psutil_available:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        print(f"CPU Usage: {cpu_percent:.1f}% | RAM: {mem.used // (1024**2):,}MB/{mem.total // (1024**2):,}MB ({mem.percent:.1f}%)")
    else:
        print("CPU stats: psutil not available.")

def print_xpu_stats():
    """Print GPU stats if available, otherwise CPU stats."""
    if not print_gpu_stats():
        print_cpu_stats()

def print_status(status):
    # Clear screen and move cursor to top
    print("\033[2J\033[H", end="")    
    
    line_number = 0
    for program, measurements in status.items():
        count = float(measurements["count"])
        bytes = float(measurements["bytes"])
        blocking_duration = float(measurements["blocking_duration"])
        if measurements.get("start_blocking"):
            start_blocking = float(measurements["start_blocking"])
        else:
            start_blocking = None
        start = float(measurements["start"])
        if measurements.get("stop"):
            stop = float(measurements["stop"])
        else:
            stop = None
        duration = float(measurements["duration"])
        is_alive = stop is None
        

        is_blocking =  start_blocking is not None
        # Calculate total runtime
        if stop:
            total_runtime = stop - start
        else:
            total_runtime = time.time() - start
        
        # Calculate total blocking time (including current blocking session if any)
        total_blocking_time = blocking_duration
        if not stop:
            if is_blocking:
                cur_blocking_duration = time.time() - start_blocking
                total_blocking_time += cur_blocking_duration

        # Calculate processing time (time not spent blocking)
        processing_duration = total_runtime - total_blocking_time
        
        # Calculate percentage of time spent blocking
        percent_running = (processing_duration / total_runtime) * 100
        
        rate = count / processing_duration
        bytes_rate = bytes / processing_duration
        bytes_rate_mb = bytes_rate / 1024 / 1024
        
        actually_print_status(program, count, bytes, is_blocking, is_alive, processing_duration, total_runtime, percent_running, rate, bytes_rate_mb, duration)
        line_number += 1
    
    # Print XPU statistics at the end
    print_xpu_stats()

global status 
status = {}

def run(stats_queue):
    count = 0
    start_time = time.time()
    while True:
        count += 1
        # else:
        #     time.sleep(settings.status_update_seconds)
        if time.time() - start_time > 3:
            print_status(status)
            start_time = time.time()
            continue
        try: 
            measurement = stats_queue.get(timeout=settings.status_update_seconds)
            #print(f"Measurement: {measurement}")
            program = measurement["program"]
            name = measurement["name"]
            value = measurement["value"]
            
            if status.get(program, None) is None:
                #print(f"New program: {program}")
                assert name == "start"
                status[program] = {
                    "start": float(value), 
                    "count": 0.0, 
                    "bytes": 0.0, 
                    "duration": 0.0,
                    "start_blocking": None,
                    "blocking_duration": 0.0,
                    "stop": None,
                }
            if name == "start":
                status[program]["start"] = value
            elif name == "stop":
                status[program]["stop"] = value
            elif name == "count":
                status[program]["count"] += value
            elif name == "bytes":
                status[program]["bytes"] += value
            elif name == "duration":
                status[program]["duration"] += value
            elif name == "start_blocking":
                status[program]["start_blocking"] = value
            elif name == "stop_blocking":
                start_blocking = status[program]["start_blocking"]
                if start_blocking is not None:
                    this_blocking_duration = time.time() - start_blocking
                    status[program]["blocking_duration"] += this_blocking_duration
                    status[program]["start_blocking"] = None
        except Empty:
            continue
