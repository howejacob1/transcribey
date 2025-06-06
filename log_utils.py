import logging
import time
from contextlib import contextmanager
from utils import seconds_to_ydhms
import gpu

def info_header(message):
    logging.info(f"--------------------{message}.--------------------")

@contextmanager
def with_timing(message):
    info_header(message)
    start_time = time.time()
    yield
    end_time = time.time()
    gpu_ram_used_gb = gpu.gpu_ram_allocated_bytes() / (1024**3)
    duration = end_time - start_time
    duration_str = seconds_to_ydhms(duration)
    logging.info(f"done {duration_str} {gpu_ram_used_gb:.2f}GB")