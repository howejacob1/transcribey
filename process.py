import ctypes
import os
import logging
import torch.multiprocessing as multiprocessing
import signal
import sys
import threading
import time

try:
    from setproctitle import setproctitle
    HAS_SETPROCTITLE = True
except ImportError:
    HAS_SETPROCTITLE = False

def block_until_threads_finish(threads):
    for thread in threads:
        thread.join()

def block_until_processes_finish(processes):
    for process in processes:
        process.join()

def setup_signal_handlers():
    """Set up signal handlers that raise ShutdownException"""
    def signal_handler(signum, frame):
        logging.info(f"Received signal {signum}, requesting shutdown...")
        raise ShutdownException(f"Shutdown requested via signal {signum}")
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

def set_process_title_for_function(func):
    """Set the process title based on function name"""
    if HAS_SETPROCTITLE:
        process_name = f"transcribey-{func.__name__}"
        setproctitle(process_name)
        print(f"[PID {os.getpid()}] Set process title to: {process_name}")
    else:
        print(f"[PID {os.getpid()}] setproctitle not available, process will show as 'python' in system tools")

def process_wrapper_with_signal_handlers(target_and_args):
    """Top-level wrapper function that can be pickled for multiprocessing spawn"""
    target, args = target_and_args
    
    # Set the process title to the function name so it shows up in nvidia-smi and ps
    set_process_title_for_function(target)
    
    setup_signal_handlers()
    return target(*args)

def start_process(target, args):
    assert multiprocessing.get_start_method() == "spawn", f"Expected spawn, got {multiprocessing.get_start_method()}"
    # Pack target and args together so they can be unpacked in the wrapper
    target_and_args = (target, args)
    process = multiprocessing.Process(target=process_wrapper_with_signal_handlers, args=(target_and_args,))
    process.name = target.__name__  # Use function name instead of str(target)
    process.start()
    return process

class ShutdownException(Exception):
    """Custom exception raised when shutdown is requested"""
    pass

def our_thread_name():
    return threading.current_thread().name

def our_process_name():
    return multiprocessing.current_process().name

def our_program_name():
    """Get a unique identifier for current thread/process"""
    return f"{our_process_name()}.{our_thread_name()}"

def make_a_shutdown_exception_happen(thread: threading.Thread):
    make_an_exception_happen(thread, ShutdownException("Thread shutdown requested"))

def tell_thread_to_shutdown(thread: threading.Thread):
    if thread.is_alive():
        make_a_shutdown_exception_happen(thread)

def get_thread_id(thread: threading.Thread):
    return thread.ident

def make_an_exception_happen(thread: threading.Thread, exception: Exception):
    thread_id = get_thread_id(thread)
    if thread_id is None: #idk why we check
        return
    ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, ctypes.py_object(exception))
    
    # This is a CPython-specific hack using ctypes
    # It forces an exception in the target thread
    exc_type = type(exception)
    ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_long(thread_id), 
        ctypes.py_object(exc_type)
    )
    
    if ret > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, None)

def block_until_threads_and_processes_finish(threads_and_processes: list[threading.Thread | multiprocessing.Process]):
    for program in threads_and_processes:
        program.join()

def is_thread(program):
    return isinstance(program, threading.Thread)

def is_process(program):
    return isinstance(program, multiprocessing.Process)

def tell_thing_to_shutdown(thing: threading.Thread | multiprocessing.Process):
    if is_thread(thing):
        tell_thread_to_shutdown(thing)
    elif is_process(thing):
        os.system('kill %d' % thing.pid)

def stop_threads_and_processes(threads_and_processes: list[threading.Thread | multiprocessing.Process], block=True):
    for program in threads_and_processes:
        tell_thing_to_shutdown(program)
    if block:
        block_until_threads_and_processes_finish(threads_and_processes)
