import multiprocessing
import queue
import threading
import time
from typing import List, Optional

import settings
from vcon_class import Vcon

class VconQueue:
    def __init__(self, max_bytes: int = settings.vcon_queue_max_bytes, process: bool = False):
        if multiprocessing:
            self._queue = multiprocessing.Queue()
        else:
            self._queue = queue.Queue()
        self.max_bytes: int = max_bytes
        self.current_bytes: int = 0
        self._lock: multiprocessing.Lock | threading.Lock = None
        self._not_full: multiprocessing.Condition | threading.Condition = None
        if process:
            self._lock = multiprocessing.Lock()
            self._not_full = multiprocessing.Condition(self._lock)
        else:
            self._lock = threading.Lock()
            self._not_full = threading.Condition(self._lock)

    def put(self, vcon: Vcon, block: bool = True, timeout: Optional[float] = None) -> None:
        vcon_size = vcon.size
        
        with self._not_full:
            if timeout is not None:
                endtime = time.time() + timeout
                
            while self.current_bytes + vcon_size > self.max_bytes:
                if not block:
                    raise queue.Full
                if timeout is not None:
                    remaining = endtime - time.time()
                    if remaining <= 0.0:
                        raise queue.Full
                    self._not_full.wait(remaining)
                else:
                    self._not_full.wait()
            
            # Now we have space, add to queue and update size
            self._queue.put(vcon, block=False)  # Won't block since we have space
            self.current_bytes += vcon_size

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Vcon:
        vcon = self._queue.get(block, timeout)
        vcon_size = vcon.size
        
        with self._not_full:
            self.current_bytes -= vcon_size
            self._not_full.notify_all()  # Wake up any waiting put() calls
            
        return vcon

    def put_many(self, vcons: List[Vcon], block: bool = True, timeout: Optional[float] = None) -> None:
        for vcon in vcons:
            self.put(vcon, block, timeout)

    def empty(self) -> bool:
        return self._queue.empty()

    def qsize(self) -> int:
        return self._queue.qsize()
        
    def bytes(self) -> int:
        with self._lock:
            return self.current_bytes
