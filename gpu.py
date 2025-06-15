import gc
import torch

try:
    import GPUtil
except ModuleNotFoundError:
    GPUtil = None

import settings

def get_device():
    """Return 'cuda' when CUDA is available else 'cpu'.
    This helper is used throughout the codebase.
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def we_have_a_gpu():
    return torch.cuda.is_available()

def tensor_on_gpu(tensor):
    return tensor.device.type == "cuda"

def tensor_on_cpu(tensor):
    return tensor.device.type == "cpu"

def move_to_gpu(obj):
    if we_have_a_gpu():
        device = get_device()
        move_to_gpu = False
        if isinstance(obj, torch.nn.Module):
            # Check if the model is already on the GPU
            if not next(obj.parameters()).is_cuda:
                move_to_gpu = True
        elif isinstance(obj, torch.Tensor):
            if tensor_on_cpu(obj):
                obj = obj.pin_memory()
                move_to_gpu = True
        if move_to_gpu:
            obj = obj.to(device, non_blocking=True)
    return obj

def move_to_gpu_maybe(obj):
    if we_have_a_gpu():
        obj = move_to_gpu(obj)
    return obj

def gpu_ram_total_bytes():
    """Return total GPU memory in bytes or 0 if no GPU.

    When CUDA is available but GPUtil is absent, fall back to torch.
    When CUDA is unavailable, return 0 so downstream code can decide.
    """
    if not torch.cuda.is_available():
        return 0
    try:
        return torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    except Exception:  # pragma: no cover – catch any CUDA querying issues
        return 0

def gpu_ram_allocated_bytes():
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.memory_allocated(get_device())

def gpu_ram_free_bytes():
    """Return free GPU memory in bytes or a reasonable default on CPU-only systems.

    The logic tries, in order:
    1. If no CUDA device, return a large constant (16 GB) to allow batching.
    2. Use GPUtil if available to query precise usage.
    3. Fall back to torch memory stats.
    """
    if not torch.cuda.is_available():
        # Assume plenty of system RAM; 16 GB is a safe, conservative default.
        return 16 * (1024 ** 3)

    # If GPUtil is present, try to get memory info from the first GPU.
    if GPUtil is not None:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                memory_used_bytes = gpu.memoryUsed * 1024 * 1024
                total = gpu_ram_total_bytes()
                free = total - memory_used_bytes
                return max(free, 0)
        except Exception:  # pragma: no cover – GPUtil may fail unexpectedly
            pass

    # Fallback: use torch's allocated memory to estimate free memory.
    total = gpu_ram_total_bytes()
    allocated = gpu_ram_allocated_bytes()
    free = total - allocated
    return max(free, 0)

def batch_bytes():
    """Return a batch size in bytes based on free GPU (or system) memory.
    For CUDA systems we use half of the reported free GPU memory.
    On CPU-only systems we default to 256 MB to keep memory usage reasonable.
    """
    free_bytes = gpu_ram_free_bytes()
    if not torch.cuda.is_available():
        # Fixed batch size on CPU-only setups.
        return 256 * (1024 ** 2)  # 256 MB
    # On GPU systems, be conservative and use half of the free memory.
    return max(free_bytes // 2, 32 * (1024 ** 2))  # At least 32 MB

def max_gpu_memory_usage():
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        max_allocated = torch.cuda.max_memory_allocated(current_device)
        return max_allocated
    return None

def print_gpu_memory_usage():
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(current_device)
        max_allocated = torch.cuda.max_memory_allocated(current_device)
        max_reserved = torch.cuda.max_memory_reserved(current_device)
        print(f"Peak - Allocated: {max_allocated/(1024**3):.2f} GB, Reserved: {max_reserved/(1024**3):.2f} GB, Allocated: {allocated/(1024**3):.2f} GB")
    else:
        print("CUDA not available")

def reset_gpu_memory_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        print("GPU memory peak statistics reset")

def gc_collect_maybe():
    if we_have_a_gpu():
        if gpu_ram_free_bytes() < settings.gc_limit_bytes:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()