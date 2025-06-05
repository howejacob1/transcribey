import torch

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def we_have_a_gpu():
    return get_device() != "cpu"

def move_to_gpu(tensor):
    if we_have_a_gpu():
        tensor.to(get_device(), non_blocking=True)
        tensor.pin_memory()
        tensor.contiguous()

def move_to_gpu_maybe(tensor):
    if we_have_a_gpu():
        move_to_gpu(tensor)

def gpu_ram_total_bytes():
    return torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory

def gpu_ram_allocated_bytes():
    return torch.cuda.memory_allocated(get_device())

def gpu_ram_free_bytes():
    return gpu_ram_total_bytes() - gpu_ram_allocated_bytes()

def batch_bytes():
    free_bytes = gpu_ram_free_bytes()
    return free_bytes // 64
    
def max_gpu_memory_usage():
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        max_allocated = torch.cuda.max_memory_allocated(current_device)
        return max_allocated
    return None

def print_gpu_memory_usage():
    """Print current GPU memory usage for debugging."""
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(current_device)
        # reserved = torch.cuda.memory_reserved(current_device)
        max_allocated = torch.cuda.max_memory_allocated(current_device)
        max_reserved = torch.cuda.max_memory_reserved(current_device)
        # total = torch.cuda.get_device_properties(current_device).total_memory
        
        print(f"Peak - Allocated: {max_allocated/(1024**3):.2f} GB, Reserved: {max_reserved/(1024**3):.2f} GB, Allocated: {allocated/(1024**3):.2f} GB")
    else:
        print("CUDA not available")

def reset_gpu_memory_stats():
    """Reset peak GPU memory statistics for more accurate tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        print("GPU memory peak statistics reset")
