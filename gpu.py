import torch
import gc
import settings

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def we_have_a_gpu():
    return get_device() != "cpu"

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
    return torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory

def gpu_ram_allocated_bytes():
    return torch.cuda.memory_allocated(get_device())

def gpu_ram_free_bytes():
    return gpu_ram_total_bytes() - gpu_ram_allocated_bytes()

def batch_bytes():
    free_bytes = gpu_ram_free_bytes()
    return free_bytes // 8
    
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
            gc.collect()