import logging

import torch
import torchaudio

import gpu
import settings
from utils import num_cores

def get_duration(filename):
    """
    Return the duration in seconds of a wav file using torchaudio.info.
    Returns None if the file is corrupt or unreadable.
    """ 
    try:
        info = torchaudio.info(filename)
        return info.num_frames / info.sample_rate
    except Exception as e:
        return None

def duration(audio_data, sample_rate):
    # Handle both 1D and 2D audio data
    if audio_data.ndim == 1:
        # 1D audio data: shape is [samples]
        return audio_data.shape[0] / sample_rate
    else:
        # 2D audio data: shape is [channels, samples]
        return audio_data.shape[1] / sample_rate

def is_valid(file_path):
    try:
        duration = get_duration(file_path)
        return duration is not None
    except Exception as e:
        logging.info(f"Error in is_valid: {e}")
        return False
    
def load_to_gpu(filename):
    audio_data, sample_rate = torchaudio.load(filename)
    audio_data = gpu.move_to_gpu_maybe(audio_data)
    return audio_data, sample_rate

def load_to_cpu(filename):
    audio, sample_rate = torchaudio.load(filename)
    return audio, sample_rate

def resample(audio: torch.Tensor, 
             original_sample_rate: int, 
             resampler: torchaudio.transforms.Resample | None = None) -> torch.Tensor:
    target_sample_rate = settings.sample_rate
    if original_sample_rate != target_sample_rate:
        if not resampler:
            resampler = torchaudio.transforms.Resample(original_sample_rate, target_sample_rate)
        audio = resampler(audio)
    return audio

def get_size(audio):
    """Return the size in bytes of the audio tensor/array.

    Supports torch.Tensor, NumPy ndarray, CuPy ndarray and other array-like
    objects that expose an ``nbytes`` attribute.
    """

    # PyTorch tensor
    if isinstance(audio, torch.Tensor):
        return audio.element_size() * audio.numel()

    # Objects (e.g. NumPy/CuPy arrays) exposing ``nbytes``
    nbytes = getattr(audio, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)

    # Fallback: use ``size`` × ``itemsize`` if available
    if hasattr(audio, "size") and hasattr(audio, "itemsize"):
        try:
            return int(audio.size * audio.itemsize)  # type: ignore[attr-defined]
        except Exception:
            pass

    # Unknown type – return 0 to avoid hard failure
    return 0

def is_mono(audio_data: torch.Tensor):
    return audio_data.shape[0] == 1

def convert_to_mono(audio_data: torch.Tensor):
    audio_data = audio_data.mean(dim=0, keepdim=True)
    return audio_data

def cpu_cores_for_mono_conversion():
    return num_cores() - 2

def cpu_cores_for_resampling():
    return num_cores() - 2

def cpu_cores_for_vad():
    return num_cores() - 2

def pad_audio(audio_data, sample_rate: int, duration: float):
    """Pads audio data on the right to ensure it has a minimum duration."""
    import cupy as cp
    
    target_samples = int(duration * sample_rate)
    
    # Handle CuPy arrays by converting to PyTorch tensor
    is_cupy = hasattr(audio_data, 'get')  # CuPy arrays have .get() method
    if is_cupy:
        # Convert CuPy array to PyTorch tensor
        if hasattr(audio_data, 'device'):
            # If CuPy array is on GPU, create GPU tensor
            audio_tensor = torch.as_tensor(audio_data, device='cuda')
        else:
            # Convert to CPU tensor
            audio_tensor = torch.from_numpy(audio_data.get())
    else:
        audio_tensor = audio_data
    
    # Handle both 1D and 2D audio data
    if audio_tensor.ndim == 1:
        # 1D audio data: shape is [samples]
        current_samples = audio_tensor.shape[0]
    else:
        # 2D audio data: shape is [channels, samples]
        current_samples = audio_tensor.shape[1]
    
    if current_samples < target_samples:
        padding_needed = target_samples - current_samples
        if audio_tensor.ndim == 1:
            # For 1D: pad on the right (last dimension)
            padded_tensor = torch.nn.functional.pad(audio_tensor, (0, padding_needed))
        else:
            # For 2D: pad on the right (last dimension)
            padded_tensor = torch.nn.functional.pad(audio_tensor, (0, padding_needed))
        
        # Convert back to CuPy if original was CuPy
        if is_cupy:
            # CuPy cannot convert PyTorch tensors directly; go through NumPy first.
            return cp.asarray(padded_tensor.detach().cpu().numpy())
        else:
            return padded_tensor
    
    return audio_data

def cpu_cores_for_preprocessing():
    return num_cores() - 1

def ensure_mono(audio_data: torch.Tensor) -> torch.Tensor:
    if not is_mono(audio_data):
        audio_data = convert_to_mono(audio_data)
    return audio_data

def vad(audio_data: torch.Tensor) -> torch.Tensor:
    vad_fun = torchaudio.transforms.Vad(sample_rate=settings.sample_rate, trigger_level=0.5)
    return vad_fun(audio_data)

def format_bytes(size):
    # 2**10 = 1024
    for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
        if size < 1024.0:
            return f"{size:3.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"