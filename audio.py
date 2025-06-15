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

def get_size(audio: torch.Tensor):
    return audio.element_size() * audio.numel()

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

def pad_audio(audio_data: torch.Tensor, sample_rate: int, duration: float):
    """Pads audio data on the right to ensure it has a minimum duration."""
    target_samples = int(duration * sample_rate)
    
    if audio_data.shape[1] < target_samples:
        padding_needed = target_samples - audio_data.shape[1]
        return torch.nn.functional.pad(audio_data, (0, padding_needed))
    
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