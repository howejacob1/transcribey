import os
import torchaudio
import binpacking
import gpu
from gpu import we_have_a_gpu
import settings 
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
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

def is_valid(file_path):
    duration = get_duration(file_path)
    return duration is not None

def move_to_gpu_maybe(audio):
    if we_have_a_gpu():
        gpu.move_to_gpu_maybe(audio)
        audio.to(gpu.get_device(), non_blocking=True)

def load_to_cpu(filename):
    audio, sample_rate = torchaudio.load(filename)
    audio.pin_memory()
    audio.contiguous()
    return audio, sample_rate

def resample_audio(audio, sample_rate):
    target_sample_rate = settings.sample_rate
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        resampled_audio = resampler(audio)
        return resampled_audio
    else:
        return audio

def get_size(audio):
    return audio.element_size() * audio.numel()

def audio_is_mono(audio_data):
    return audio_data.shape[0] == 1

def convert_to_mono(audio_data):
    audio_data = audio_data.mean(dim=0, keepdim=True)
    return audio_data

def cpu_cores_for_mono_conversion():
    return num_cores() - 2

def apply_vad(audio_data, vad):
    target_sample_rate = settings.sample_rate
    vad = torchaudio.transforms.Vad(sample_rate=target_sample_rate, trigger_level=0.5)
    return vad(audio_data)

def cpu_cores_for_vad():
    return num_cores() - 2