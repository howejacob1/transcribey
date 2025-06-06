import torchaudio
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

def is_valid(file_path):
    duration = get_duration(file_path)
    return duration is not None

def load_to_cpu(filename):
    audio, sample_rate = torchaudio.load(filename)
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

def is_mono(audio_data):
    return audio_data.shape[0] == 1

def convert_to_mono(audio_data):
    audio_data = audio_data.mean(dim=0, keepdim=True)
    return audio_data

def cpu_cores_for_mono_conversion():
    return num_cores() - 2

def cpu_cores_for_vad():
    return num_cores() - 2