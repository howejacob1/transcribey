from utils import print_gpu_memory_usage, reset_gpu_memory_stats
import torch
import gc
from ai import AIModel, load_and_resample_wavs
import torchaudio
from audio import get_valid_wav_files
import os

# Path to the test FLAC file
# (Update this path if needed)
target_file = "../openslr-12/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac"
fake_wavs_dir = os.path.expanduser("~/conserver/fake_wavs/")


def test_gpu_memory():
    print("===== GPU Memory Test: Initial State =====")
    reset_gpu_memory_stats()
    assert torch.cuda.is_available()
    print_gpu_memory_usage()

    # 1. Load FLAC file into GPU
    print("\n===== Load FLAC into GPU =====")
    wav, sr = torchaudio.load(target_file)
    wav_gpu = wav.to('cuda')
    print_gpu_memory_usage()

    # 2. Load FLAC file into CPU (already loaded above)
    print("\n===== Load FLAC into CPU =====")
    wav_cpu = wav.to('cpu')
    print_gpu_memory_usage()

    print("\n===== Load FLAC into GPU (again) =====")
    wav_gpu = wav.to('cuda')
    print_gpu_memory_usage()

    # 3. Load Whisper model
    print("\n===== Load Whisper Model =====")
    ai = AIModel()
    ai.load_lang_detect()
    print_gpu_memory_usage()

    # 4. Identify language (first time, with torch.no_grad)
    print("\n===== Identify Language (1st, torch.no_grad) =====")
    with torch.no_grad():
        langs1 = ai.identify_languages([target_file])
    print_gpu_memory_usage()

    # 5. Identify language (10 times, with torch.no_grad)
    print("\n===== Identify Language (10 times, torch.no_grad) =====")
    with torch.no_grad():
        for i in range(1, 10):
            langs2 = ai.identify_languages([target_file])
    print_gpu_memory_usage()

    print("\n===== Unload AI  =====")
    ai.unload()
    print_gpu_memory_usage()

    # 6. Test with 100 files from fake_wavs
    print("\n===== Load 100 files from fake_wavs =====")
    valid_wavs = list(get_valid_wav_files(fake_wavs_dir).values())
    test_wavs = valid_wavs[:100]
    print(f"Loaded {len(test_wavs)} valid wav/flac files from {fake_wavs_dir}")
    print_gpu_memory_usage()

    print("\n===== Load Whisper Model for 100 files =====")
    ai = AIModel()
    ai.load_lang_detect()
    print_gpu_memory_usage()

    print("\n===== Identify Language for 100 files =====")
    langs_batch = ai.identify_languages(test_wavs)
    print_gpu_memory_usage()

    print("\n===== Identify Language for 100 files (torch.no_grad) =====")
    with torch.no_grad():
        langs_batch = ai.identify_languages(test_wavs)
    print_gpu_memory_usage()

    print("\n===== Transcribe 100 files (torch.no_grad) =====")
    ai.load_en_transcription()
    with torch.no_grad():
        trans_batch = ai.transcribe(test_wavs, english_only=True)
    print_gpu_memory_usage()

    print("\n===== Unload AI after batch =====")
    ai.unload()
    print_gpu_memory_usage()

    print("\n===== delete variables =====")
    del wav, wav_gpu, wav_cpu, ai, langs1, langs2, valid_wavs, test_wavs, langs_batch, trans_batch
    print_gpu_memory_usage()
    print("\n===== gc =====")
    gc.collect()
    print_gpu_memory_usage()
    print("\n===== empty_cache =====")
    torch.cuda.empty_cache()
    print_gpu_memory_usage()
    print("\n===== sync =====")
    torch.cuda.synchronize()
    print_gpu_memory_usage()

    print("\n===== load all models and some of wavs =====")
    lang_detect_model = AIModel()
    lang_detect_model.load_lang_detect()
    print_gpu_memory_usage()
    en_transcription_model = AIModel()
    en_transcription_model.load_en_transcription()
    print_gpu_memory_usage()
    non_en_transcription_model = AIModel()
    non_en_transcription_model.load_non_en_transcription()
    print_gpu_memory_usage()
    print("\n===== unload all models =====")
    lang_detect_model.unload()
    en_transcription_model.unload()
    non_en_transcription_model.unload()
    print_gpu_memory_usage()
    print("\n===== gc =====")
    gc.collect()
    print_gpu_memory_usage()
    print("\n===== empty_cache =====")
    torch.cuda.empty_cache()
    print_gpu_memory_usage()
    print("\n===== sync =====")
    torch.cuda.synchronize()
    print_gpu_memory_usage()
    print("\n===== done =====")
    

if __name__ == "__main__":
    test_gpu_memory()