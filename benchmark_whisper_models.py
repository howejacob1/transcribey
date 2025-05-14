import warnings
warnings.filterwarnings("ignore", message="The input name `inputs` is deprecated. Please make sure to use `input_features` instead.", category=FutureWarning)

import time
import torch
from transformers import pipeline
import torchaudio
from difflib import SequenceMatcher
from transformers.utils import logging
logging.set_verbosity_error()
from jiwer import wer

# Path to audio
audio_path = 'training/transcription-example-2.mp3'

# List of HuggingFace model IDs, largest to smallest
model_ids = [
    'openai/whisper-large-v2',
    'distil-whisper/distil-large-v3',
    'openai/whisper-large',
    'openai/whisper-medium',
    'openai/whisper-medium.en',
    'openai/whisper-small',
    'openai/whisper-small.en',
    'openai/whisper-base',
    'openai/whisper-base.en',
    'openai/whisper-tiny',
    'openai/whisper-tiny.en'
]

def get_audio_duration(path):
    info = torchaudio.info(path)
    return info.num_frames / info.sample_rate

audio_duration = get_audio_duration(audio_path)
print("cuda is available? ", torch.cuda.is_available())
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'cuda' if device == 0 else 'cpu'}")

results = []
total_models = len(model_ids)
reference_transcription = None

for idx, model_id in enumerate(model_ids):
    model_name = model_id.split('/')[-1]
    percent = int((idx + 1) / total_models * 100)
    print(f"\n[{idx+1}/{total_models} {percent}%] Loading model: {model_id} ...")
    start_load = time.time()
    asr = pipeline("automatic-speech-recognition", model=model_id, device=device)
    print(f"  Pipeline loaded in {time.time() - start_load:.1f} seconds.")

    print(f"  Transcribing with {model_name}...")
    start = time.time()
    logging.set_verbosity_error()
    result = asr(audio_path, return_timestamps=True)
    logging.set_verbosity_warning()
    elapsed = time.time() - start
    transcription = result["text"]
    print(transcription)
    rtf = audio_duration / elapsed if elapsed > 0 else float('inf')
    # Calculate words per minute (WPM)
    num_words = len(transcription.split())
    elapsed_minutes = elapsed / 60 if elapsed > 0 else float('inf')
    wpm = num_words / elapsed_minutes if elapsed_minutes > 0 else 0
    print(f"  Model {model_name} took {elapsed:.1f} seconds to transcribe {audio_duration:.1f} seconds of audio. Calls per second: {rtf:.2f}")
    print(f"  Words per minute: {wpm:.1f}")

    # Store reference transcription for large-v2
    if model_id == 'openai/whisper-large-v2':
        reference_transcription = transcription
        accuracy = 100.0
        print(f"  Accuracy: 100.0% (Reference transcription set for accuracy comparison.)")
    else:
        # Compute accuracy vs reference using jiwer WER
        if reference_transcription is not None:
            try:
                word_error_rate = wer(reference_transcription, transcription)
                accuracy = (1 - word_error_rate) * 100
                print(f"  Accuracy vs large-v2: {accuracy:.2f}% (WER: {word_error_rate:.3f})")
            except Exception as e:
                print(f"  Could not compute WER: {e}")
                accuracy = 0.0
        else:
            accuracy = 0.0
            print(f"  Accuracy vs large-v2: N/A (reference not available yet)")

    results.append((model_name, elapsed, rtf, wpm, accuracy))

print(f"\nSummary (Transcribe {audio_duration:.1f} seconds of audio):")
print(f"{'Model':<20} {'Time (s)':>10} {'Calls':>10} {'WPM':>10} {'Acc%':>8}")
print('-' * 63)
for model_name, elapsed, rtf, wpm, accuracy in results:
    print(f"{model_name:<20} {elapsed:10.1f} {rtf:10.2f} {wpm:10.1f} {accuracy:8.2f}")