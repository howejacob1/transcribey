import whisper
import random
import os
import torch

# Directory containing wav files
dir = 'fake_wavs/level7151'

# List all wav files in the directory
wav_files = [f for f in os.listdir(dir) if f.endswith('.wav')]
if not wav_files:
    raise RuntimeError('No wav files found in directory!')

# Pick a random wav file
wav_path = os.path.join(dir, random.choice(wav_files))
print(f'Using wav file: {wav_path}')

# Load the tiny Whisper model
model = whisper.load_model('tiny')

# Load and preprocess the audio to a mel spectrogram
audio = whisper.load_audio(wav_path)
mel = whisper.log_mel_spectrogram(audio)
mel = whisper.pad_or_trim(mel, 3000)  # 3000 is N_FRAMES for Whisper

# Run language detection
_, probs = model.detect_language(mel)
language = max(probs, key=probs.get)

# Print detected language
print(f"Detected language: {language}") 