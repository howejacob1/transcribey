import whisper
import random
import os

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

# Run language detection
result = model.transcribe(wav_path, task='transcribe', language=None, verbose=True)

# Print detected language
print(f"Detected language: {result.get('language', 'unknown')}") 