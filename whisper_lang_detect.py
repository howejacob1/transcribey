import whisper
import random
import os
from collections import Counter

# Directory containing wav files
dir = 'fake_wavs'

# Recursively find up to 100 wav files in the directory and subdirectories
wav_files = []
for root, _, files in os.walk(dir):
    for f in files:
        if f.endswith('.wav'):
            wav_files.append(os.path.join(root, f))
            if len(wav_files) >= 100:
                break
    if len(wav_files) >= 100:
        break
if not wav_files:
    raise RuntimeError('No wav files found in directory!')

# Shuffle the list to randomize selection
random.shuffle(wav_files)

# Load the tiny Whisper model
model = whisper.load_model('tiny')

# Counter for detected languages
lang_counter = Counter()

for i, wav_path in enumerate(wav_files):
    # Load and preprocess the audio to a mel spectrogram
    audio = whisper.load_audio(wav_path)
    mel = whisper.log_mel_spectrogram(audio)
    mel = whisper.pad_or_trim(mel, 3000)  # 3000 is N_FRAMES for Whisper
    # Run language detection
    _, probs = model.detect_language(mel)
    language = max(probs, key=probs.get)
    lang_counter[language] += 1
    if (i+1) % 10 == 0:
        print(f"Completed {i+1} iterations...")

# Print detected language rates
print("\nLanguage detection rates after sampling:")
total = sum(lang_counter.values())
for lang, count in lang_counter.most_common():
    print(f"{lang}: {count} ({count/total:.2%})") 