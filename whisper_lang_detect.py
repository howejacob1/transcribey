import whisper
import random
import os
from collections import Counter
import time

# Directory containing wav files
dir = 'fake_wavs'

def wav_file_generator(directory):
    """
    Generator that recursively walks a directory and yields .wav file paths one at a time.
    Retains state between calls.
    """
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith('.wav'):
                yield os.path.join(root, f)

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

start_time = time.time()

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

end_time = time.time()

# Print detected language rates
print("\nLanguage detection rates after sampling:")
total = sum(lang_counter.values())
for lang, count in lang_counter.most_common():
    print(f"{lang}: {count} ({count/total:.2%})")

print(f"\nTotal time: {end_time - start_time:.2f} seconds for {len(wav_files)} files.")
print(f"Average time per file: {(end_time - start_time)/len(wav_files):.2f} seconds.") 