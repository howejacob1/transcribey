import os
import random
import shutil

FAKE_WAVS_DIR = 'fake_wavs'
TARGET_COUNT = 10000

# Get all current wav files
wav_files = [f for f in os.listdir(FAKE_WAVS_DIR) if f.lower().endswith('.wav')]
existing_count = len(wav_files)

if existing_count == 0:
    raise RuntimeError('No wav files found in fake_wavs to copy!')

# Generate a set of all existing filenames for fast lookup
existing_filenames = set(wav_files)

# Function to generate a unique filename

def generate_unique_filename(existing_filenames, ext='.wav'):
    while True:
        new_name = f'copy_{random.randint(0, 99999999):08d}{ext}'
        if new_name not in existing_filenames:
            return new_name

copies_needed = TARGET_COUNT - existing_count
print(f"Existing files: {existing_count}. Need to create {copies_needed} more.")

for _ in range(copies_needed):
    src_file = random.choice(wav_files)
    new_file = generate_unique_filename(existing_filenames)
    shutil.copy(os.path.join(FAKE_WAVS_DIR, src_file), os.path.join(FAKE_WAVS_DIR, new_file))
    existing_filenames.add(new_file)
    wav_files.append(new_file)  # So new files can also be copied

print(f"Done. There are now {len(existing_filenames)} wav files in {FAKE_WAVS_DIR}.") 