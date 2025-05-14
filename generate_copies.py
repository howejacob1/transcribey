import os
import random
import shutil
from pathlib import Path

FAKE_WAVS_DIR = 'fake_wavs'
TARGET_COUNT = 200000
MAX_NESTING = 7

# Recursively find all eligible wav files (not starting with 'copy')
all_wav_files = []
for root, dirs, files in os.walk(FAKE_WAVS_DIR):
    for f in files:
        if f.lower().endswith('.wav') and not f.startswith('copy'):
            all_wav_files.append(os.path.join(root, f))

existing_count = 0
for root, dirs, files in os.walk(FAKE_WAVS_DIR):
    for f in files:
        if f.lower().endswith('.wav'):
            existing_count += 1

if not all_wav_files:
    raise RuntimeError('No eligible wav files found in fake_wavs to copy!')

# Set of all existing relative file paths for fast lookup
existing_relpaths = set()
for root, dirs, files in os.walk(FAKE_WAVS_DIR):
    for f in files:
        if f.lower().endswith('.wav'):
            relpath = os.path.relpath(os.path.join(root, f), FAKE_WAVS_DIR)
            existing_relpaths.add(relpath)

def generate_random_nested_path(existing_relpaths, ext='.wav'):
    while True:
        depth = random.randint(1, MAX_NESTING)
        parts = [f"level{random.randint(0, 9999):04d}" for _ in range(depth)]
        fname = f"copy_{random.randint(0, 99999999):08d}{ext}"
        relpath = os.path.join(*parts, fname)
        if relpath not in existing_relpaths:
            return relpath

copies_needed = TARGET_COUNT - existing_count
print(f"Existing files: {existing_count}. Need to create {copies_needed} more.")

for i in range(copies_needed):
    src_file = random.choice(all_wav_files)
    rel_dest = generate_random_nested_path(existing_relpaths)
    dest_path = os.path.join(FAKE_WAVS_DIR, rel_dest)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.copy(src_file, dest_path)
    existing_relpaths.add(rel_dest)
    # Optionally, add the new file to all_wav_files to allow recursive copying
    all_wav_files.append(dest_path)
    if (i+1) % 1000 == 0:
        print(f"{i+1} copies made...")

print(f"Done. There are now {sum(len(files) for _, _, files in os.walk(FAKE_WAVS_DIR))} files in {FAKE_WAVS_DIR}.") 