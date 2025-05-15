# Architecture: 

# We will be getting several hard drives, each one has tons of wavs.
# Load some set of wavs into ram. Always fill up.
# For each wav, detect language.
# For each wav, transcribe.
# Then, save the vcon.

# Our job is to transcribe each wav file to a vcon. 
# We will put each back onto the target hard drive.

# Exo is great-- However, it can't do nemo files. 

# I am stuck with nemo files. I need the acceleration. 
# Here's what we'll do-- 

# Find wav files- load into buffer (in ram)

# # Detect language on all in batch
# Transcribe all (by language) in each batch
# Save each into a vcon
# Make a vcon for each file. 

# Make a vcon for each file

# In background, always load up wav files into buffer

# Detect language on each, save to vcon

# If english, transcribe with fastest english model on each

# Else, add the wav to the list of files to be transcribed. 

# Write the vcon and remove the wav if successful.

# Now, go through all non-english ones and use fastest model for each language. 

from utils import get_all_filenames, wav_file_generator
import time
import os
import transcription_models
from wav_preload import preload_wavs_threaded
import shutil


model_comparison = [
    {
        "model": "nvidia/parakeet-tdt-0.6b-v2",
        "languages": ["en"],
        "WER": 6.05,
        "RTFx": 3386.02,
        "size_gb": 2.47,
        "note": "Most accurate english model"
    },
    {
        "model": "nvidia/parakeet-tdt_ctc-110m",
        "languages": ["en"],
        "WER": 7.49,
        "RTFx": 5345.14,
        "size_gb": 0.5,
        "note": "Fastest English Model"
    },
    {
        "model": "nvidia/canary-1b-flash",
        "languages": ["en", "de", "fr", "es"],
        "WER": 6.35,
        "RTFx": 1045.75,
        "size_gb": 3.54,
        "note": "Fastest German, French, and Spanish Model"
    }
]



languages = ["en", "es"]

def select_transcription_model(language, prioritize_speed=True):
    """
    Select the best model for the given language.
    If prioritize_speed is True, select the fastest model (highest RTFx, lowest size).
    Otherwise, select the most accurate model (lowest WER).
    Returns the model name as a string, or None if no suitable model is found.
    """
    candidates = [m for m in model_comparison if language in m["languages"]]
    if not candidates:
        return None
    if prioritize_speed:
        # Fastest: highest RTFx, then lowest size
        candidates = sorted(candidates, key=lambda m: (-m["RTFx"], m["size_gb"]))
    else:
        # Most accurate: lowest WER
        candidates = sorted(candidates, key=lambda m: m["WER"])
    return candidates[0]["model"]

def clear_wav_cache():
    """
    Remove all files and directories under working_memory.
    """
    import shutil
    cache_dir = 'working_memory'
    for entry in os.listdir(cache_dir):
        path = os.path.join(cache_dir, entry)
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)

def main():
    # Clear the working_memory cache
    clear_wav_cache()
    # Start background thread to preload wavs
    source_dir = '/media/jhowe/BACKUPBOY/fake_wavs/'
    dest_dir = 'working_memory/raw_wavs/'
    preload_thread = preload_wavs_threaded(source_dir, dest_dir)
    print("Loading nvidia/parakeet-tdt_ctc-110m ...")
    parakeet_model = transcription_models.load_nvidia_parakeet_tdt_ctc_110m()
    print("Loaded nvidia/parakeet-tdt_ctc-110m.")
    print("Loading nvidia/canary-1b-flash ...")
    canary_model = transcription_models.load_nvidia_canary_1b_flash()
    print("Loaded nvidia/canary-1b-flash.")
    print("Loading openai/whisper-tiny ...")
    whisper_tiny_model, whisper_tiny_processor, whisper_tiny_device = transcription_models.load_openai_whisper_tiny()
    print("Loaded openai/whisper-tiny.")
 
    # After loading models, move up to 1GB of wavs to wavs_to_id in a loop until preload_thread exits
    raw_wavs_dir = 'working_memory/raw_wavs/'
    wavs_to_id_dir = 'working_memory/wavs_to_id/'
    os.makedirs(wavs_to_id_dir, exist_ok=True)
    max_bytes = 1073741824  # 1GB

    def move_up_to_1gb():
        wav_files = [f for f in os.listdir(raw_wavs_dir) if f.endswith('.wav')]
        moved_bytes = 0
        moved_files = 0
        for wav_file in wav_files:
            src = os.path.join(raw_wavs_dir, wav_file)
            dst = os.path.join(wavs_to_id_dir, wav_file)
            file_size = os.path.getsize(src)
            if moved_bytes + file_size > max_bytes:
                break
            shutil.move(src, dst)
            moved_bytes += file_size
            moved_files += 1
        print(f"Moved {moved_files} wav files totaling {moved_bytes / (1024*1024):.2f} MB to {wavs_to_id_dir}")
        return moved_files

    # Loop until preload_thread exits
    while preload_thread.is_alive():
        moved = move_up_to_1gb()
        if moved == 0:
            time.sleep(1)

    # Do one final move for any remaining files
    move_up_to_1gb()

if __name__ == "__main__":
    main()