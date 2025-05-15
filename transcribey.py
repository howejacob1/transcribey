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

def main():
    # Example: Load up to 10 wav files using wav_file_generator
    directory = '/media/jhowe/BACKUPBOY/fake_wavs/'  # Or set to your actual wav directory
    num_files = 10
    gen = wav_file_generator(directory)
    wavs = []
    for _ in range(num_files):
        try:
            wavs.append(next(gen))
        except StopIteration:
            break
    print(f"Loaded {len(wavs)} wav files:")
    for wav in wavs:
        print(wav)

if __name__ == "__main__":
    main()