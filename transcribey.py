# Architecture: 

# Load all filesystem filenames into memory

# Make a vcon for each file

# In background, always load up wav files into buffer

# Detect language on each, save to vcon

# If english, transcribe with fastest english model on each

# Else, add the wav to the list of files to be transcribed. 

# Write the vcon and remove the wav if successful.

# Now, go through all non-english ones and use fastest model for each language. 

from utils import get_all_filenames
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