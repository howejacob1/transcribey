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
import numpy as np
import torch
import torchaudio


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

def get_detected_languages(wav_path, model, processor, device, threshold=0.2):
    import torch
    import torchaudio
    # Load audio
    waveform, sample_rate = torchaudio.load(wav_path)
    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
        sample_rate = target_sample_rate
    # Prepare input features for Whisper
    input_features = processor(waveform.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt").input_features.to(device)

    # Get the language tokens from the tokenizer
    tokenizer = processor.tokenizer
    # All language tokens in Whisper are 6 characters long, e.g., '<|en|>'
    language_tokens = [t for t in tokenizer.additional_special_tokens if len(t) == 6]
    language_token_ids = tokenizer.convert_tokens_to_ids(language_tokens)

    # Run a single decoding step to get logits for language tokens
    # 50258 is the transcribe token for Whisper
    with torch.no_grad():
        logits = model(input_features, decoder_input_ids=torch.tensor([[50258]], device=device)).logits
    # logits shape: (batch, seq_len=1, vocab_size)
    logits = logits[:, 0, :]  # (batch, vocab_size)
    lang_logits = logits[0, language_token_ids]
    lang_probs = torch.softmax(lang_logits, dim=-1).cpu().numpy()

    # Get all languages above threshold
    detected_langs = [language_tokens[i][2:-2] for i, prob in enumerate(lang_probs) if prob >= threshold]
    return detected_langs

def batch_get_detected_languages(wav_paths, model, processor, device, threshold=0.2):
    import torch
    import torchaudio

    waveforms = []
    for wav_path in wav_paths:
        waveform, sample_rate = torchaudio.load(wav_path)
        target_sample_rate = 16000
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)
        waveforms.append(waveform.squeeze().numpy())

    # Batch process
    input_features = processor(waveforms, sampling_rate=16000, return_tensors="pt").input_features.to(device)

    tokenizer = processor.tokenizer
    language_tokens = [t for t in tokenizer.additional_special_tokens if len(t) == 6]
    language_token_ids = tokenizer.convert_tokens_to_ids(language_tokens)

    with torch.no_grad():
        logits = model(input_features, decoder_input_ids=torch.tensor([[50258]] * len(wav_paths), device=device)).logits
    logits = logits[:, 0, :]  # (batch, vocab_size)

    results = []
    for i in range(len(wav_paths)):
        lang_logits = logits[i, language_token_ids]
        lang_probs = torch.softmax(lang_logits, dim=-1).cpu().numpy()
        detected_langs = [language_tokens[j][2:-2] for j, prob in enumerate(lang_probs) if prob >= threshold]
        results.append(detected_langs)
    return results

def main():
    import time
    total_start_time = time.time()
    # Clear the working_memory cache
    clear_wav_cache()
    # Start background thread to preload wavs
    source_dir = '/media/jhowe/BACKUPBOY/fake_wavs/'
    dest_dir = 'working_memory/raw_wavs/'
    preload_thread = preload_wavs_threaded(source_dir, dest_dir, size_limit_bytes=104857600)
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
    wavs_processing_dir = 'working_memory/wavs_processing/'
    os.makedirs(wavs_processing_dir, exist_ok=True)
    max_bytes = 104857600  # 100MB
    
    def start_processing_wavs():
        wav_files = [f for f in os.listdir(raw_wavs_dir) if f.endswith('.wav')]
        moved_bytes = 0
        moved_files = 0
        for wav_file in wav_files:
            src = os.path.join(raw_wavs_dir, wav_file)
            dst = os.path.join(wavs_processing_dir, wav_file)
            file_size = os.path.getsize(src)
            if moved_bytes + file_size > max_bytes:
                break
            shutil.move(src, dst)
            moved_bytes += file_size
            moved_files += 1
        print(f"Moved {moved_files} wav files totaling {moved_bytes / (1024*1024):.2f} MB to {wavs_processing_dir}")
        return moved_files

    vcons = {}
    # Identify languages above threshold for each wav in wavs_to_id_dir
    non_english_dir = 'working_memory/non_english/'
    os.makedirs(non_english_dir, exist_ok=True)
    processed_file_count = 0
    def detect_langs_in_wavs_processing():
        import time
        nonlocal processed_file_count
        wav_files = [f for f in os.listdir(wavs_processing_dir) if f.endswith('.wav')]
        wav_paths = [os.path.join(wavs_processing_dir, f) for f in wav_files]
        if not wav_paths:
            return {}
        total_bytes = sum(os.path.getsize(p) for p in wav_paths)
        start_time = time.time()
        batch_results = batch_get_detected_languages(wav_paths, whisper_tiny_model, whisper_tiny_processor, whisper_tiny_device)
        elapsed = time.time() - start_time
        print(f"Processed {len(wav_files)} files, total size: {total_bytes/(1024*1024):.2f} MB, time taken: {elapsed:.2f} seconds")
        lang_results = {}
        for wav_file, detected_langs in zip(wav_files, batch_results):
            lang_results[wav_file] = detected_langs
            processed_file_count += 1
            print(f"{wav_file}: Detected languages (>=20%): {detected_langs}")
            # If not exclusively English, move to non_english_dir
            if any(lang != 'en' for lang in detected_langs):
                dest_path = os.path.join(non_english_dir, wav_file)
                shutil.move(os.path.join(wavs_processing_dir, wav_file), dest_path)
                print(f"Moved {wav_file} to {non_english_dir} (non-English detected)")
        return lang_results
    lang_results = {}
    transcription_results = {}
    last_time_there_were_files = time.time()
    max_no_new_file_seconds = 5
    MAX_FILES = 10000
    while True:
        start_processing_wavs()
        # Check for new wav files in raw_wavs_dir
        current_raw_wavs = set(os.listdir(raw_wavs_dir))
        if len(current_raw_wavs) != 0:
            print(f"New wav files detected: {current_raw_wavs}")
            last_time_there_were_files = time.time()
        else:
            if time.time() - last_time_there_were_files >= max_no_new_file_seconds:
                print("No new wav files detected for 5 seconds. Exiting main loop.")
                break

        # First
        lang_results.update(detect_langs_in_wavs_processing())
        if processed_file_count >= MAX_FILES:
            print(f"Processed {processed_file_count} files, reached limit of {MAX_FILES}. Exiting main loop.")
            break
        # INSERT_YOUR_CODE
        # Check if non_english_dir exceeds 100MB
        non_english_wavs = [f for f in os.listdir(non_english_dir) if f.endswith('.wav')]
        total_non_english_bytes = sum(os.path.getsize(os.path.join(non_english_dir, f)) for f in non_english_wavs)
        if total_non_english_bytes > 100 * 1024 * 1024 and non_english_wavs:
            print(f"Non-English buffer exceeds 100MB ({total_non_english_bytes/(1024*1024):.2f} MB). Transcribing with canary-1b-flash.")
            # Load canary-1b-flash model if not already loaded
            if 'canary_model' not in globals():
                canary_model = transcription_models.load_nvidia_canary_1b_flash()
            for wav_file in non_english_wavs:
                wav_path = os.path.join(non_english_dir, wav_file)
                try:
                    print(f"Transcribing {wav_file} with canary-1b-flash")
                    transcription = canary_model.transcribe([wav_path])[0]
                    transcription_results[wav_file] = transcription
                    print(f"Done Transcribed {wav_file} with canary-1b-flash")
                    os.remove(wav_path)
                except Exception as e:
                    print(f"Error transcribing {wav_file} with canary-1b-flash: {str(e)}")
        # Transcribe English files with Parakeet model
        for wav_file in os.listdir(wavs_processing_dir):
            if not wav_file.endswith('.wav'):
                continue
            wav_path = os.path.join(wavs_processing_dir, wav_file)
            try:
                # Transcribe with Parakeet
                print(f"Transcribing {wav_file} with Parakeet")
                transcription = parakeet_model.transcribe([wav_path])[0]
                transcription_results[wav_file] = transcription
                print(f"Done Transcribed {wav_file} with Parakeet")
                # Remove file after successful transcription
                os.remove(wav_path)
            except Exception as e:
                print(f"Error transcribing {wav_file}: {str(e)}")

        print(f"Number of files processed: {len(lang_results)}")
        time.sleep(3)

    total_elapsed = time.time() - total_start_time
    print(f"\nAll language results:")
    print(lang_results)
    print(f"\nTotal files processed: {processed_file_count}")
    print(f"Total script runtime: {total_elapsed:.2f} seconds")

if __name__ == "__main__":
    main()