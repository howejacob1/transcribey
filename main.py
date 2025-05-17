import logging
# Define a custom TRACE level (5) that's more detailed than DEBUG (10)
logging.basicConfig(level=logging.INFO)
import time
import os
import transcription_models
from utils import get_all_filenames, wav_file_generator, filter_wav_files
from wav_cache import preload_wavs_threaded
import shutil
import numpy as np
import torch
import torchaudio
import settings
from mongo_utils import get_mongo_collection
import threading

# Add vCon imports
import datetime
from vcon import Vcon
from vcon.party import Party
from vcon.dialog import Dialog

def clear_wav_cache():
    """
    Remove all files and directories under working_memory.
    """
    logging.info("Starting to clear working_memory cache...")
    start_time = time.time()
    cache_dir = 'working_memory'
    for entry in os.listdir(cache_dir):
        path = os.path.join(cache_dir, entry)
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
    elapsed = time.time() - start_time
    logging.info(f"Finished clearing working_memory cache in {elapsed:.2f} seconds.")

def load_and_resample_waveforms(wav_paths, target_sample_rate=16000):
    """
    Loads and resamples a list of wav files to the target sample rate.
    Returns a list of numpy arrays (waveforms).
    """
    waveforms = []
    for wav_path in wav_paths:
        waveform, sample_rate = torchaudio.load(wav_path)
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)
        waveforms.append(waveform.squeeze().numpy())
    return waveforms

def batch_get_detected_languages(wav_paths, model, processor, device, threshold=0.2):
    waveforms = load_and_resample_waveforms(wav_paths, target_sample_rate=16000)

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

def create_vcon_for_wav(rel_path, abs_path):
    """
    Create a vCon for the given wav file and return it as a dict.
    """
    logging.getLogger().setLevel(logging.WARN)
    vcon = Vcon.build_new()
    party = Party(name="Unknown", role="participant")
    vcon.add_party(party)
    now = datetime.datetime.now(datetime.timezone.utc)
    dialog = Dialog(
        type="audio",
        start=now.isoformat(),
        parties=[0],
        originator=0,
        mimetype="audio/wav",
        filename=rel_path,
        body=None,
        encoding=None
    )
    vcon.add_dialog(dialog)
    vcon.add_attachment(type="audio", body=rel_path, encoding="none")
    logging.getLogger().setLevel(logging.INFO)
    return vcon.to_dict()

def get_existing_vcon_filenames(collection):
    """
    Find all filenames that already have vCons in the MongoDB collection.
    Returns a set of filenames.
    """
    existing_filenames = set()
    t_exist_start = time.time()
    
    # Check dialog.filename which may contain filenames
    dialog_count = 0
    for doc in collection.find({"dialog.filename": {"$exists": True}}, {"dialog.filename": 1}):
        dialogs = doc.get("dialog", [])
        for dlg in dialogs:
            fname = dlg.get("filename")
            if fname:
                existing_filenames.add(fname)
        dialog_count += 1
        if dialog_count % 10000 == 0:
            logging.info(f"Scanned {dialog_count} MongoDB documents for existing vCons (dialogs)...")
    
    t_exist_end = time.time()
    logging.info(f"Scanned {dialog_count} MongoDB documents for existing vCons in {t_exist_end - t_exist_start:.2f} seconds.")
    logging.info(f"Found {len(existing_filenames)} unique filenames in the database.")
    
    return existing_filenames

def build_vcon_dicts(wavs, existing_filenames):
    """
    Build vCon dictionaries for wav files that don't already have vCons.
    
    Args:
        wavs: Dictionary mapping relative paths to absolute paths of wav files
        existing_filenames: Set of filenames that already have vCons
        
    Returns:
        List of vCon dictionaries and count of skipped files
    """
    vcon_dicts = []
    skipped = 0
    loop_start_time = time.time()
    last_print_time = loop_start_time
    processed = 0
    for rel_path, abs_path in wavs.items():
        file_start_time = time.time()
        if rel_path in existing_filenames:
            skipped += 1
            continue
        vcon_dicts.append(create_vcon_for_wav(rel_path, abs_path))
        processed += 1
        file_elapsed = time.time() - file_start_time
        now = time.time()
        # Print every 10000 files or every 5 seconds
        if processed % 10000 == 0 or (now - last_print_time) > 5:
            logging.info(f"Processed {processed} new vCons so far (skipped {skipped})")
            last_print_time = now
    loop_elapsed = time.time() - loop_start_time
    logging.info(f"Finished building {len(vcon_dicts)} new vCons in {loop_elapsed:.2f} seconds. Skipped {skipped} files.")
    
    return vcon_dicts, skipped

def insert_vcon_dicts_to_mongo(collection, vcon_dicts, skipped):
    """
    Insert vCon dictionaries into MongoDB in batches.
    
    Args:
        collection: MongoDB collection to insert into
        vcon_dicts: List of vCon dictionaries to insert
        skipped: Count of skipped files for logging
    """
    t3 = time.time()
    if vcon_dicts:
        batch_size = 10000
        total = len(vcon_dicts)
        for i in range(0, total, batch_size):
            batch = vcon_dicts[i:i+batch_size]
            collection.insert_many(batch)
            logging.info(f"Inserted batch {i//batch_size + 1} ({min(i+batch_size, total)}/{total}) vCons into MongoDB.")
        t4 = time.time()
        logging.info(f"Inserted {len(vcon_dicts)} vCons into MongoDB in {t4 - t3:.2f} seconds.")
    else:
        logging.info("No new vCons to insert.")
    if skipped:
        logging.info(f"Skipped {skipped} files that already had vCons.")

def maybe_add_vcons_to_mongo(target_dir):
    """
    For each .wav file in target_dir (recursively), create a vCon referencing it and insert into MongoDB in bulk,
    unless a vCon for that wav already exists.
    """
    collection = get_mongo_collection()
    
    # Ensure we have an index on dialog filenames for faster lookups
    collection.create_index([("dialog.filename", 1)])
    
    logging.info(f"Getting all wav files in {target_dir}")
    start_time = time.time()
    file_dict = get_all_filenames(target_dir)
    logging.info(f"Got all filenames in {time.time() - start_time:.2f} seconds.")
    
    wavs = filter_wav_files(file_dict)
    logging.info(f"Found {len(wavs)} wav files in {target_dir}")

    # Find which files already have vCons
    existing_filenames = get_existing_vcon_filenames(collection)

    vcon_dicts, skipped = build_vcon_dicts(wavs, existing_filenames)
    
    insert_vcon_dicts_to_mongo(collection, vcon_dicts, skipped)

def main():
    total_start_time = time.time()
    
    # Run maybe_add_vcons_to_mongo in a separate thread
    vcon_thread = threading.Thread(target=maybe_add_vcons_to_mongo, args=(settings.source_dir,))
    vcon_thread.start()
    
    # Load models while the vcon thread is running
    logging.info("Loading transcription models...")
    parakeet_model = transcription_models.load_nvidia_parakeet_tdt_ctc_110m()
    canary_model = transcription_models.load_nvidia_canary_1b_flash()
    whisper_tiny_model, whisper_tiny_processor, whisper_tiny_device = transcription_models.load_openai_whisper_tiny()
    logging.info("Finished loading transcription models")
    
    # Wait for the vcon thread to complete
    logging.info("Waiting for vCon thread to complete...")
    vcon_thread.join()
    logging.info("vCon thread completed")

    logging.info(f"Total runtime: {time.time() - total_start_time:.2f}")

def old_main():
    total_start_time = time.time()
    # Clear the working_memory cache
    clear_wav_cache()
    # Start background thread to preload wavs
    source_dir = settings.source_dir
    dest_dir = settings.dest_dir
    preload_thread = preload_wavs_threaded(source_dir, dest_dir, size_limit_bytes=settings.preload_size_limit_bytes)
    # Remove print statements for model loading
    parakeet_model = transcription_models.load_nvidia_parakeet_tdt_ctc_110m()
    canary_model = transcription_models.load_nvidia_canary_1b_flash()
    whisper_tiny_model, whisper_tiny_processor, whisper_tiny_device = transcription_models.load_openai_whisper_tiny()
 
    # After loading models, move up to 1GB of wavs to wavs_to_id in a loop until preload_thread exits
    raw_wavs_dir = settings.dest_dir
    wavs_processing_dir = settings.wavs_in_progress_dir
    os.makedirs(wavs_processing_dir, exist_ok=True)
    max_bytes = settings.max_bytes
    
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
        logging.info(f"Moved {moved_files} wav files totaling {moved_bytes / (1024*1024):.2f} MB to {wavs_processing_dir}")
        return moved_files

    vcons = {}
    # Identify languages above threshold for each wav in wavs_to_id_dir
    non_english_dir = settings.non_en_wavs_in_progress_dir
    os.makedirs(non_english_dir, exist_ok=True)
    processed_file_count = 0
    def detect_langs_in_wavs_processing():
        nonlocal processed_file_count
        wav_files = [f for f in os.listdir(wavs_processing_dir) if f.endswith('.wav')]
        wav_paths = [os.path.join(wavs_processing_dir, f) for f in wav_files]
        if not wav_paths:
            return {}
        total_bytes = sum(os.path.getsize(p) for p in wav_paths)
        start_time = time.time()
        batch_results = batch_get_detected_languages(wav_paths, whisper_tiny_model, whisper_tiny_processor, whisper_tiny_device, threshold=settings.lang_detect_threshold)
        elapsed = time.time() - start_time
        logging.info(f"Processed {len(wav_files)} files, total size: {total_bytes/(1024*1024):.2f} MB, time taken: {elapsed:.2f} seconds")
        lang_results = {}
        for wav_file, detected_langs in zip(wav_files, batch_results):
            lang_results[wav_file] = detected_langs
            processed_file_count += 1
            logging.info(f"{wav_file}: Detected languages (>=20%): {detected_langs}")
            # If not exclusively English, move to non_english_dir
            if any(lang != 'en' for lang in detected_langs):
                dest_path = os.path.join(non_english_dir, wav_file)
                shutil.move(os.path.join(wavs_processing_dir, wav_file), dest_path)
                logging.info(f"Moved {wav_file} to {non_english_dir} (non-English detected)")
        return lang_results
    lang_results = {}
    transcription_results = {}
    last_time_there_were_files = time.time()
    max_no_new_file_seconds = settings.max_no_new_file_seconds
    MAX_FILES = settings.max_files
    while True:
        start_processing_wavs()
        # Check for new wav files in raw_wavs_dir
        current_raw_wavs = set(os.listdir(raw_wavs_dir))
        if len(current_raw_wavs) != 0:
            logging.info(f"New wav files detected: {current_raw_wavs}")
            last_time_there_were_files = time.time()
        else:
            if time.time() - last_time_there_were_files >= max_no_new_file_seconds:
                logging.info("No new wav files detected for 5 seconds. Exiting main loop.")
                break

        # First
        lang_results.update(detect_langs_in_wavs_processing())
        if processed_file_count >= MAX_FILES:
            logging.info(f"Processed {processed_file_count} files, reached limit of {MAX_FILES}. Exiting main loop.")
            break
        # INSERT_YOUR_CODE
        # Check if non_english_dir exceeds 100MB
        non_english_wavs = [f for f in os.listdir(non_english_dir) if f.endswith('.wav')]
        total_non_english_bytes = sum(os.path.getsize(os.path.join(non_english_dir, f)) for f in non_english_wavs)
        if total_non_english_bytes > 100 * 1024 * 1024 and non_english_wavs:
            logging.info(f"Non-English buffer exceeds 100MB ({total_non_english_bytes/(1024*1024):.2f} MB). Transcribing with canary-1b-flash.")
            # Load canary-1b-flash model if not already loaded
            if 'canary_model' not in globals():
                canary_model = transcription_models.load_nvidia_canary_1b_flash()
            for wav_file in non_english_wavs:
                wav_path = os.path.join(non_english_dir, wav_file)
                try:
                    logging.info(f"Transcribing {wav_file} with canary-1b-flash")
                    transcription = canary_model.transcribe([wav_path])[0]
                    transcription_results[wav_file] = transcription
                    logging.info(f"Done Transcribed {wav_file} with canary-1b-flash")
                    os.remove(wav_path)
                except Exception as e:
                    logging.error(f"Error transcribing {wav_file} with canary-1b-flash: {str(e)}")
        # Transcribe English files with Parakeet model
        for wav_file in os.listdir(wavs_processing_dir):
            if not wav_file.endswith('.wav'):
                continue
            wav_path = os.path.join(wavs_processing_dir, wav_file)
            try:
                # Transcribe with Parakeet
                logging.info(f"Transcribing {wav_file} with Parakeet")
                transcription = parakeet_model.transcribe([wav_path])[0]
                transcription_results[wav_file] = transcription
                logging.info(f"Done Transcribed {wav_file} with Parakeet")
                # Remove file after successful transcription
                os.remove(wav_path)
            except Exception as e:
                logging.error(f"Error transcribing {wav_file}: {str(e)}")

        logging.info(f"Number of files processed: {len(lang_results)}")
        time.sleep(3)

    total_elapsed = time.time() - total_start_time
    logging.info(f"\nAll language results:")
    logging.info(lang_results)
    logging.info(f"\nTotal files processed: {processed_file_count}")
    logging.info(f"Total script runtime: {total_elapsed:.2f} seconds")

if __name__ == "__main__":
    main()