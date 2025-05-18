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
    Returns a list of numpy arrays (waveforms) and a list of valid indices.
    """
    waveforms = []
    valid_indices = []
    for i, wav_path in enumerate(wav_paths):
        try:
            waveform, sample_rate = torchaudio.load(wav_path)
            if sample_rate != target_sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
                waveform = resampler(waveform)
            waveforms.append(waveform.squeeze().numpy())
            valid_indices.append(i)
        except Exception as e:
            logging.error(f"Error loading file {wav_path}: {str(e)}")
    return waveforms, valid_indices

def batch_get_detected_languages(wav_paths, model, processor, device, threshold=0.2):
    waveforms, valid_indices = load_and_resample_waveforms(wav_paths, target_sample_rate=16000)
    
    if not waveforms:
        return [[] for _ in wav_paths]  # Return empty language list for all inputs
    
    # Batch process
    input_features = processor(waveforms, sampling_rate=16000, return_tensors="pt").input_features.to(device)

    tokenizer = processor.tokenizer
    language_tokens = [t for t in tokenizer.additional_special_tokens if len(t) == 6]
    language_token_ids = tokenizer.convert_tokens_to_ids(language_tokens)

    with torch.no_grad():
        logits = model(input_features, decoder_input_ids=torch.tensor([[50258]] * len(waveforms), device=device)).logits
    logits = logits[:, 0, :]  # (batch, vocab_size)

    # Initialize results with empty lists
    results = [[] for _ in wav_paths]
    
    # Fill in results for valid files
    for i, orig_idx in enumerate(valid_indices):
        lang_logits = logits[i, language_token_ids]
        lang_probs = torch.softmax(lang_logits, dim=-1).cpu().numpy()
        detected_langs = [language_tokens[j][2:-2] for j, prob in enumerate(lang_probs) if prob >= threshold]
        results[orig_idx] = detected_langs
        
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

def process_language_identification(collection, model, processor, device, batch_size=100, max_vcons=1000, threshold=0.2):
    """
    Identify languages in wav files referenced in vCons using the Whisper model.
    
    Args:
        collection: MongoDB collection containing vCons
        model: Whisper model for language identification
        processor: Whisper processor
        device: Device to run model on (cuda/cpu)
        batch_size: Number of files to process in each batch
        max_vcons: Maximum number of vCons to process
        threshold: Confidence threshold for language detection
        
    Returns:
        Tuple of (english_vcons, non_english_vcons) lists with vCon IDs and their file paths
    """
    logging.info(f"Starting language identification for up to {max_vcons} vCons")
    start_time = time.time()
    
    # Find vCons with audio attachments that don't have language identification yet
    query = {
        "attachments": {"$elemMatch": {"type": "audio"}},
        "analysis": {"$not": {"$elemMatch": {"type": "language_identification"}}}
    }
    
    vcons_to_process = list(collection.find(query).limit(max_vcons))
    total_vcons = len(vcons_to_process)
    logging.info(f"Found {total_vcons} vCons to process for language identification")
    
    # Initialize result containers
    english_vcons = []
    non_english_vcons = []
    processed = 0
    
    # Process in batches
    for i in range(0, total_vcons, batch_size):
        batch = vcons_to_process[i:min(i+batch_size, total_vcons)]
        batch_files = []
        batch_ids = []
        file_id_map = {}  # Map file index to (vcon_id, file_path)
        
        # Collect file paths and IDs
        for idx, vcon in enumerate(batch):
            for attachment in vcon.get('attachments', []):
                if attachment.get('type') == 'audio':
                    rel_path = attachment.get('body')
                    abs_path = os.path.join(settings.source_dir, rel_path)
                    if os.path.exists(abs_path) and os.path.isfile(abs_path):
                        batch_files.append(abs_path)
                        batch_ids.append(vcon['_id'])
                        file_id_map[len(batch_files)-1] = (vcon['_id'], abs_path)
                        break
        
        if not batch_files:
            continue
            
        batch_start = time.time()
        
        try:
            # Detect languages
            detected_languages = batch_get_detected_languages(
                batch_files, 
                model, 
                processor, 
                device, 
                threshold=threshold
            )
            
            batch_elapsed = time.time() - batch_start
            logging.info(f"Processed batch of {len(batch_files)} files in {batch_elapsed:.2f} seconds")
            
            # Update vCons with language info and segregate English vs non-English
            for idx, (file_path, langs) in enumerate(zip(batch_files, detected_languages)):
                vcon_id = batch_ids[idx]
                
                # Add language identification analysis to vCon
                analysis = {
                    "type": "language_identification",
                    "dialog": [0],  # Assuming the first dialog contains the audio
                    "vendor": "whisper-tiny",
                    "body": langs,
                    "encoding": "none"
                }
                
                try:
                    collection.update_one(
                        {"_id": vcon_id},
                        {"$push": {"analysis": analysis}}
                    )
                    
                    logging.info(f"vCon {vcon_id}: File {os.path.basename(file_path)}, detected languages: {langs}")
                    
                    # Categorize as English or non-English
                    if langs and all(lang == 'en' for lang in langs):
                        english_vcons.append((vcon_id, file_path))
                    elif langs:  # Has some non-English language
                        non_english_vcons.append((vcon_id, file_path))
                        
                    processed += 1
                    if processed % 100 == 0:
                        logging.info(f"Processed {processed}/{total_vcons} vCons for language identification")
                        
                except Exception as e:
                    logging.error(f"Error updating vCon {vcon_id} in MongoDB: {str(e)}")
                    
        except Exception as e:
            logging.error(f"Error processing batch for language identification: {str(e)}")
    
    total_elapsed = time.time() - start_time
    logging.info(f"Completed language identification for {processed} vCons in {total_elapsed:.2f} seconds")
    logging.info(f"Found {len(english_vcons)} English vCons and {len(non_english_vcons)} non-English vCons")
    
    return english_vcons, non_english_vcons

def transcribe_vcons(collection, model_name, model, vcons_to_transcribe, batch_size=20):
    """
    Transcribe a list of vCons using the specified model.
    
    Args:
        collection: MongoDB collection containing vCons
        model_name: Name of the model being used (for logging)
        model: The transcription model to use
        vcons_to_transcribe: List of (vcon_id, file_path) tuples to transcribe
        batch_size: Number of files to transcribe in each batch
    """
    if not vcons_to_transcribe:
        logging.info(f"No vCons to transcribe with {model_name}")
        return
        
    total = len(vcons_to_transcribe)
    logging.info(f"Starting transcription of {total} vCons with {model_name}")
    start_time = time.time()
    processed = 0
    
    # Process in batches
    for i in range(0, total, batch_size):
        batch = vcons_to_transcribe[i:min(i+batch_size, total)]
        batch_ids = [vcon_id for vcon_id, _ in batch]
        batch_files = [file_path for _, file_path in batch]
        
        # Check if all files exist
        valid_indices = []
        valid_files = []
        valid_ids = []
        for idx, file_path in enumerate(batch_files):
            if os.path.exists(file_path) and os.path.isfile(file_path):
                valid_indices.append(idx)
                valid_files.append(file_path)
                valid_ids.append(batch_ids[idx])
            else:
                logging.warning(f"File not found or not accessible: {file_path}")
        
        if not valid_files:
            logging.warning(f"No valid files in batch {i//batch_size + 1}")
            continue
        
        batch_start = time.time()
        try:
            # Transcribe all files in the batch
            transcriptions = model.transcribe(valid_files)
            batch_elapsed = time.time() - batch_start
            logging.info(f"Transcribed batch of {len(valid_files)} files with {model_name} in {batch_elapsed:.2f} seconds")
            
            # Update each vCon with its transcription
            for vcon_id, file_path, transcription in zip(valid_ids, valid_files, transcriptions):
                # Add transcription analysis to vCon
                analysis = {
                    "type": "transcription",
                    "dialog": [0],  # Assuming the first dialog contains the audio
                    "vendor": model_name,
                    "body": transcription.text if hasattr(transcription, 'text') else str(transcription),
                    "encoding": "none"
                }
                
                try:
                    collection.update_one(
                        {"_id": vcon_id},
                        {"$push": {"analysis": analysis}}
                    )
                    
                    logging.info(f"Transcribed {os.path.basename(file_path)} with {model_name}: {analysis['body']}")
                    processed += 1
                except Exception as e:
                    logging.error(f"Error updating transcription for vCon {vcon_id}: {str(e)}")
                
        except Exception as e:
            logging.error(f"Error transcribing batch with {model_name}: {str(e)}")
            
        if processed % 100 == 0 or processed == total:
            logging.info(f"Progress: {processed}/{total} vCons transcribed with {model_name}")
    
    total_elapsed = time.time() - start_time
    logging.info(f"Completed transcription of {processed} vCons with {model_name} in {total_elapsed:.2f} seconds")

def find_vcons_pending_transcription(collection, max_vcons=1000):
    """
    Find vCons that have language identification but don't have transcription yet.
    
    Args:
        collection: MongoDB collection containing vCons
        max_vcons: Maximum number of vCons to process
        
    Returns:
        Tuple of (english_vcons, non_english_vcons) lists with vCon IDs and their file paths
    """
    logging.info(f"Looking for vCons with language identification but no transcription...")
    start_time = time.time()
    
    # Find vCons with language_identification but no transcription
    pipeline = [
        {
            "$match": {
                "analysis": {
                    "$elemMatch": {"type": "language_identification"}
                }
            }
        },
        {
            "$match": {
                "analysis.type": {"$ne": "transcription"}
            }
        },
        {"$limit": max_vcons}
    ]
    
    vcons_to_process = list(collection.aggregate(pipeline))
    total_vcons = len(vcons_to_process)
    logging.info(f"Found {total_vcons} vCons with language identification but no transcription")
    
    # Initialize result containers
    english_vcons = []
    non_english_vcons = []
    
    # Process vCons
    for vcon in vcons_to_process:
        # Find the language identification analysis
        lang_analysis = next((a for a in vcon.get('analysis', []) if a.get('type') == 'language_identification'), None)
        if not lang_analysis:
            continue
        
        # Get file path
        file_path = None
        for attachment in vcon.get('attachments', []):
            if attachment.get('type') == 'audio':
                rel_path = attachment.get('body')
                abs_path = os.path.join(settings.source_dir, rel_path)
                if os.path.exists(abs_path) and os.path.isfile(abs_path):
                    file_path = abs_path
                    break
        
        if not file_path:
            continue
        
        # Categorize based on detected languages
        langs = lang_analysis.get('body', [])
        if langs and all(lang == 'en' for lang in langs):
            english_vcons.append((vcon['_id'], file_path))
        elif langs:  # Has some non-English language
            non_english_vcons.append((vcon['_id'], file_path))
    
    total_elapsed = time.time() - start_time
    logging.info(f"Processed {len(english_vcons) + len(non_english_vcons)} vCons in {total_elapsed:.2f} seconds")
    logging.info(f"Found {len(english_vcons)} English vCons and {len(non_english_vcons)} non-English vCons for transcription")
    
    return english_vcons, non_english_vcons

def main():
    total_start_time = time.time()

    # Create working_memory directory if it doesn't exist
    os.makedirs("working_memory", exist_ok=True)
    
    try:
        # Run maybe_add_vcons_to_mongo in a separate thread
        vcon_thread = threading.Thread(target=maybe_add_vcons_to_mongo, args=(settings.source_dir,))
        vcon_thread.start()
        
        # Load models while the vcon thread is running
        parakeet_model = transcription_models.load_nvidia_parakeet_tdt_ctc_110m()
        canary_model = transcription_models.load_nvidia_canary_1b_flash()
        whisper_tiny_model, whisper_tiny_processor, whisper_tiny_device = transcription_models.load_openai_whisper_tiny()
        
        # Wait for the vcon thread to complete
        logging.info("Waiting for vCon loading thread to complete...")
        vcon_thread.join()
        logging.info("vCon loading thread completed")

        # Get MongoDB connection
        collection = get_mongo_collection()
        work_start_time = time.time()
        
        # First, process language identification for vCons that don't have it yet
        english_vcons, non_english_vcons = process_language_identification(
            collection, 
            whisper_tiny_model, 
            whisper_tiny_processor, 
            whisper_tiny_device,
            batch_size=50,  # Smaller batch size for more reliability
            max_vcons=1000,
            threshold=settings.lang_detect_threshold
        )
        
        # If no vCons were found for language identification, check for ones that need transcription
        if len(english_vcons) == 0 and len(non_english_vcons) == 0:
            logging.info("No new vCons for language identification. Looking for vCons pending transcription...")
            english_vcons, non_english_vcons = find_vcons_pending_transcription(collection, max_vcons=1000)
        
        # Process English vCons with Parakeet (up to 1000)
        logging.info(f"Processing {len(english_vcons)} English vCons with Parakeet model")
        english_vcons_to_process = english_vcons[:1000]
        if english_vcons_to_process:
            transcribe_vcons(
                collection,
                "nvidia/parakeet-tdt_ctc-110m",
                parakeet_model,
                english_vcons_to_process,
                batch_size=10  # Smaller batch size for more reliability
            )
        
        # Process non-English vCons with Canary (up to 1000)
        logging.info(f"Processing {len(non_english_vcons)} non-English vCons with Canary model")
        non_english_vcons_to_process = non_english_vcons[:1000]
        if non_english_vcons_to_process:
            transcribe_vcons(
                collection,
                "nvidia/canary-1b-flash",
                canary_model,
                non_english_vcons_to_process,
                batch_size=10  # Smaller batch size for more reliability
            )
        logging.info(f"Total time to process vCons: {time.time() - work_start_time:.2f} seconds")

    except Exception as e:
        logging.error(f"Error in main function: {str(e)}")
        import traceback
        traceback.print_exc()

    total_elapsed = time.time() - total_start_time
    logging.info(f"Total runtime: {total_elapsed:.2f} seconds")

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