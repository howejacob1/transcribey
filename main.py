import traceback
import logging
import sys
import time
import os
from transcription_models import load_model_by_name
from utils import get_valid_wav_files, get_total_wav_size, get_total_wav_duration
import shutil
import numpy as np
import torch
import torchaudio
import settings
from mongo_utils import get_mongo_collection, delete_all_vcons, delete_all_faqs, print_all_vcons
import threading
import datetime
from vcon import Vcon
from vcon.party import Party
from vcon.dialog import Dialog


#architecture
#  transcribies are running.

# Each transcribey has a unique name. 
# Each hard drive partition has a unique UUID

# When unloaded, slaves first check if there are new hard drives.
# If there are, mount them if unmounted, and check if there are any new wavs. 
# then, if there are new wavs, it will report these files to the database,
# including UUID and slave name. 
# Slaves additionally report their IPs regularly. 
# Slaves additionally scan existing folders for new wavs.
# Slaves only need to wake up and report every once in a while. 

# When the slave detects new wavs, it reports these files to the master.
# When the slave has spare GPU power, it goes to the database and sees 
# which ones have undetermined languages or lack of transcriptions locally, 
# then over the network, prefering models already loaded.
# The slave reserves these files. 
# Then, the slave works on these files. 
# Slaves attempt not to load another model as much as possible. 
# Slaves report results to the master.
# Slaves tell the database that they are still alive.
# If a slave detects that the database has an entry of a slave that isn't alive, it will remove all locks in all database entries that have that entry. 
# All slaves have some static IP. 

import transcription_models

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
    wavs = get_valid_wav_files(target_dir)
    # Filter out unreadable wavs
    logging.info(f"Found {len(wavs)} readable wav files in {target_dir}")

    # Find which files already have vCons
    existing_filenames = get_existing_vcon_filenames(collection)

    vcon_dicts, skipped = build_vcon_dicts(wavs, existing_filenames)
    
    insert_vcon_dicts_to_mongo(collection, vcon_dicts, skipped)

def identify_languages(collection, wav_paths, max_vcons=10000):
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
    whisper_tiny_model, whisper_tiny_processor, whisper_tiny_device = transcription_models.load_openai_whisper_tiny()
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
    total_bytes = get_total_wav_size(list(wav_paths.values()))
    total_audio_seconds = get_total_wav_duration(list(wav_paths.values()))
    all_files = []
    for vcon in vcons_to_process:
        for attachment in vcon.get('attachments', []):
            if attachment.get('type') == 'audio':
                rel_path = attachment.get('body')
                abs_path = os.path.join(settings.source_dir, rel_path)
                if os.path.exists(abs_path) and os.path.isfile(abs_path):
                    all_files.append(abs_path)
    # Process in batches
    for i in range(0, len(vcons_to_process), settings.lang_detect_batch_size):
        batch = vcons_to_process[i:min(i+settings.lang_detect_batch_size, len(vcons_to_process))]
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
                whisper_tiny_model, 
                whisper_tiny_processor, 
                whisper_tiny_device, 
                threshold=settings.lang_detect_threshold
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
    logging.info(f"Total data: {total_bytes / (1024**3):.2f} GB, total length: {total_audio_seconds:.2f} seconds, real time factor: {total_audio_seconds / total_elapsed:.1f}x")
    logging.info(f"Found {len(english_vcons)} English vCons and {len(non_english_vcons)} non-English vCons")
    
    # Unload the model to free up GPU memory
    del whisper_tiny_model
    del whisper_tiny_processor
    del whisper_tiny_device
    torch.cuda.empty_cache()
    
    return english_vcons, non_english_vcons, {
        'duration': total_audio_seconds,
        'size': total_bytes,
        'batch_size': settings.lang_detect_batch_size
    }


def transcribe_vcons(collection, model, vcons_to_transcribe, batch_size):
    """
    Transcribe a list of vCons using the specified model.
    
    Args:
        collection: MongoDB collection containing vCons
        model: Loaded model instance (should have a .transcribe method)
        vcons_to_transcribe: List of (vcon_id, file_path) tuples to transcribe
        batch_size: Number of files to process in each batch
    """

    if not vcons_to_transcribe:
        logging.info(f"No vCons to transcribe with {getattr(model, 'name', getattr(model, '__class__', type(model)).__name__)}")
        return
        
    total = len(vcons_to_transcribe)
    model_name = getattr(model, 'name', getattr(model, '__class__', type(model)).__name__)
    logging.info(f"Starting transcription of {total} vCons with {model_name}")
    start_time = time.time()
    processed = 0
    file_paths_only = [file_path for _, file_path in vcons_to_transcribe]
    total_bytes = get_total_wav_size(file_paths_only)
    total_audio_seconds = get_total_wav_duration(file_paths_only)
    all_files = file_paths_only
    for i in range(0, total, batch_size):
        batch = vcons_to_transcribe[i:min(i+batch_size, total)]
        batch_ids = [vcon_id for vcon_id, _ in batch]
        batch_files = [file_path for _, file_path in batch]
        valid_indices = []
        valid_files = []
        valid_ids = []
        for idx, file_path in enumerate(batch_files):
            if os.path.exists(file_path) and os.path.isfile(file_path):
                valid_indices.append(idx)
                valid_files.append(file_path)
                valid_ids.append(batch_ids[idx])
            else:
                logging.warning(f"File not found, not accessible, or invalid: {file_path}")
        if not valid_files:
            logging.warning(f"No valid files in batch {i//batch_size + 1}")
            continue
        batch_start = time.time()
        try:
            print(f"Files to transcribe: {valid_files}")
            transcriptions = model.transcribe(valid_files)
            batch_elapsed = time.time() - batch_start
            logging.info(f"Transcribed batch of {len(valid_files)} files with {model_name} in {batch_elapsed:.2f} seconds")
            for vcon_id, file_path, transcription in zip(valid_ids, valid_files, transcriptions):
                analysis = {
                    "type": "transcription",
                    "dialog": [0],
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
        except RuntimeError as e:
            if 'CUDA error' in str(e):
                logging.error(f"CUDA error during transcription: {str(e)}. Clearing CUDA cache and continuing.")
                torch.cuda.empty_cache()
            else:
                logging.error(f"Error transcribing batch with {model_name}: {str(e)}")
        except Exception as e:
            logging.error(f"Error transcribing batch with {model_name}: {str(e)}")
        if processed % 100 == 0 or processed == total:
            logging.info(f"Progress: {processed}/{total} vCons transcribed with {model_name}")
    total_elapsed = time.time() - start_time
    rtf = None
    if total_elapsed > 0 and total_audio_seconds > 0:
        rtf = total_audio_seconds / total_elapsed
    logging.info(f"Completed transcription of {processed} vCons with {model_name} in {total_elapsed:.2f} seconds")
    logging.info(f"Total data: {total_bytes / (1024**3):.2f} GB, total length: {total_audio_seconds:.2f} seconds, real time factor: {rtf:.1f}x")
    del model
    torch.cuda.empty_cache()
    return {
        'rtf': rtf,
        'duration': total_audio_seconds,
        'size': total_bytes,
        'batch_size': batch_size
    }

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
    
    # Check for debug=true in command-line arguments
    if any(arg.lower() == 'debug=true' for arg in sys.argv):
        print("Debug mode: clearing MongoDB collections...")
        delete_all_vcons()
        delete_all_faqs()

    total_start_time = time.time()
    logging.basicConfig(level=logging.INFO)

    try:

        maybe_add_vcons_to_mongo(settings.source_dir)
        # Get MongoDB connection
        collection = get_mongo_collection()
        work_start_time = time.time()

        # First, process language identification for vCons that don't have it yet
        english_vcons, non_english_vcons, lang_stats = identify_languages(collection, get_valid_wav_files(settings.source_dir), max_vcons=10000)
            
        # If no vCons were found for language identification, check for ones that need transcription
        if len(english_vcons) == 0 and len(non_english_vcons) == 0:
            logging.info("No new vCons for language identification. Looking for vCons pending transcription...")
            english_vcons, non_english_vcons = find_vcons_pending_transcription(collection, max_vcons=10000)
        parakeet_model = transcription_models.load_model_by_name("nvidia/parakeet-tdt_ctc-110m")
        # Process English vCons with Parakeet (up to 1000)
        logging.info(f"Processing {len(english_vcons)} English vCons with Canary model")
        english_vcons_to_process = english_vcons
        if english_vcons_to_process:
            en_stats = transcribe_vcons(
                collection,
                parakeet_model,
                english_vcons_to_process,
                settings.en_transcription_batch_size)
        del parakeet_model
        torch.cuda.empty_cache()
        # Process non-English vCons with Canary (up to 1000)
        logging.info(f"Processing {len(non_english_vcons)} non-English vCons with Canary model")
        canary_model = transcription_models.load_model_by_name("nvidia/canary-1b-flash")        
        non_english_vcons_to_process = non_english_vcons
        if non_english_vcons_to_process:
            non_en_stats = transcribe_vcons(
                collection,
                canary_model,
                non_english_vcons_to_process,
                settings.non_en_transcription_batch_size
            )
        del canary_model
        torch.cuda.empty_cache()
        logging.info(f"Total time to process vCons: {time.time() - work_start_time:.2f} seconds")

        print(f"\nFinal Real Time Factors and Stats:")
        if en_stats:
            print(f"  English transcription RTF: {en_stats['rtf'] if en_stats['rtf'] is not None else 'N/A'}x")
            print(f"    Total duration: {en_stats['duration']:.2f} seconds")
            print(f"    Total size: {en_stats['size'] / (1024**3):.2f} GB")
            print(f"    Batch size: {en_stats['batch_size']}")
            print(f"    Total files processed: {len(english_vcons_to_process) if 'english_vcons_to_process' in locals() else 'N/A'}")
        if non_en_stats:
            print(f"  Non-English transcription RTF: {non_en_stats['rtf'] if non_en_stats['rtf'] is not None else 'N/A'}x")
            print(f"    Total duration: {non_en_stats['duration']:.2f} seconds")
            print(f"    Total size: {non_en_stats['size'] / (1024**3):.2f} GB")
            print(f"    Batch size: {non_en_stats['batch_size']}")
            print(f"    Total files processed: {len(non_english_vcons_to_process) if 'non_english_vcons_to_process' in locals() else 'N/A'}")
        if lang_stats:
            print(f"  Language detection:")
            print(f"    Total duration: {lang_stats['duration']:.2f} seconds")
            print(f"    Total size: {lang_stats['size'] / (1024**3):.2f} GB")
            print(f"    Batch size: {lang_stats['batch_size']}")
            print(f"    Total files processed: {len(get_valid_wav_files(settings.source_dir))}")

    except Exception as e:
        logging.error(f"Error in main function: {str(e)}")
        
        traceback.print_exc()

    total_elapsed = time.time() - total_start_time
    logging.info(f"Total runtime: {total_elapsed:.2f} seconds")

if __name__ == "__main__":
    if "print_vcons" in sys.argv:
        print_all_vcons()
    else:
        main()