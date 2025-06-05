from mongo_utils import get_mongo_collection
from bson import ObjectId # To check for ObjectId type if necessary
import vcon # Import the vcon library

def print_vcon_details():
    vcons_collection = get_mongo_collection()
    
    print("Fetching vCon information using vcon library...")
    
    for vcon_doc_from_db in vcons_collection.find():
        # Prepare the dictionary for vcon.from_dict()
        # The Vcon object uses 'uuid', MongoDB uses '_id' by default.
        # vcon_utils.py should have stored a 'uuid' from vcon_obj.to_dict().
        dict_for_load = vcon_doc_from_db.copy()
        doc_id_for_print = str(dict_for_load.get("_id", "UNKNOWN_ID"))
        dict_for_load.pop('_id', None) # Remove MongoDB _id, rely on uuid within the doc

        try:
            v_obj = vcon.Vcon(dict_for_load)
            # If from_dict worked, v_obj.uuid should be populated.
            # If 'uuid' was not in dict_for_load, from_dict might raise error or uuid might be None.
            if not v_obj.uuid:
                # Fallback if uuid wasn't in the dict or not set by from_dict correctly
                # This case implies an issue with how vCons are stored/structured by vcon_utils or main.py
                v_obj.uuid = doc_id_for_print 
                print(f"Warning: vCon loaded from DB (original _id: {doc_id_for_print}) did not have a UUID in its dictionary representation or from_dict did not set it. Using original _id as UUID for printing.")

        except Exception as e:
            print(f"--- vCon DB _id (raw): {doc_id_for_print} ---")
            print(f"  Error loading DB doc into Vcon object: {e}")
            print(f"  Raw document from DB: {vcon_doc_from_db}")
            print("-" * (40 + len(doc_id_for_print)))
            print()
            continue

        vcon_id_to_display = v_obj.uuid
        transcriptions = []
        languages = []
        
        # Access analysis data through the vcon object's properties
        # Assuming v_obj.analysis is a list of dicts as per vCon structure
        if hasattr(v_obj, 'analysis') and isinstance(v_obj.analysis, list):
            for analysis_entry in v_obj.analysis:
                analysis_type = analysis_entry.get("type")
                analysis_body = analysis_entry.get("body")
                
                if analysis_type == "transcription":
                    if isinstance(analysis_body, str):
                        transcriptions.append(analysis_body)
                    elif isinstance(analysis_body, list): # Should not happen if body is just text
                        transcriptions.extend([str(item) for item in analysis_body]) 
                elif analysis_type == "language_identification":
                    if isinstance(analysis_body, list):
                        languages.extend(analysis_body)
                    elif isinstance(analysis_body, str): # e.g. "en"
                        languages.append(analysis_body)
        else:
            # This case means the Vcon object (as understood by the library) has no .analysis list
            print(f"Note: Vcon object for UUID {vcon_id_to_display} (original _id: {doc_id_for_print}) has no 'analysis' attribute or it's not a list after from_dict.")

        print(f"--- vCon UUID: {vcon_id_to_display} (DB _id: {doc_id_for_print}) ---")
        if languages:
            flat_languages = []
            for lang_item in languages:
                if isinstance(lang_item, list): # e.g. ["en", "es"]
                    flat_languages.extend(lang_item)
                else: # e.g. "en"
                    flat_languages.append(lang_item)
            unique_languages = sorted(list(set(flat_languages)))
            print(f"  Identified Languages: {unique_languages}")
        else:
            print("  Identified Languages: Not found")
            
        if transcriptions:
            print("  Transcriptions:")
            for i, trans_text in enumerate(transcriptions):
                print(f"    [{i+1}]: {trans_text}")
        else:
            print("  Transcriptions: Not found")
        print("-" * (40 + len(str(vcon_id_to_display))))
        print()

if __name__ == "__main__":
    print_vcon_details() 

    # def load_model(model_name):
#     if model_name == "openai/whisper-tiny":
#         return load_whisper_tiny()
#     else:
#         return load_nvidia(model_name)

# class AIModel:
#     def __init__(self):
#         self.model = None
#         self.model_name = None

#     def unload(self):
#         del self.model
#         del self.model_name
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#         self.model = None
#         self.model_name = None

#     def load(self, model_to_load):
#         if model_to_load != self.model_name:
#             self.unload()
#             self.model = load_model(model_to_load)
#             self.model_name = model_to_load

#     def load_en_transcription(self):
#         self.load(transcribe_english_model_name)
    
#     def load_non_en_transcription(self):
#         self.load(transcribe_nonenglish_model_name)
    
#     def load_lang_detect(self):
#         self.load(identify_languages_model_name)

    # def transcribe(self, wav_files, english_only=False):
    #     # Load the correct model if needed
    #     if english_only:
    #         self.load(transcribe_english_model_name)
    #     else:
    #         self.load(transcribe_nonenglish_model_name)

    #     # NVIDIA NeMo model: batch by total file size, using 1/4 GPU RAM
    #     batch_bytes = calculate_batch_bytes()
    #     batches = make_wav_batches(wav_files, batch_bytes)

    #     all_transcriptions = []
    #     for batch in batches:
    #         transcriptions = self.model.transcribe(batch)
    #         all_transcriptions.extend(transcriptions)


    #     return all_transcriptions

    def loaded_model_mode(self):
        if self.model_name == transcribe_english_model_name:
            return "en"
        elif self.model_name == transcribe_nonenglish_model_name:
            return "non_en"
        elif self.model_name == identify_languages_model_name:
            return "lang_detect"
        else:
            return None  # Return None when no model is loaded - let the system determine what to load
        
    def load_by_mode(self, mode):
        if mode == "en":
            self.load_en_transcription()
        elif mode == "non_en":
            self.load_non_en_transcription()
        elif mode == "lang_detect":
            self.load_lang_detect()

# def resample_wav_maybe(wav, sample_rate, target_sample_rate=16000):
#     if sample_rate != target_sample_rate:
#         resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
#         wav = resampler(wav)
#         wav = wav.squeeze().numpy()
#     return wav

# def load_and_resample_wavs(all_wav_paths, target_sample_rate=16000):
#     """
#     Loads and resamples a list of wav files to the target sample rate.
#     Applies VAD and removes silent sections from the waveform.
#     Returns a tuple: (list of numpy arrays (waveforms), list of valid indices, list of unreadable file paths)
#     """
#     wavs = []
#     valid_indices = []
#     unreadable_files = []
#     vad = torchaudio.transforms.Vad(sample_rate=target_sample_rate, trigger_level=0.5)
#     for idx, wav_path in enumerate(all_wav_paths):
#         if is_readable_wav(wav_path):
#             try:
#                 raw_wav, sample_rate = torchaudio.load(wav_path)
#                 # Resample first
#                 wav = resample_wav_maybe(raw_wav, sample_rate, target_sample_rate=target_sample_rate)
#                 # Convert to mono if necessary (always convert to mono)
#                 if isinstance(wav, torch.Tensor):
#                     if wav.ndim > 1 and wav.shape[0] > 1:
#                         wav = wav.mean(dim=0, keepdim=True)
#                 else:
#                     if wav.ndim > 1 and wav.shape[0] > 1:
#                         wav = wav.mean(axis=0, keepdims=True)
#                 # Convert to torch tensor for VAD
                
#                 # Ensure the tensor is on the CPU before converting to NumPy.
#                 # .squeeze() removes the batch dimension if it's 1 (e.g., from [1, M] to [M,]).
#                 processed_wav = wav.squeeze().cpu().numpy()
#                 wavs.append(processed_wav)
#                 valid_indices.append(idx)
#                 del processed_wav
#                 del raw_wav
#                 del wav
#                 gc.collect()
#                 if torch.cuda.is_available():
#                     torch.cuda.empty_cache()
#                     torch.cuda.synchronize()
#             except Exception as e:
#                 unreadable_files.append(wav_path)
#                 logging.warning(f"Failed to load wav file {wav_path}: {e}")
#         else:
#             unreadable_files.append(wav_path)
#     return wavs, valid_indices, unreadable_files

# def split_wavs_into_batches(wavs, batch_size):
#     """
#     Splits a list of wavs into batches of size batch_size.
#     """
#     return [wavs[i:i + batch_size] for i in range(0, len(wavs), batch_size)]


# def identify_languages(all_wav_paths, model_and_processor, threshold=None, vcon_ids=None, vcon_collection=None):
#     model, processor = model_and_processor
#     if threshold is None:
#         threshold = lang_detect_threshold
#     device = get_device()

#     # Use make_wav_batches for batching by file size
#     batch_bytes = calculate_batch_bytes()
#     all_wav_paths_batched = make_wav_batches(all_wav_paths, batch_bytes)

#     all_wav_languages_detected = []
#     corrupt_indices = []

#     # Keep track of the global index across all batches
#     global_idx = 0

#     for batch_idx, wav_paths in enumerate(all_wav_paths_batched):
#         gc.collect()
#         # Clear GPU cache before processing each batch
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#             torch.cuda.synchronize()
        
#         wavs, valid_indices, unreadable_files = load_and_resample_wavs(wav_paths, target_sample_rate=16000)
        
#         # Mark corrupt files in DB
#         if vcon_ids and vcon_collection is not None and unreadable_files:
#             for idx, wav_path in enumerate(wav_paths):
#                 if wav_path in unreadable_files:
#                     vcon_id = vcon_ids[global_idx + idx]
#                     analysis = {
#                         "type": "corrupt",
#                         "body": "Unreadable or corrupt audio file",
#                         "encoding": "none"
#                     }
#                     vcon_collection.update_one({"_id": vcon_id}, {"$push": {"analysis": analysis}})
        
#         if not wavs:
#             all_wav_languages_detected.extend([[] for _ in wav_paths])
#             global_idx += len(wav_paths)
#             continue
#         print(f"Processing {len(wavs)} wavs")

#         # Process in smaller chunks to avoid memory accumulation
#         input_features = None
#         decoder_input_ids = None
#         model_output = None
#         logits = None
        
#         try:
#             # Create input features and move to GPU
#             input_features = processor(wavs, sampling_rate=16000, return_tensors="pt").input_features
#             input_features = input_features.to(device)
            
#             # Create decoder input ids
#             decoder_input_ids = torch.tensor([[whisper_start_transcription_token_id]] * len(wavs)).to(device)
            
#             # Run model inference with no_grad to prevent gradient accumulation
#             with torch.no_grad():
#                 model_output = model(input_features, decoder_input_ids=decoder_input_ids)
#                 logits = model_output.logits
#                 # Move logits to CPU immediately to free GPU memory
#                 logits = logits[:, 0, :].cpu()  # (batch, vocab_size)
            
#             # Delete GPU tensors immediately
#             del input_features
#             del decoder_input_ids
#             del model_output
#             input_features = None
#             decoder_input_ids = None
#             model_output = None
            
#             # Force GPU memory cleanup
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
#                 torch.cuda.synchronize()
            
#             # Process results using CPU tensors
#             valid_counter = 0
#             for index in range(len(wav_paths)):
#                 if index in valid_indices:
#                     lang_logits = logits[valid_counter, whisper_token_ids]
#                     lang_probs = torch.softmax(lang_logits, dim=-1).numpy()
#                     # Convert whisper language tokens to simplified language codes
#                     wav_languages_detected = [whisper_tokens[j] for j, prob in enumerate(lang_probs) if prob >= threshold]
                    
#                     # If no languages meet the threshold, use the highest probability language
#                     if not wav_languages_detected:
#                         max_prob_idx = lang_probs.argmax()
#                         wav_languages_detected = [whisper_tokens[max_prob_idx]]
                    
#                     all_wav_languages_detected.append(wav_languages_detected)
#                     valid_counter += 1
#                 else:
#                     all_wav_languages_detected.append([])
        
#         except Exception as e:
#             # Make sure to clean up GPU memory even if there's an error
#             if input_features is not None:
#                 del input_features
#             if decoder_input_ids is not None:
#                 del decoder_input_ids
#             if model_output is not None:
#                 del model_output
#             if logits is not None:
#                 del logits
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
#             raise e
        
#         finally:
#             # Final cleanup
#             if logits is not None:
#                 del logits
#             del wavs
#             gc.collect()
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
#                 torch.cuda.synchronize()
        
#         global_idx += len(wav_paths)
    
#     # Final memory cleanup
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         torch.cuda.synchronize()
    
#     assert len(all_wav_languages_detected) == len(all_wav_paths)
#     return all_wav_languages_detected
import paramiko
from ai import AIModel
import mongo_utils
import settings
import secrets_utils
import make_vcons_from_sftp
from utils import get_hostname, calculate_batch_bytes, reset_gpu_memory_stats, max_gpu_memory_usage, get_all_filenames_from_sftp, seconds_to_ydhms
from audio import is_wav_filename, clear_cache_directory, is_readable_wav, get_wav_duration
import os
import shutil
import threading
import time
from sftp import download_sftp_file, sftp_connect, get_sftp_file_size, parse_sftp_url
import argparse
import vcon as vcon_library_module
import json
import logging
import print_vcon_info
import torch
import gc

from vcon_utils import get_vcon_collection

def reserve_vcons(vcons, vcons_collection, size_bytes):
    hostname = get_hostname()
    reserved = []
    total_size = 0
    for vcon in vcons:
        size = vcon.get("size", 0)
        if total_size + size > size_bytes:
            break
        result = vcons_collection.update_one({"_id": vcon["_id"], "processed_by": None}, {"$set": {"processed_by": hostname}})
        if result.modified_count == 1:
            reserved.append(vcon)
            total_size += size
    return reserved

def find_vcons_lang_detect(vcons_collection):
    query = {
        "processed_by": None,
        "$and": [
            {"analysis": {"$not": {"$elemMatch": {"type": "language_identification"}}}},
            {"analysis": {"$not": {"$elemMatch": {"type": "corrupt"}}}}
        ],
        "attachments": {"$elemMatch": {"type": "audio"}},
        "corrupt": {"$ne": True}
    }
    vcons = list(vcons_collection.find(query))
    return vcons

def reserve_vcons_for_lang_detect(vcons_collection, size_bytes):
    # Find vCons needing language detection and not being processed
    vcons = find_vcons_lang_detect(vcons_collection)
    return reserve_vcons(vcons, vcons_collection, size_bytes)

def find_vcons_en_transcription(vcons_collection):
    query = {
        "processed_by": None,
        "$and": [
            {"analysis": {"$elemMatch": {"type": "language_identification", "body": ["en"]}}},
            {"analysis": {"$not": {"$elemMatch": {"type": "transcription"}}}},
            {"analysis": {"$not": {"$elemMatch": {"type": "corrupt"}}}}
        ],
        "attachments": {"$elemMatch": {"type": "audio"}},
        "corrupt": {"$ne": True}
    }
    vcons = list(vcons_collection.find(query))
    return vcons

def reserve_vcons_for_en_transcription(vcons_collection, size_bytes):
    # Find vCons with language_identification == ["en"] and no transcription, not being processed
    vcons = find_vcons_en_transcription(vcons_collection)
    return reserve_vcons(vcons, vcons_collection, size_bytes)

def find_vcons_non_en_transcription(vcons_collection):
    query = {
        "processed_by": None,
        "$and": [
            {"analysis": {"$elemMatch": {"type": "language_identification", "body": {"$ne": ["en"]}}}},
            {"analysis": {"$not": {"$elemMatch": {"type": "transcription"}}}},
            {"analysis": {"$not": {"$elemMatch": {"type": "corrupt"}}}}
        ],
        "attachments": {"$elemMatch": {"type": "audio"}},
        "corrupt": {"$ne": True}
    }
    vcons = list(vcons_collection.find(query))
    return vcons

def reserve_vcons_for_non_en_transcription(vcons_collection, size_bytes):
    vcons = find_vcons_non_en_transcription(vcons_collection)
    return reserve_vcons(vcons, vcons_collection, size_bytes)

def make_modes_to_try(model_mode):
    all_modes = ["lang_detect", "en", "non_en"]
    if model_mode:
        all_modes.remove(model_mode)
        return [model_mode] + all_modes
    else:
        return all_modes

def reserve_vcons_for_processing(model_mode, vcons_collection, size_bytes):
    # Try to reserve for the current model mode
    modes_to_try = make_modes_to_try(model_mode)
    reserved = []
    for mode in modes_to_try:
        if mode == "lang_detect":
            reserved = reserve_vcons_for_lang_detect(vcons_collection, size_bytes)
            if reserved:
                return reserved, mode
        elif mode == "en":
            reserved = reserve_vcons_for_en_transcription(vcons_collection, size_bytes)
            if reserved:
                return reserved, mode
        elif mode == "non_en":
            reserved = reserve_vcons_for_non_en_transcription(vcons_collection, size_bytes)
            if reserved:
                return reserved, mode
    return [], None

def _download_vcons_to_cache(vcons_to_process, sftp):
    cache_dir = settings.cache_directory
    os.makedirs(cache_dir, exist_ok=True)
    vcon_collection = get_mongo_collection()
    
    for vcon in vcons_to_process:
        for attachment in vcon.get('attachments', []):
            if attachment.get('type') == 'audio':
                wav_url = attachment.get('body')
                if not is_wav_filename(wav_url):
                    continue
                wav_basename = os.path.basename(wav_url)
                cache_path = os.path.join(cache_dir, wav_basename)
                temp_path = cache_path + ".part"
                if os.path.exists(cache_path):
                    continue
                try:
                    if wav_url.startswith("sftp://"):
                        download_sftp_file(wav_url, temp_path, sftp)
                    else:
                        shutil.copy2(wav_url, temp_path)
                    os.rename(temp_path, cache_path)
                    
                    # Validate the downloaded WAV file
                    if not is_readable_wav(cache_path):
                        print(f"Downloaded WAV file is corrupt: {wav_url}")
                        # Mark as corrupt in database
                        analysis = {
                            "type": "corrupt",
                            "body": "Corrupt or unreadable audio file",
                            "encoding": "none"
                        }
                        try:
                            result = vcon_collection.update_one(
                                {"_id": vcon["_id"]},
                                {"$push": {"analysis": analysis}, "$set": {"corrupt": True}, "$unset": {"processed_by": ""}}
                            )
                            if result.modified_count == 1:
                                print(f"Marked vCon {vcon['_id']} as corrupt in database")
                            else:
                                print(f"Failed to mark vCon {vcon['_id']} as corrupt")
                        except Exception as e:
                            print(f"Error marking vCon {vcon['_id']} as corrupt: {e}")
                        
                        # Remove the corrupt file from cache
                        if os.path.exists(cache_path):
                            os.remove(cache_path)
                        
                except Exception as e:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    print(f"Failed to download {wav_url} to cache: {e}")
    print(f"Downloaded {len(vcons_to_process)} vcons to cache")

def cleanup_cache_directory():
    """Clean up any leftover files in the cache directory from previous runs."""
    cache_dir = settings.cache_directory
    if not os.path.exists(cache_dir):
        return
    
    files_removed = 0
    total_size_removed = 0
    corrupt_files_removed = 0
    
    for filename in os.listdir(cache_dir):
        file_path = os.path.join(cache_dir, filename)
        if os.path.isfile(file_path):
            try:
                file_size = os.path.getsize(file_path)
                
                # Check if it's a WAV file and validate it
                if is_wav_filename(filename):
                    if not is_readable_wav(file_path):
                        print(f"Found corrupt WAV file in cache during cleanup: {file_path}")
                        corrupt_files_removed += 1
                
                os.remove(file_path)
                files_removed += 1
                total_size_removed += file_size
            except Exception as e:
                print(f"Failed to clean up {file_path}: {e}")
    
    if files_removed > 0:
        print(f"Cleaned up {files_removed} leftover files ({total_size_removed / (1024*1024):.1f} MB) from cache directory")
        if corrupt_files_removed > 0:
            print(f"  - {corrupt_files_removed} of these were corrupt WAV files")

def load_vcons_in_background(vcons_to_process, sftp):
    """
    Start a background thread to download all wav files referenced in the vcons' attachments into the cache folder.
    """
    thread = threading.Thread(target=_download_vcons_to_cache, args=(vcons_to_process, sftp), daemon=True)
    thread.start()
    return thread

def process_vcons(download_thread, vcons_to_process, loaded_ai, mode):
    # Reset GPU memory stats for accurate tracking
    
    # Map vcon_id to cache wav path
    cache_dir = settings.cache_directory
    vcon_id_to_cache_path = {}
    vcon_id_to_vcon = {}
    for vcon in vcons_to_process:
        vcon_id = vcon['_id']
        vcon_id_to_vcon[vcon_id] = vcon
        for attachment in vcon.get('attachments', []):
            if attachment.get('type') == 'audio':
                wav_url = attachment.get('body')
                if is_wav_filename(wav_url):
                    wav_basename = os.path.basename(wav_url)
                    cache_path = os.path.join(cache_dir, wav_basename)
                    vcon_id_to_cache_path[vcon_id] = cache_path
                    break

    vcon_collection = get_mongo_collection()
    loaded_ai.load_by_mode(mode)

    vcon_ids = list(vcon_id_to_cache_path.keys())
    processed_ids = set()
    total_bytes_to_process = sum(vcon_id_to_vcon[vid].get('size', 0) for vid in vcon_ids)
    total_bytes_processed = 0
    start_time = time.time()

    def available_vcon_ids():
        valid_ids = []
        for vid in vcon_ids:
            cache_path = vcon_id_to_cache_path[vid]
            if os.path.exists(cache_path):
                # Additional validation: check if the WAV file is readable
                if is_readable_wav(cache_path):
                    valid_ids.append(vid)
                else:
                    print(f"Found corrupt WAV file in cache: {cache_path}")
                    # Mark as corrupt in database
                    analysis = {
                        "type": "corrupt",
                        "body": "Corrupt or unreadable audio file",
                        "encoding": "none"
                    }
                    try:
                        result = vcon_collection.update_one(
                            {"_id": vid},
                            {"$push": {"analysis": analysis}, "$set": {"corrupt": True}, "$unset": {"processed_by": ""}}
                        )
                        if result.modified_count == 1:
                            print(f"Marked vCon {vid} as corrupt in database")
                        else:
                            print(f"Failed to mark vCon {vid} as corrupt")
                    except Exception as e:
                        print(f"Error marking vCon {vid} as corrupt: {e}")
                    
                    # Remove the corrupt file from cache
                    try:
                        os.remove(cache_path)
                    except Exception as e:
                        print(f"Failed to remove corrupt file {cache_path}: {e}")
        return valid_ids

    print(f"Starting {mode} processing for {len(vcon_ids)} vcons ({total_bytes_to_process / (1024*1024):.1f} MB total)")

    while True:
        avail_ids = [vid for vid in available_vcon_ids() if vid not in processed_ids]
        if not avail_ids:
            if not download_thread.is_alive():
                break
            time.sleep(1)
            continue
        batch_files = [vcon_id_to_cache_path[vid] for vid in avail_ids]
        batch_vcons = [vcon_id_to_vcon[vid] for vid in avail_ids]
        batch_bytes = sum(vcon_id_to_vcon[vid].get('size', 0) for vid in avail_ids)

        if mode == 'lang_detect':
            results = loaded_ai.identify_languages(batch_files, vcon_ids=avail_ids, vcon_collection=vcon_collection)
            gc.collect()
            # Force GPU memory cleanup after language detection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            for vcon, langs in zip(batch_vcons, results):
                analysis = {
                    "type": "language_identification",
                    "dialog": [0],
                    "vendor": "whisper-tiny",
                    "body": langs,
                    "encoding": "none"
                }
                update_success = False
                try:
                    result = vcon_collection.update_one({"_id": vcon["_id"]}, {"$push": {"analysis": analysis}})
                    if result.modified_count == 1:
                        update_success = True
                    else:
                        print(f"Warning: Language identification update for vCon {vcon['_id']} did not modify any document")
                except Exception as e:
                    print(f"Error updating language identification for vCon {vcon['_id']}: {e}")
                
                # Only release the vcon if the update was successful
                if update_success:
                    try:
                        vcon_collection.update_one({"_id": vcon["_id"]}, {"$unset": {"processed_by": ""}})
                    except Exception as e:
                        print(f"Error unsetting processed_by for vCon {vcon['_id']}: {e}")
                else:
                    print(f"Keeping vCon {vcon['_id']} reserved due to failed language identification update")
        else:
            successful_vcon_ids = []
            try:
                transcriptions_result = loaded_ai.transcribe(batch_files, english_only=(mode == "en"))

                if not isinstance(transcriptions_result, list) or len(transcriptions_result) != len(batch_vcons):
                    print(f"ERROR main.py: Transcription result is not a list or length mismatch. Got: {len(transcriptions_result) if isinstance(transcriptions_result, list) else type(transcriptions_result)}, Expected: {len(batch_vcons)}. Skipping this batch.")
                    # To prevent further errors with zip, we should probably skip this batch or handle it carefully.
                    # For now, an empty list will make the loop not run.
                    transcriptions_for_loop = [] 
                else:
                    transcriptions_for_loop = transcriptions_result

                for vcon_dict_from_batch, transcription_text in zip(batch_vcons, transcriptions_for_loop):
                    if not transcription_text:
                        print(f"INFO main.py: Skipping empty transcription for vCon {vcon_dict_from_batch.get('_id')}")
                        continue

                    original_bson_id = vcon_dict_from_batch["_id"]

                    try:
                        current_vcon_doc_from_db = vcon_collection.find_one({"_id": original_bson_id})
                        if not current_vcon_doc_from_db:
                            print(f"ERROR main.py: Could not re-fetch vCon {original_bson_id} for Vcon object creation.")
                            continue

                        dict_for_vcon_load = current_vcon_doc_from_db.copy()
                        dict_for_vcon_load.pop('_id', None) 

                        try:
                            v_obj = vcon_library_module.Vcon(dict_for_vcon_load)
                            logging.info(f"Loaded vCon {v_obj.uuid} from dict using constructor for transcription processing.")
                        except Exception as e:
                            logging.error(f"Error initializing Vcon from dict for {dict_for_vcon_load.get('uuid', 'Unknown UUID')} (transcription part): {e}")
                            continue

                        dialog_index_to_attach = 0
                        if not v_obj.dialog or len(v_obj.dialog) == 0:
                            print(f"Warning main.py: No dialogs found in vCon {v_obj.uuid}. Attempting vCon-level analysis (dialog=-1).")
                            dialog_index_to_attach = -1
                        
                        # If transcription_text is a Hypothesis object, extract the .text attribute
                        body_text = transcription_text.text if hasattr(transcription_text, 'text') else transcription_text
                        analysis_result_index = v_obj.add_analysis(
                            dialog=dialog_index_to_attach,
                            type="transcription", # This must match what print_vcon_info.py looks for
                            body=body_text,
                            vendor=loaded_ai.model_name if hasattr(loaded_ai, 'model_name') else "unknown",
                            encoding="none"
                        )
                        
                        if analysis_result_index is None: # Should not be None if add_analysis is successful
                            print(f"WARNING main.py: add_analysis for transcription on vCon {v_obj.uuid} returned None. Transcription may not be saved.")
                            # continue # Optional: skip saving if add_analysis didn't confirm addition.

                        save_dict = v_obj.to_dict()

                        update_result = vcon_collection.replace_one(
                            {"_id": original_bson_id}, 
                            save_dict 
                        )

                        if update_result.modified_count == 1:
                            successful_vcon_ids.append(original_bson_id)
                        elif update_result.matched_count == 1 and update_result.modified_count == 0:
                            print(f"WARNING main.py: Transcription update for vCon {original_bson_id} matched but did not modify document (data might be identical).")
                            successful_vcon_ids.append(original_bson_id) # Still consider it successful if data was identical
                        else:
                            print(f"ERROR main.py: Transcription update for vCon {original_bson_id} failed. Matched: {update_result.matched_count}, Modified: {update_result.modified_count}")
                    except Exception as e:
                        print(f"Error processing/saving transcription for vCon {original_bson_id} using vcon library: {e}")
                        import traceback
                        print(traceback.format_exc())
            except Exception as e:
                print(f"Error in transcription batch with vcon library: {e}")
            finally:
                # Only release vcons that were successfully updated
                for vid in successful_vcon_ids:
                    try:
                        vcon_collection.update_one({"_id": vid}, {"$unset": {"processed_by": ""}})
                    except Exception as e:
                        print(f"Error unsetting processed_by for vCon {vid}: {e}")
                
                # For failed vcons, keep them reserved but log them
                failed_vcon_ids = [vid for vid in avail_ids if vid not in successful_vcon_ids]
                if failed_vcon_ids:
                    print(f"Keeping {len(failed_vcon_ids)} vCons reserved due to failed transcription updates: {failed_vcon_ids}")
                    # Release them anyway to avoid blocking indefinitely
                    # for vid in failed_vcon_ids:
                    #     try:
                    #         vcon_collection.update_one({"_id": vid}, {"$unset": {"processed_by": ""}})
                    #     except Exception as e:
                    #         print(f"Error unsetting processed_by for failed vCon {vid}: {e}")

        total_bytes_processed += batch_bytes
        processed_ids.update(avail_ids)
        batch_time = time.time() - start_time
        progress_pct = (total_bytes_processed / total_bytes_to_process) * 100 if total_bytes_to_process > 0 else 0
        print(f"Processed {len(avail_ids)} files, {progress_pct:.1f}% complete, max GPU memory usage: {max_gpu_memory_usage()/(1024**3):.2f} GB")

        for file_path in batch_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")

        if not download_thread.is_alive() and len(processed_ids) == len(vcon_ids):
            break
    clear_cache_directory()

    total_time = time.time() - start_time
    final_mb_per_sec = (total_bytes_processed / (1024*1024)) / total_time if total_time > 0 else 0
    print(f"\n{mode} processing completed: {total_bytes_processed/(1024*1024):.1f} MB in {total_time:.1f}s ({final_mb_per_sec:.1f} MB/s)")

def main(sftp_url, measure_mode=False):
    secrets_utils.init()
    mongo_utils.init()
    loaded_ai = AIModel()
    vcons_collection = get_vcon_collection()
    sftp = sftp_connect(sftp_url)

    total_files = None
    total_wav_files = 0
    total_wav_files_size = 0
    total_wav_files_duration = 0
    total_valid_wav_files = 0
    total_invalid_wav_files = 0
    delete_time = None
    slurp_time = None
    total_start_time = None

    if measure_mode:
        parsed_sftp_url = parse_sftp_url(sftp_url)
        print("Fetching all filenames from SFTP...")
        files = list(get_all_filenames_from_sftp(sftp, parsed_sftp_url["path"]))
        total_files = len(files)

        print("downloading and measuring wav files...")
        for file in files:
            if is_wav_filename(file):
                total_wav_files += 1
                # Download the wav file to a temporary location
                temp_path = os.path.join(settings.cache_directory, os.path.basename(file))
                try:
                    # Build the correct SFTP URL for this file
                    full_sftp_url = f"sftp://{parsed_sftp_url['username']}@{parsed_sftp_url['hostname']}:{parsed_sftp_url['port']}{file}"
                    download_sftp_file(full_sftp_url, temp_path, sftp)
                    # Verify the wav file is legitimate
                    if is_readable_wav(temp_path):
                        # If legitimate, increment counter and add duration
                        total_wav_files_size += os.path.getsize(temp_path)
                        total_wav_files_duration += get_wav_duration(temp_path)
                        total_valid_wav_files += 1
                        print(f"Valid wav file: {file}")
                    else:
                        total_invalid_wav_files += 1
                finally:
                    # Clean up the temporary file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
    
        total_start_time = time.time()
        delete_start_time = time.time()
        print("Deleting all vcons...")
        delete_all_vcons()
        delete_time = time.time() - delete_start_time
        slurp_start_time = time.time()
        print("Slurping into db")
        make_vcons_from_sftp.main(sftp_url)
        slurp_time = time.time() - slurp_start_time

    # Clean up any leftover files from previous runs
    cleanup_cache_directory()

    process_start_time = time.time()
    while True:
        print("Reserving vcons for processing...")
        vcons_to_process, mode = reserve_vcons_for_processing(loaded_ai.loaded_model_mode(),
                                                              vcons_collection,
                                                              settings.total_vcon_filesize_to_process_bytes)
        print(f"Reserved {len(vcons_to_process)} vcons totaling {sum(vcon.get('size', 0) for vcon in vcons_to_process)/(1024*1024):.2f} MB for processing in mode {mode}")
        
        # If no vcons were reserved or mode is None, wait and try again
        if vcons_to_process:
            thread = load_vcons_in_background(vcons_to_process, sftp)
            process_vcons(thread, vcons_to_process, loaded_ai, mode)
        else:
            if measure_mode:
                break
            else:
                time.sleep(1)
    total_time = time.time() - total_start_time
    process_time = time.time() - process_start_time

    if measure_mode:
        print(f"Total files: {total_files}")
        print(f"Total wav files: {total_wav_files}")
        print(f"Total wav files size: {total_wav_files_size / (1024*1024):.2f} MB")
        print(f"Total wav files duration: {total_wav_files_duration:.2f} seconds ({seconds_to_ydhms(total_wav_files_duration)} minutes)")
        print(f"Total valid wav files: {total_valid_wav_files}")
        print(f"Total invalid wav files: {total_invalid_wav_files}")
        print(f"RTF: {total_wav_files_duration / process_time:.2f}x")
        print(f"Data rate: {(total_wav_files_size / process_time) / (1024*1024):.2f} MB/s")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Slurp time: {slurp_time:.2f} seconds")
        print(f"Delete time: {delete_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribey main entry point")
    parser.add_argument("mode", choices=["head", "worker", "slurp", "print", "delete_all", "measure"], help="head:slurp and run worker. ")
    parser.add_argument("--sftp_url", type=str, default=settings.sftp_url, help="Override SFTP URL (applies to both head and worker)")
    parser.add_argument("--production", action="store_true", default=False, help="Enable production mode (applies to both head and worker)")
    args = parser.parse_args()

    settings.debug = not args.production

    if args.mode == "head":
        if settings.debug:
            delete_all_vcons()
        sftp_thread = threading.Thread(target=make_vcons_from_sftp.main, args=(args.sftp_url,), daemon=True)
        sftp_thread.start()
        main(args.sftp_url)
    elif args.mode == "worker":
        main(args.sftp_url)
    elif args.mode == "slurp":
        if settings.debug:
            delete_all_vcons()
        make_vcons_from_sftp.main(args.sftp_url)
    elif args.mode == "print":
        print_vcon_info.print_vcon_details()
    elif args.mode == "delete_all":
        delete_all_vcons()
    elif args.mode == "measure":
        main(args.sftp_url, measure_mode=True)

# architecture
# We can fit all AI models in GPU memory. 
# So, what if we do the following:
# In background, always reserve vcons until cache is full, download said vcons. 

# now repeat. 
# Move all wavs in cache to processing.
# move all wavs to RAM.
# Apply VAD.
# Move all wavs to GPU.
# Resample to 16000kHz.
# Identify languages 
# transcribe english
# transcribe non-english
# Start a thread to update vcons. 
# (If model size is a problem, save to disk, handle later.)
