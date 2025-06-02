import paramiko
from transcription_models import AIModel
from mongo_utils import get_mongo_collection, delete_all_vcons
import settings
import make_vcons_from_sftp
from utils import get_hostname, calculate_batch_bytes, print_gpu_memory_usage, reset_gpu_memory_stats, max_gpu_memory_usage
from wavs import is_wav_filename, clear_cache_directory, is_readable_wav
import os
import shutil
import threading
import time
from sftp_utils import download_sftp_file, sftp_connect, get_sftp_file_size, parse_sftp_url
import argparse
import vcon as vcon_library_module
import json
import logging

def reserve_vcons(vcons, vcons_collection, size_bytes):
    hostname = get_hostname()
    reserved = []
    total_size = 0
    for vcon in vcons:
        size = vcon.get("size", 0)
        if total_size + size > size_bytes:
            break
        result = vcons_collection.update_one({"_id": vcon["_id"], "being_processed_by": None}, {"$set": {"being_processed_by": hostname}})
        if result.modified_count == 1:
            reserved.append(vcon)
            total_size += size
    return reserved

def find_vcons_lang_detect(vcons_collection):
    query = {
        "being_processed_by": None,
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
        "being_processed_by": None,
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
        "being_processed_by": None,
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
                                {"$push": {"analysis": analysis}, "$set": {"corrupt": True}, "$unset": {"being_processed_by": ""}}
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
                            {"$push": {"analysis": analysis}, "$set": {"corrupt": True}, "$unset": {"being_processed_by": ""}}
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
                        vcon_collection.update_one({"_id": vcon["_id"]}, {"$unset": {"being_processed_by": ""}})
                    except Exception as e:
                        print(f"Error unsetting being_processed_by for vCon {vcon['_id']}: {e}")
                else:
                    print(f"Keeping vCon {vcon['_id']} reserved due to failed language identification update")
        else:
            successful_vcon_ids = []
            try:
                print(f"DEBUG main.py: About to call loaded_ai.transcribe for {len(batch_files)} files in mode '{mode}'. First file: {batch_files[0] if batch_files else 'N/A'}")
                transcriptions_result = loaded_ai.transcribe(batch_files, english_only=(mode == "en"))
                print(f"DEBUG main.py: loaded_ai.transcribe returned: {transcriptions_result}")

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
                    print(f"DEBUG main.py: Processing transcription for vCon BSON ID {original_bson_id}")

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
                        
                        # print(f"DEBUG main.py: vCon {v_obj.uuid} analysis BEFORE add: {v_obj.analysis}")

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
                        print(f"DEBUG main.py: v_obj.add_analysis for transcription returned: {analysis_result_index} for vCon UUID {v_obj.uuid}")
                        print(f"DEBUG main.py: vCon {v_obj.uuid} analysis AFTER add: {v_obj.analysis}")
                        
                        if analysis_result_index is None: # Should not be None if add_analysis is successful
                            print(f"WARNING main.py: add_analysis for transcription on vCon {v_obj.uuid} returned None. Transcription may not be saved.")
                            # continue # Optional: skip saving if add_analysis didn't confirm addition.

                        save_dict = v_obj.to_dict()
                        # print(f"DEBUG main.py: Dictionary to save for {v_obj.uuid}: {json.dumps(save_dict, indent=2)}")

                        update_result = vcon_collection.replace_one(
                            {"_id": original_bson_id}, 
                            save_dict 
                        )
                        print(f"DEBUG main.py: MongoDB replace_one result for UUID {v_obj.uuid} (BSON ID {original_bson_id}): Matched={update_result.matched_count}, Modified={update_result.modified_count}")

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
                        vcon_collection.update_one({"_id": vid}, {"$unset": {"being_processed_by": ""}})
                    except Exception as e:
                        print(f"Error unsetting being_processed_by for vCon {vid}: {e}")
                
                # For failed vcons, keep them reserved but log them
                failed_vcon_ids = [vid for vid in avail_ids if vid not in successful_vcon_ids]
                if failed_vcon_ids:
                    print(f"Keeping {len(failed_vcon_ids)} vCons reserved due to failed transcription updates: {failed_vcon_ids}")
                    # Release them anyway to avoid blocking indefinitely
                    # for vid in failed_vcon_ids:
                    #     try:
                    #         vcon_collection.update_one({"_id": vid}, {"$unset": {"being_processed_by": ""}})
                    #     except Exception as e:
                    #         print(f"Error unsetting being_processed_by for failed vCon {vid}: {e}")

        total_bytes_processed += batch_bytes
        processed_ids.update(avail_ids)
        batch_time = time.time() - start_time
        progress_pct = (total_bytes_processed / total_bytes_to_process) * 100 if total_bytes_to_process > 0 else 0
        print(f"Processed {len(avail_ids)} files, {progress_pct:.1f}% complete, max GPU memory usage: {max_gpu_memory_usage()/(1024**3):.2f} GB")
        
        # Monitor GPU memory usage after each batch
        #print_gpu_memory_usage()

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

def main(sftp_url):
    loaded_ai = AIModel()
    vcons_collection = get_mongo_collection()
    sftp = sftp_connect(sftp_url)
    collection = get_mongo_collection()

    # Clean up any leftover files from previous runs
    cleanup_cache_directory()

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
            time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribey main entry point")
    parser.add_argument("mode", choices=["head", "worker", "slurp"], help="Run as head or worker")
    parser.add_argument("--sftp_url", type=str, default=settings.sftp_url, help="Override SFTP URL (applies to both head and worker)")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode (applies to both head and worker)")
    args = parser.parse_args()

    settings.debug = args.debug

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