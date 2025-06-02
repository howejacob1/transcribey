import paramiko
import settings
from transcription_models import AIModel
from mongo_utils import get_mongo_collection
import settings
import make_vcons_from_sftp
from utils import get_hostname
from wavs import is_wav_filename
import os
import shutil
import threading
import time
from sftp_utils import download_sftp_file, sftp_connect, get_sftp_file_size, parse_sftp_url
import argparse

def reserve_vcons_for_lang_detect(vcons_collection):
    hostname = get_hostname()
    max_total_size_bytes = 2 * (1024**3)
    # Find vCons needing language detection and not being processed
    query = {
        "being_processed_by": None,
        "analysis": {"$not": {"$elemMatch": {"type": "language_identification"}}},
        "attachments": {"$elemMatch": {"type": "audio"}},
        "analysis.type": {"$ne": "corrupt"}
    }
    vcons = list(vcons_collection.find(query))
    # Reserve up to max_total_size_bytes worth of vcons
    reserved = []
    total_size = 0
    for vcon in vcons:
        size = vcon.get("size", 0)
        if total_size + size > max_total_size_bytes:
            break
        result = vcons_collection.update_one({"_id": vcon["_id"], "being_processed_by": None}, {"$set": {"being_processed_by": hostname}})
        if result.modified_count == 1:
            reserved.append(vcon)
            total_size += size
    return reserved

def reserve_vcons_for_en_transcription(vcons_collection, max_total_size_gb):
    hostname = get_hostname()
    max_total_size_bytes = max_total_size_gb * (1024**3)
    # Find vCons with language_identification == ["en"] and no transcription, not being processed
    query = {
        "being_processed_by": None,
        "analysis": {"$elemMatch": {"type": "language_identification", "body": ["en"]}},
        "analysis.type": {"$ne": "transcription"},
        "attachments": {"$elemMatch": {"type": "audio"}},
        "analysis.type": {"$ne": "corrupt"}
    }
    vcons = list(vcons_collection.find(query))
    reserved = []
    total_size = 0
    for vcon in vcons:
        size = vcon.get("size", 0)
        if total_size + size > max_total_size_bytes:
            break
        result = vcons_collection.update_one({"_id": vcon["_id"], "being_processed_by": None}, {"$set": {"being_processed_by": hostname}})
        if result.modified_count == 1:
            reserved.append(vcon)
            total_size += size
    return reserved

def reserve_vcons_for_non_en_transcription(vcons_collection, max_total_size_gb):
    hostname = get_hostname()
    max_total_size_bytes = max_total_size_gb * (1024**3)
    # Find vCons with language_identification != ["en"] and no transcription, not being processed
    query = {
        "being_processed_by": None,
        "analysis": {"$elemMatch": {"type": "language_identification"}},
        "analysis.type": {"$ne": "transcription"},
        "attachments": {"$elemMatch": {"type": "audio"}},
        "analysis.type": {"$ne": "corrupt"}
    }
    vcons = list(vcons_collection.find(query))
    reserved = []
    total_size = 0
    for vcon in vcons:
        # Only reserve if not all langs are 'en'
        lang_analysis = next((a for a in vcon.get('analysis', []) if a.get('type') == 'language_identification'), None)
        if lang_analysis and (not all(l == 'en' for l in lang_analysis.get('body', []))):
            size = vcon.get("size", 0)
            if total_size + size > max_total_size_bytes:
                break
            result = vcons_collection.update_one({"_id": vcon["_id"], "being_processed_by": None}, {"$set": {"being_processed_by": hostname}})
            if result.modified_count == 1:
                reserved.append(vcon)
                total_size += size
    return reserved

def make_modes_to_try(model_mode):
    all_modes = ["lang_detect", "en", "non_en"]
    if model_mode:
        all_modes.remove(model_mode)
        return [model_mode] + all_modes
    else:
        return all_modes

def reserve_vcons_for_processing(model_mode, vcons_collection, max_total_size_gb):
    # Try to reserve for the current model mode
    modes_to_try = make_modes_to_try(model_mode)
    reserved = []
    for mode in modes_to_try:
        if mode == "lang_detect":
            reserved = reserve_vcons_for_lang_detect(vcons_collection)
            if reserved:
                return reserved, mode
        elif mode == "en":
            reserved = reserve_vcons_for_en_transcription(vcons_collection, max_total_size_gb)
            if reserved:
                return reserved, mode
        elif mode == "non_en":
            reserved = reserve_vcons_for_non_en_transcription(vcons_collection, max_total_size_gb)
            if reserved:
                return reserved, mode
    return [], None

def _download_vcons_to_cache(vcons_to_process, sftp):
    cache_dir = settings.dest_dir
    os.makedirs(cache_dir, exist_ok=True)
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
                except Exception as e:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    print(f"Failed to download {wav_url} to cache: {e}")

def cleanup_cache_directory():
    """Clean up any leftover files in the cache directory from previous runs."""
    cache_dir = settings.dest_dir
    if not os.path.exists(cache_dir):
        return
    
    files_removed = 0
    total_size_removed = 0
    
    for filename in os.listdir(cache_dir):
        file_path = os.path.join(cache_dir, filename)
        if os.path.isfile(file_path):
            try:
                file_size = os.path.getsize(file_path)
                os.remove(file_path)
                files_removed += 1
                total_size_removed += file_size
            except Exception as e:
                print(f"Failed to clean up {file_path}: {e}")
    
    if files_removed > 0:
        print(f"Cleaned up {files_removed} leftover files ({total_size_removed / (1024*1024):.1f} MB) from cache directory")

def load_vcons_in_background(vcons_to_process, sftp):
    """
    Start a background thread to download all wav files referenced in the vcons' attachments into the cache folder.
    """
    thread = threading.Thread(target=_download_vcons_to_cache, args=(vcons_to_process, sftp), daemon=True)
    thread.start()
    return thread

def process_vcons(download_thread, vcons_to_process, loaded_ai, mode):
    # Map vcon_id to cache wav path
    cache_dir = settings.dest_dir
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

    # Choose batch size
    if mode == 'lang_detect':
        batch_size = settings.lang_detect_batch_size
    elif mode == 'en':
        batch_size = settings.en_transcription_batch_size
    elif mode == 'non_en':
        batch_size = settings.non_en_transcription_batch_size
    else:
        print(f"Unknown mode: {mode}")
        return

    vcon_collection = get_mongo_collection()
    loaded_ai.load_by_mode(mode)

    # List of vcon_ids to process
    vcon_ids = list(vcon_id_to_cache_path.keys())
    processed_ids = set()
    
    # Progress tracking
    total_bytes_to_process = sum(vcon_id_to_vcon[vid].get('size', 0) for vid in vcon_ids)
    total_bytes_processed = 0
    start_time = time.time()
    batch_count = 0

    def available_vcon_ids():
        return [vid for vid in vcon_ids if os.path.exists(vcon_id_to_cache_path[vid])]

    print(f"Starting {mode} processing for {len(vcon_ids)} vcons ({total_bytes_to_process / (1024*1024):.1f} MB total)")

    while True:
        # Get available files for this batch
        avail_ids = [vid for vid in available_vcon_ids() if vid not in processed_ids]
        if not avail_ids:
            if not download_thread.is_alive():
                break  # Loader is done, nothing left
            time.sleep(1)
            continue
        batch_ids = avail_ids[:batch_size]
        batch_files = [vcon_id_to_cache_path[vid] for vid in batch_ids]
        batch_vcons = [vcon_id_to_vcon[vid] for vid in batch_ids]
        
        # Calculate batch size in bytes
        batch_bytes = sum(vcon_id_to_vcon[vid].get('size', 0) for vid in batch_ids)
        batch_count += 1
        
        print(f"Processing batch {batch_count} with {len(batch_ids)} files ({batch_bytes / (1024*1024):.1f} MB)...")
        batch_start_time = time.time()

        if mode == 'lang_detect':
            # Run language identification
            results = loaded_ai.identify_languages(batch_files, vcon_ids=batch_ids, vcon_collection=vcon_collection)
            for vcon, langs in zip(batch_vcons, results):
                analysis = {
                    "type": "language_identification",
                    "dialog": [0],
                    "vendor": "whisper-tiny",
                    "body": langs,
                    "encoding": "none"
                }
                try:
                    vcon_collection.update_one({"_id": vcon["_id"]}, {"$push": {"analysis": analysis}})
                except Exception as e:
                    print(f"Error updating language identification for vCon {vcon['_id']}: {e}")
                # Remove being_processed_by after processing
                try:
                    vcon_collection.update_one({"_id": vcon["_id"]}, {"$unset": {"being_processed_by": ""}})
                except Exception as e:
                    print(f"Error unsetting being_processed_by for vCon {vcon['_id']}: {e}")
        else:
            # Transcription (en or non_en)
            try:
                transcriptions = loaded_ai.transcribe(batch_files, english_only=(mode == "en"))
                for vcon, transcription in zip(batch_vcons, transcriptions):
                    analysis = {
                        "type": "transcription",
                        "dialog": [0],
                        "vendor": loaded_ai.model_name if hasattr(loaded_ai, 'model_name') else "unknown",
                        "body": transcription,
                        "encoding": "none"
                    }
                    try:
                        vcon_collection.update_one({"_id": vcon["_id"]}, {"$push": {"analysis": analysis}})
                    except Exception as e:
                        print(f"Error updating transcription for vCon {vcon['_id']}: {e}")
            except Exception as e:
                print(f"Error in transcription batch: {e}")
            finally:
                # Remove being_processed_by after processing, regardless of success or failure
                for vid in batch_ids:
                    try:
                        vcon_collection.update_one({"_id": vid}, {"$unset": {"being_processed_by": ""}})
                    except Exception as e:
                        print(f"Error unsetting being_processed_by for vCon {vid}: {e}")

        # Update progress tracking
        total_bytes_processed += batch_bytes
        batch_time = time.time() - batch_start_time
        total_time = time.time() - start_time
        
        # Calculate speeds
        batch_mb_per_sec = (batch_bytes / (1024*1024)) / batch_time if batch_time > 0 else 0
        overall_mb_per_sec = (total_bytes_processed / (1024*1024)) / total_time if total_time > 0 else 0
        progress_pct = (total_bytes_processed / total_bytes_to_process) * 100 if total_bytes_to_process > 0 else 0
        
        print(f"Batch completed in {batch_time:.1f}s ({batch_mb_per_sec:.1f} MB/s) | "
              f"Overall: {progress_pct:.1f}% ({total_bytes_processed/(1024*1024):.1f}/{total_bytes_to_process/(1024*1024):.1f} MB, {overall_mb_per_sec:.1f} MB/s)")

        # Delete processed wav files
        for file_path in batch_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
        processed_ids.update(batch_ids)

        # If loader thread is dead and all processed, break
        if not download_thread.is_alive() and len(processed_ids) == len(vcon_ids):
            break
    
    # Clean up any remaining files in cache directory at the end
    remaining_files = []
    for filename in os.listdir(cache_dir):
        file_path = os.path.join(cache_dir, filename)
        if os.path.isfile(file_path):
            remaining_files.append(file_path)
    
    if remaining_files:
        print(f"Cleaning up {len(remaining_files)} remaining files from cache...")
        for file_path in remaining_files:
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Failed to clean up remaining file {file_path}: {e}")
    
    # Final summary
    total_time = time.time() - start_time
    final_mb_per_sec = (total_bytes_processed / (1024*1024)) / total_time if total_time > 0 else 0
    print(f"\n{mode} processing completed: {total_bytes_processed/(1024*1024):.1f} MB in {total_time:.1f}s ({final_mb_per_sec:.1f} MB/s)")

def main():
    loaded_ai = AIModel()
    vcons_collection = get_mongo_collection()
    sftp = sftp_connect(settings.sftp_url)
    collection = get_mongo_collection()

    # Clean up any leftover files from previous runs
    cleanup_cache_directory()

    while True:
        print("Reserving vcons for processing...")
        vcons_to_process, mode = reserve_vcons_for_processing(loaded_ai.loaded_model_mode(),
                                                              vcons_collection,
                                                              settings.total_vcon_filesize_to_process_gb)
        print(f"Reserved {len(vcons_to_process)} vcons totaling {sum(vcon.get('size', 0) for vcon in vcons_to_process)/(1024*1024):.2f} MB for processing in mode {mode}")
        
        # If no vcons were reserved or mode is None, wait and try again
        if not vcons_to_process or mode is None:
            time.sleep(1)
            continue
            
        thread = load_vcons_in_background(vcons_to_process, sftp)
        process_vcons(thread, vcons_to_process, loaded_ai, mode)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribey main entry point")
    parser.add_argument("mode", choices=["head", "worker"], help="Run as head or worker")
    parser.add_argument("--sftp_url", type=str, default=None, help="Override SFTP URL (applies to both head and worker)")
    parser.add_argument("--debug", type=str, default=False, help="Override debug mode (debug=True or debug=False, applies to both head and worker)")
    args = parser.parse_args()

    # Allow overrides from command line (applies to both head and worker)
    if args.sftp_url and args.sftp_url.startswith("sftp://"):
        setattr(settings, "sftp_url", args.sftp_url)
    if args.debug is not None:
        if args.debug.lower() == "true" or args.debug.lower() == "debug=true":
            setattr(settings, "debug", True)
        elif args.debug.lower() == "false" or args.debug.lower() == "debug=false":
            setattr(settings, "debug", False)

    if args.mode == "head":
        from mongo_utils import delete_all_vcons
        if getattr(settings, "debug", False):
            print("[HEAD] Debug mode: deleting all vcons from the database...")
            delete_all_vcons()
        print("[HEAD] Starting make_vcons_from_sftp in a background thread...")
        sftp_thread = threading.Thread(target=make_vcons_from_sftp.main, daemon=True)
        sftp_thread.start()
        print("[HEAD] Running main loop in main thread...")
        main()
    elif args.mode == "worker":
        print("[WORKER] Running main loop...")
        main()