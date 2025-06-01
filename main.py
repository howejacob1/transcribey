import paramiko
import settings
from transcription_models import AIModel, transcribe_vcons
from mongo_utils import get_mongo_collection
import settings
from utils import get_hostname, download_sftp_file, is_wav_filename, parse_sftp_url
import os
import shutil
import threading
import time

def reserve_vcons_for_lang_detect(vcons_collection, max_total_size_gb):
    hostname = get_hostname()
    max_total_size_bytes = max_total_size_gb * (1024**3)
    # Find vCons needing language detection and not being processed
    query = {
        "being_processed_by": None,
        "analysis": {"$not": {"$elemMatch": {"type": "language_identification"}}},
        "attachments": {"$elemMatch": {"type": "audio"}}
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
        "attachments": {"$elemMatch": {"type": "audio"}}
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
        "attachments": {"$elemMatch": {"type": "audio"}}
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
            reserved = reserve_vcons_for_lang_detect(vcons_collection, max_total_size_gb)
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

def load_vcons_in_background(vcons_to_process, sftp):
    """
    Start a background thread to download all wav files referenced in the vcons' attachments into the cache folder.
    """
    thread = threading.Thread(target=_download_vcons_to_cache, args=(vcons_to_process, sftp), daemon=True)
    thread.start()
    return thread

def process_vcons(thread, vcons_to_process, loaded_ai, mode):
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

    def available_vcon_ids():
        return [vid for vid in vcon_ids if os.path.exists(vcon_id_to_cache_path[vid])]

    while True:
        # Get available files for this batch
        avail_ids = [vid for vid in available_vcon_ids() if vid not in processed_ids]
        if not avail_ids:
            if not thread.is_alive():
                break  # Loader is done, nothing left
            time.sleep(1)
            continue
        batch_ids = avail_ids[:batch_size]
        batch_files = [vcon_id_to_cache_path[vid] for vid in batch_ids]
        batch_vcons = [vcon_id_to_vcon[vid] for vid in batch_ids]

        if mode == 'lang_detect':
            # Run language identification
            results = loaded_ai.identify_languages(batch_files)
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
        else:
            # Transcription (en or non_en)
            vcon_file_tuples = [(vid, vcon_id_to_cache_path[vid]) for vid in batch_ids]
            try:
                transcribe_vcons(vcon_collection, loaded_ai, vcon_file_tuples, batch_size)
            except Exception as e:
                print(f"Error in transcription batch: {e}")

        # Delete processed wav files
        for file_path in batch_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
        processed_ids.update(batch_ids)

        # If loader thread is dead and all processed, break
        if not thread.is_alive() and len(processed_ids) == len(vcon_ids):
            break

def main():
    loaded_ai = AIModel()
    print(loaded_ai.model)
    vcons_collection = get_mongo_collection()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    sftp_url_parsed = parse_sftp_url(settings.sftp_url)
    username = sftp_url_parsed["username"]
    hostname = sftp_url_parsed["hostname"]
    port = sftp_url_parsed["port"]
    path = sftp_url_parsed["path"]

    client.connect(hostname, port=port, username=username)
    sftp = client.open_sftp()
    collection = get_mongo_collection()

    while True:
        print("Reserving vcons for processing...")
        vcons_to_process, mode = reserve_vcons_for_processing(loaded_ai.loaded_model_mode(),
                                                              vcons_collection,
                                                              settings.total_vcon_filesize_to_process_gb)
        print(f"Reserved {len(vcons_to_process)} vcons totaling {sum(vcon.get('size', 0) for vcon in vcons_to_process)/(1024*1024*1024)} bytes for processing in mode {mode}")
        thread = load_vcons_in_background(vcons_to_process, sftp)
        process_vcons(thread, vcons_to_process, loaded_ai, mode)

if __name__ == "__main__":
    main()