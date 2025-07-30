import datetime
import shutil
import json
import logging
import mimetypes
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from torch.multiprocessing import Event, Process
from pprint import pprint
from typing import List
import binpacking
import paramiko
import torchaudio
from pymongo import MongoClient, ReplaceOne
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError, NetworkTimeout
from vcon import Vcon as VconBase
from vcon.dialog import Dialog
from vcon.party import Party
import audio
import gpu
import secrets_utils
import settings
# SFTP imports removed - now using NFS
from mongo_utils import db
from process import block_until_threads_and_processes_finish
from settings import hostname
# parse_url import removed - now using NFS
from stats import with_blocking_time
from utils import extension, suppress_output, is_audio_filename
from vcon_class import Vcon

def _retry_db_operation(operation_func, max_retries=10, initial_delay=0.5):
    """Retry database operations with linear backoff"""
    operation_start = time.time()
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"🔄 DB_RETRY: Starting attempt {attempt + 1}/{max_retries}")
            
            op_start = time.time()
            result = operation_func()
            op_time = time.time() - op_start
            
            if op_time > 3.0:
                print(f"⚠️ SLOW_DB_OP: Database operation took {op_time:.1f}s")
            
            return result
            
        except (ConnectionFailure, ServerSelectionTimeoutError, NetworkTimeout) as e:
            print(f"❌ DB_CONNECTION_ERROR (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {e}")
            
            if attempt == max_retries - 1:
                total_time = time.time() - operation_start
                print(f"❌ DB_FAILED: All {max_retries} attempts failed after {total_time:.1f}s")
                raise e
                
            delay = initial_delay * (attempt + 1)  # Linear backoff: 0.5s, 1s, 1.5s, 2s, etc.
            print(f"⏳ DB_RETRY_WAIT: Retrying in {delay:.1f} seconds...")
            time.sleep(delay)
            
        except Exception as e:
            print(f"❌ DB_ERROR: Non-retryable error: {type(e).__name__}: {e}")
            # For other errors, don't retry
            raise e


def is_mono(vcon):
    audio_data = vcon.audio
    return audio.is_mono(audio_data)

def ensure_mono(vcon: Vcon):
    if not is_mono(vcon):
        audio_data = vcon.audio
        audio_data_mono = audio.convert_to_mono(audio_data)
        vcon.audio = audio_data_mono
    return vcon

# def convert_to_mono_many(vcons):
#     vcons_mono = []
#     vcons_mono_futures = []
#     with ThreadPoolExecutor(max_workers=audio.cpu_cores_for_mono_conversion()) as executor:
#         for vcon in vcons:
#             vcons_mono_futures.append(executor.submit(convert_to_mono_maybe, vcon))
#         for future in as_completed(vcons_mono_futures):
#             vcons_mono.append(future.result())
#     return vcons_mono

# def batch_to_audio_data(batch):
#     audio_data_list = []
#     for vcon in batch:
#         audio_data_val = get_audio(vcon)
#         audio_data_list.append(audio_data_val)
#     return audio_data_list

# def make_batches(vcons, batch_bytes):
#     return binpacking.to_constant_volume(vcons, batch_bytes, key=get_size)

def resample_vcon_one(vcon):
    audio_data_val = vcon.audio
    sample_rate_val = vcon["sample_rate"]
    resampled_audio_data = audio.resample_audio(audio_data_val, sample_rate_val)
    vcon.audio = resampled_audio_data
    return vcon

def resample_many(vcons):
    print(f"Resampling {len(vcons)} vcons.")
    futures = []
    resampled_vcons = []
    with ThreadPoolExecutor(max_workers=audio.cpu_cores_for_resampling()) as executor:
        for vcon in vcons:
            futures.append(executor.submit(resample_vcon_one, vcon))
        for future in as_completed(futures):
            vcon_val = future.result()
            resampled_vcons.append(vcon_val)
    print(f"Resampled vcons: {len(resampled_vcons)}")
    return resampled_vcons

# downloading_filename function removed - no longer needed with NFS

# cache_audio and cache_audio_batch functions removed - no longer needed with NFS

# def cache_vcon_audio_many(vcons, sftp):
#     total_count = len(vcons)
#     count = 0
#     for vcon in vcons:
#         cache_vcon_audio(vcon, sftp)
#         count += 1
#         if count % 50 == 0:
#             print(f"Cached {count/total_count*100:.2f}% of vcons")
#     print(f"Finished caching {total_count} vcons.")
#     return vcons

def processing_filename(vcon: Vcon):
    # For NFS, just return the original filename - no caching needed
    return vcon.filename

def mark_vcon_as_invalid(vcon: Vcon):
    db.update_one({"uuid": vcon.uuid}, {"$set": {"corrupt": True, "done": True}})
# 
def remove_vcon_from_processing(vcon: Vcon):
    os.remove(processing_filename(vcon))

def is_audio_valid(vcon):
    audio_data = vcon.audio
    return audio.is_valid(audio_data)

def process_invalids(vcons: List[Vcon]):
    vcons_valid = []
    for vcon in vcons:
        filename = processing_filename(vcon)
        if not audio.is_valid(filename):
            vcon["done"] = True
            vcon["corrupt"] = True
            mark_vcon_as_invalid(vcon)
            remove_vcon_from_processing(vcon)
        else:
            vcons_valid.append(vcon)
    print(f"Valid vcons: {len(vcons_valid)}")
    return vcons_valid

def unmarked_all_reserved():
    logging.info(f"Unmarking all reserved.")
    db.update_many({"processed_by": settings.hostname, "done": {"$ne": True}}, {"$unset": {"processed_by": ""}})

def insert_one(vcon: Vcon):
    def _insert():
        #print(f"📤 DB_RAW_INSERT: Starting MongoDB insert_one for vcon {vcon.uuid}...")
        vcon_dict = vcon.to_dict()
        # Remove _id to let MongoDB generate standard ObjectId
        if "_id" in vcon_dict:
            del vcon_dict["_id"]
        result = db.insert_one(vcon_dict)
        #print(f"✅ DB_RAW_INSERT_DONE: MongoDB insert_one completed for vcon {vcon.uuid}")
        return result
    
    #print(f"🔄 DB_INSERT_RETRY: Starting insert with retry logic for vcon {vcon.uuid}...")
    return _retry_db_operation(_insert)

def insert_many(vcons: List[Vcon]):
    dicts = []
    for vcon in vcons:
        vcon_dict = vcon.to_dict()
        # Remove _id to let MongoDB generate standard ObjectId
        if "_id" in vcon_dict:
            del vcon_dict["_id"]
        dicts.append(vcon_dict)
    
    # Use semaphore to limit concurrent database operations
    db.insert_many(dicts)

def insert_many_maybe(vcons: List[Vcon] | None, print_status=False):
    if not vcons:
        return
    
    # Bulk existence check for better performance
    basenames = [vcon.basename for vcon in vcons if vcon.basename is not None]
    if not basenames:
        return
    
    existence_map = exists_by_basenames_bulk(basenames)
    
    to_insert = []
    for vcon in vcons:
        basename = vcon.basename
        # Skip vcons without filenames to prevent duplicate key errors
        if basename is not None and not existence_map.get(basename, False):
            if print_status:
                print(f"Inserting {vcon.filename}")
            to_insert.append(vcon)
        else:
            if print_status:
                print(f"Skipping {vcon.filename} because it already exists")
    
    if to_insert:  # Only insert if there are items to insert
        insert_many(to_insert)

def insert_many_maybe_async(vcons: List[Vcon] | None, print_status=False):
    if vcons:
        thread = threading.Thread(target=insert_many_maybe, args=(vcons, print_status), daemon=True)
        thread.start()
        return thread

def get_by_basename(basename):
    def _get():
        #print(f"🔍 DB_RAW_QUERY: Starting MongoDB find_one for basename {basename}...")
        result = db.find_one({"basename": basename})
        #print(f"✅ DB_RAW_QUERY_DONE: MongoDB find_one completed for basename {basename}")
        if result and "_id" in result:
            del result["_id"]  # Remove ObjectId to avoid serialization issues
        return result
    
    #print(f"🔄 DB_QUERY_RETRY: Starting query with retry logic for basename {basename}...")
    return _retry_db_operation(_get)

def get_all_by_basename(basename):
    """Get ALL vcons with the given basename - used to detect duplicates"""
    def _get_all():
        results = list(db.find({"basename": basename}))
        # Remove ObjectId from all results to avoid serialization issues
        for result in results:
            if "_id" in result:
                del result["_id"]
        return results
    
    return _retry_db_operation(_get_all)

def exists_by_basename(basename):
    try:
        result = get_by_basename(basename)
        return result is not None
    except Exception as e:
        print(f"Error checking if basename {basename} exists: {e}")
        # On error, assume it doesn't exist to allow processing to continue
        return False

def exists_by_basenames_bulk(basenames: List[str]) -> dict:
    if not basenames:
        return {}
    
    try:
        existing_basenames = set()
        
        # Check top-level basename field
        cursor = db.find(
            {"basename": {"$in": basenames}},
            {"basename": 1, "_id": 0}
        )
        for doc in cursor:
            basename = doc.get("basename")
            if basename:
                existing_basenames.add(basename)
        
        # Return dict mapping basename -> exists
        return {basename: basename in existing_basenames for basename in basenames}
        
    except Exception as e:
        print(f"Database error in exists_by_basenames_bulk: {e}")
        # On error, assume none exist to be safe
        return {basename: False for basename in basenames}

def insert_maybe(vcon):
    """Insert a vcon if it doesn't already exist (accepts Vcon object or dict)"""
    exists = exists_by_basename(vcon.basename)
    if not exists:
        insert_one(vcon)


# def create(url):
#     vcon = None
#     with suppress_output(should_suppress=False):
#         vcon_obj = VconBase.build_new()
#         party = Party(name="Unknown", role="participant")
#         vcon_obj.add_party(party)
#         now = datetime.datetime.now(datetime.timezone.utc)
#         mimetype, _ = mimetypes.guess_type(url)
#         dialog = Dialog(
#             type="audio",
#             start=now.isoformat(),
#             parties=[0],
#             originator=0,
#             mimetype=mimetype,
#             filename=url,
#             body=None,
#             encoding=None
#         )
#         vcon_obj.add_dialog(dialog)
#         vcon = vcon_obj.to_dict()
#         set_transcript_dict(vcon, {})
#         print(vcon)
#     return vcon

# def load_processing_into_ram(vcons):
#     logging.info(f"Loading {len(vcons)} vcons into RAM.")
#     for vcon in vcons:
#         filename = processing_filename(vcon)
#         audio_data_val, sample_rate_val = audio.load_to_cpu(filename)
#         set_audio(vcon, audio_data_val)
#         vcon["sample_rate"] = sample_rate_val
#     return vcons

# def apply_vad_one(vcon, vad):
#     audio_data = get_audio(vcon)
#     vad_data = vad(audio_data)
#     set_audio(vcon, vad_data)
#     return vcon

# def apply_vad_many(vcons):
#     vcons_vad = []
#     vcons_vad_futures = []
#     logging.info(f"Applying VAD to {len(vcons)} vcons.")
#     vad = torchaudio.transforms.Vad(sample_rate=settings.sample_rate, trigger_level=0.5)
#     with ThreadPoolExecutor(max_workers=audio.cpu_cores_for_vad()) as executor:
#         for vcon in vcons:
#             audio_data = get_audio(vcon)
#             vcons_vad_futures.append(executor.submit(apply_vad_one, vcon, vad))
#         for future in as_completed(vcons_vad_futures):
#             vcons_vad.append(future.result())
#     return vcons_vad


def move_to_gpu_many(vcons):
    vcons_on_gpu = []
    for vcon in vcons:
        move_to_gpu_maybe(vcon)
        vcons_on_gpu.append(vcon)
    return vcons_on_gpu

# def split_by_language(vcons):
#     vcons_en = []
#     vcons_non_en = []
#     for vcon in vcons:
#         if get_languages(vcon) == ["en"]:
#             vcons_en.append(vcon)
#         else:
#             vcons_non_en.append(vcon)
#     return vcons_en, vcons_non_en

def mark_vcons_as_done(vcons):
    for vcon in vcons:
        vcon.done = True

def update_vcon_on_db(vcon: Vcon):
    vcon_uuid_val = vcon.uuid
    vcon_dict = vcon.to_dict()
    # Update by uuid field, not _id
    if "_id" in vcon_dict:
        del vcon_dict["_id"]
    db.replace_one({"uuid": vcon_uuid_val}, vcon_dict, upsert=True)

def update_vcons_on_db_bulk(vcons: List[Vcon]):
    if not vcons:
        return
    
    # Prepare bulk operations
    operations = []
    for vcon in vcons:
        vcon_dict = vcon.to_dict()
        if "_id" in vcon_dict:
            del vcon_dict["_id"]
        operations.append(
            ReplaceOne(
                {"uuid": vcon.uuid},
                vcon_dict,
                upsert=True
            )
        )
    
    # Execute bulk write with semaphore protection
    if operations:
        result = db.bulk_write(operations, ordered=False)
        return result

def load_all():
    results = list(db.find())
    # Remove ObjectId from all results to avoid serialization issues
    for result in results:
        if "_id" in result:
            del result["_id"]
    return results

def load_and_print_all():
    vcons = load_all()
    pprint(vcons)

def dump_jsonl():
    vcons = load_all()
    output_filename = "vcons.jsonl"
    with open(output_filename, "w") as f:
        for vcon in vcons:
            del vcon["_id"]
            if "processed_by" in vcon:
                del vcon["processed_by"]
            if "dialog" in vcon and len(vcon["dialog"]) > 0:
                if "size_bytes" in vcon["dialog"][0]:
                    del vcon["dialog"][0]["size_bytes"]
                if "sample_rate" in vcon["dialog"][0]:
                    del vcon["dialog"][0]["sample_rate"]

            f.write(json.dumps(vcon) + "\n")

def all_urls():
    return [vcon.filename for vcon in db.find({"dialog.0.filename": 1})]

def find_and_reserve() -> Vcon | None:
    dict = db.find_one_and_update(
        {"done": {"$ne": True}, "processed_by": {"$exists": False}},
        {"$set": {"processed_by": settings.hostname}},
        return_document=True,
        sort=[("created_at", 1)]  # Process oldest first for better cache locality
    )
    if dict:
        # Remove ObjectId to avoid serialization issues
        if "_id" in dict:
            del dict["_id"]
        vcon = Vcon.from_dict(dict)
        return vcon
    return None

def find_and_reserve_many(size_bytes: int) -> List[Vcon]:
    total_bytes = 0
    reserved = []
    
    cursor = db.find(
        {"done": {"$ne": True}, "processed_by": {"$exists": False}, "corrupt": {"$ne": True}},
        {"uuid": 1, "_id": 1},
        limit=settings.mongo_reservation_batch_limit)
    
    # Convert cursor to list to avoid multiple DB trips
    candidates = list(cursor)
    #print(f"Found {len(candidates)} candidates")
    # Build list of UUIDs to reserve
    uuids_to_reserve = []
    for doc in candidates:
        uuids_to_reserve.append(doc["uuid"])
    
    if not uuids_to_reserve:
        return []
    
    # Bulk reservation update
    #print(f"Reserving {len(uuids_to_reserve)} vcons")
    result = db.update_many(
        {"uuid": {"$in": uuids_to_reserve}, "processed_by": {"$exists": False}},
        {"$set": {"processed_by": settings.hostname}}
    )
    #print(f"Reserved {result.modified_count} vcons")
    if result.modified_count == 0:
        return []
    
    # Fetch the reserved documents
    reserved_docs = db.find({"uuid": {"$in": uuids_to_reserve}, "processed_by": settings.hostname})
    for doc in reserved_docs:
        # Remove ObjectId to avoid serialization issues
        if "_id" in doc:
            del doc["_id"]
        vcon = Vcon.from_dict(doc)
        reserved.append(vcon)
    print(f"Returning {len(reserved)} vcons")
    return reserved

def get_longest_duration(vcons: List[Vcon]) -> float:
    longest_duration = 0
    for vcon in vcons:
        audio_data = vcon.audio
        duration = audio.get_duration(vcon.filename)
        if duration > longest_duration:
            longest_duration = duration
    return longest_duration

def pad_vcon(vcon: Vcon, duration: float) -> Vcon:
    audio_data = vcon.audio
    audio_data = audio.pad_audio(audio_data, settings.sample_rate, duration)
    vcon.audio = audio_data
    return vcon

def pad_many(vcons: List[Vcon]) -> List[Vcon]:
    longest_duration = get_longest_duration(vcons)    
    vcons_padded = []
    for vcon in vcons:
        vcons_padded.append(pad_vcon(vcon, longest_duration))
    return vcons_padded

def print_audio_duration_many(vcons: List[Vcon]):
    for vcon in vcons:
        audio_data = vcon.audio
        print(f"duration: {audio_data.shape[1]}")
        print(f"channels: {audio_data.shape[0]}")

def unbatch(vcons_batched):
    vcons = []
    for vcon_batch in vcons_batched:
        vcons.extend(vcon_batch)
    return vcons

def delete_all():
    result = db.delete_many({})
    print(f"Deleted {result.deleted_count} documents from the collection.")

def move_to_gpu_maybe(self):
    audio_data = self.audio
    audio_data = gpu.move_to_gpu_maybe(audio_data)
    self.audio = audio_data
    return self 

def is_mono(vcon):
    if vcon.audio is None:
        return False
    return audio.is_mono(vcon.audio)

def move_to_processing(vcon: Vcon):
    # For NFS, no file movement needed - files are already accessible
    pass

def size_of_list(vcons : List[Vcon]) -> int:
    """Compute total size in bytes of all vcons in the list.

    Historically we attempted to read a non-existent ``bytes`` attribute which
    caused an ``AttributeError`` that killed the preprocess process.  Each
    ``Vcon`` already stores its size in the ``size`` property, so that is what
    we should aggregate here instead.
    """
    total = 0
    for vcon_cur in vcons:
        # ``size`` is explicitly defined as a property on ``Vcon``.  Use it.
        total += vcon_cur.size if vcon_cur.size is not None else 0
    return total

def batch_to_audio_data(batch):
    audio_data_list = []
    for vcon in batch:
        audio_data_val = vcon.audio
        audio_data_list.append(audio_data_val)
    return audio_data_list

def is_english(vcon: Vcon):
    languages = vcon.languages
    if languages:
        if len(languages) == 1:
            if languages[0] == 'en':
                return True
    return False

def remove_audio(vcon: Vcon):
    vcon.audio = None
    return vcon