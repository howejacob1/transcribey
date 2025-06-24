import datetime
import logging
import mimetypes
import os
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import Event, Process
from pprint import pprint
from typing import List
import binpacking
import paramiko
import torchaudio
from pymongo import MongoClient, ReplaceOne
from vcon import Vcon as VconBase
from vcon.dialog import Dialog
from vcon.party import Party
import audio
import cache
import gpu
import secrets_utils
import settings
import sftp
import sftp as sftp_utils
from mongo_utils import db
from process import block_until_threads_and_processes_finish
from settings import hostname
from sftp import parse_url
from stats import with_blocking_time
from utils import extension, suppress_output, is_audio_filename
from vcon_class import Vcon


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

def downloading_filename(vcon):
    vcon_filename = vcon.filename
    audio_extension = extension(vcon_filename)
    return cache.filename_to_downloading_filename(vcon.uuid + "." + audio_extension)

def cache_audio(vcon: Vcon, sftp: paramiko.SFTPClient):
    source_filename = vcon.filename
    dest_filename = downloading_filename(vcon)
    sftp_utils.download(source_filename, dest_filename, sftp)

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
    vcon_filename = vcon.filename
    audio_extension = extension(vcon_filename)
    return cache.filename_to_processing_filename(vcon.uuid + "." + audio_extension)

def mark_vcon_as_invalid(vcon: Vcon):
    db.update_one({"_id": vcon["uuid"]}, {"$set": {"corrupt": True, "done": True}})

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
    vcon_dict = vcon.to_dict()
    # Use UUID as _id for efficient indexing
    if "uuid" in vcon_dict:
        vcon_dict["_id"] = vcon_dict["uuid"]
    db.insert_one(vcon_dict)

def insert_many(vcons: List[Vcon]):
    dicts = []
    for vcon in vcons:
        vcon_dict = vcon.to_dict()
        # Use UUID as _id for efficient indexing
        if "uuid" in vcon_dict:
            vcon_dict["_id"] = vcon_dict["uuid"]
        dicts.append(vcon_dict)
    db.insert_many(dicts)

def insert_many_maybe(vcons: List[Vcon] | None):
    if vcons:
        to_insert = []
        for vcon in vcons:
            filename = vcon.filename
            # Skip vcons without filenames to prevent duplicate key errors
            if filename is not None and not exists_by_filename(filename):
                to_insert.append(vcon)
        if to_insert:  # Only insert if there are items to insert
            insert_many(to_insert)

def insert_many_maybe_async(vcons: List[Vcon] | None):
    if vcons:
        thread = threading.Thread(target=insert_many_maybe, args=(vcons,), daemon=True)
        thread.start()
        return thread

def insert_maybe(vcon):
    """Insert a vcon if it doesn't already exist (accepts Vcon object or dict)"""
    if vcon:
        # Get filename whether it's a Vcon object or dict
        if hasattr(vcon, 'filename'):
            filename = vcon.filename
        else:
            # Handle dict case more safely
            try:
                filename = vcon["dialog"][0]["filename"] if vcon.get("dialog") and len(vcon["dialog"]) > 0 else None
            except (KeyError, IndexError, TypeError):
                filename = None
        # Skip vcons without filenames to prevent duplicate key errors
        if filename is not None and not exists_by_filename(filename):
            insert_one(vcon)  # insert_one now handles the conversion

def get_by_filename(filename):
    return db.find_one({"dialog.0.filename": filename})

def exists_by_filename(filename):
    return get_by_filename(filename) is not None

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
    # Use UUID as _id for efficient indexing
    vcon_dict["_id"] = vcon_uuid_val
    db.replace_one({"_id": vcon_uuid_val}, vcon_dict, upsert=True)

def load_all():
    return list(db.find())

def load_and_print_all():
    vcons = load_all()
    pprint(vcons)

def all_urls():
    return [vcon.filename for vcon in db.find({"dialog.0.filename": 1})]

def find_and_reserve() -> Vcon | None:
    dict = db.find_one_and_update(
        {"done": {"$ne": True}, "processed_by": {"$exists": False}},
        {"$set": {"processed_by": settings.hostname}},
        return_document=True
    )
    if dict:
        # Keep _id as uuid for consistency
        if "_id" in dict and "uuid" not in dict:
            dict["uuid"] = dict["_id"]
        vcon = Vcon.from_dict(dict)
        return vcon
    return None

def find_and_reserve_many(size_bytes: int) -> List[Vcon]:
    total_bytes = 0
    reserved = []
    while total_bytes < size_bytes:
        some_vcon = find_and_reserve()
        if not some_vcon:
            return reserved
        reserved.append(some_vcon)
        total_bytes += some_vcon.size
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
    filename = processing_filename(vcon)
    cache.move_filename_to_processing(filename)

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