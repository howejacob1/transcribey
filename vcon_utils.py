from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import pprint
import datetime
import binpacking
import cache
import mimetypes
import torchaudio
import logging
from audio import resample_audio
import audio
from gpu import device
from vcon import Vcon
from vcon.party import Party
from vcon.dialog import Dialog
import mongo_utils
import secrets_utils
import sftp as sftp_utils
from sftp import download, get_file_size, connect
from utils import suppress_output, wait_for_one_thread_to_finish, wait_for_all_threads_to_finish, hostname, extension
import settings
import threading
from mongo_utils import get_mongo_collection
from pymongo import MongoClient, ReplaceOne

def sample_rate(vcon):
    return vcon["sample_rate"]

def size(vcon):
    return vcon["size"]

def uuid(vcon):
    return vcon["uuid"]

def create(url, file_size=None):
    """ 
    Create a vCon for the given wav or flac file URL and return it as a dict.
    file_size: size in bytes of the audio file (optional)
    """
    with suppress_output():
        vcon_obj = Vcon.build_new()
        party = Party(name="Unknown", role="participant")
        vcon_obj.add_party(party)
        now = datetime.datetime.now(datetime.timezone.utc)
        mimetype, _ = mimetypes.guess_type(url)
        dialog = Dialog(
            type="audio",
            start=now.isoformat(),
            parties=[0],
            originator=0,
            mimetype=mimetype,
            filename=url,
            body=None,
            encoding=None
        )
        vcon_obj.add_dialog(dialog)
        
        # Convert to dict and add size if provided
        vcon = vcon_obj.to_dict()
        vcon["size"] = file_size
        vcon["dialogs"][0]["transcript"] = {}
        return vcon

def get_filename(vcon):
    return vcon["dialogs"][0]["filename"]

def get_collection_name():
    return secrets_utils.secrets.get('vcons_collection')

def get_collection():
    collection_name = get_collection_name()
    return mongo_utils.get_collection(collection_name)

def delete_all():
    collection = get_collection()
    result = collection.delete_many({})
    print(f"Deleted {result.deleted_count} documents from the collection.")

# Add this function to print all vcons in the collection
def load_and_print_all():
    collection = get_mongo_collection()
    pprint(collection.find())

def all_urls():
    collection = get_collection()
    return [get_filename(vcon) for vcon in collection.find({}, {"filename": 1})]

def find_and_reserve():
    collection = get_collection()
    result = collection.find_one_and_update(
        {"done": {"ne": True}},
        {"$set": {"processed_by": settings.hostname}},
        return_document=True  # Return the updated document
    )
    return result

def find_and_reserve_many(size_bytes):
    total_bytes = 0
    reserved = []
    while total_bytes < size_bytes:
        some_vcon = find_and_reserve()
        if not some_vcon:
            return reserved
        reserved.append(some_vcon)
        total_bytes += size(some_vcon)
    return reserved

def make_batches(vcons):
    return binpacking.to_constant_volume(vcons, settings.batch_bytes, key=size)

def batch_to_audio_data(batch):
    audio_data_list = []
    for vcon in batch:
        audio_data = audio_data(vcon)
        audio_data_list.append(audio_data)
    return audio_data_list

def set_languages(vcon, languages):
    vcon["dialogs"][0]["transcript"]["languages"] = languages
    return vcon

def set_transcript(vcon, transcript):
    vcon["dialogs"][0]["transcript"]["text"] = transcript
    return vcon

def transcript(vcon):
    return vcon["dialogs"][0]["transcript"]["text"]

def languages(vcon):
    return vcon["dialogs"][0]["transcript"]["languages"]

def resample_vcon_one(vcon):
    audio_data = audio_data(vcon)
    sample_rate = sample_rate(vcon)
    resampled_audio_data = audio.resample(audio_data, sample_rate)
    set_audio_data(vcon, resampled_audio_data)
    return vcon

def resample_many(vcons):
    # Use ThreadPoolExecutor to parallelize resampling
    futures = []
    resampled_vcons = []
    with ThreadPoolExecutor() as executor:
        # Submit all resampling tasks
        for vcon in vcons:
            futures.append(executor.submit(resample_vcon_one, vcon))
        # Collect results as they complete
        for future in as_completed(futures):
            vcon = future.result()
            resampled_vcons.append(vcon)
    return resampled_vcons

def downloading_filename(vcon):
    vcon_filename = get_filename(vcon)
    audio_extension = extension(vcon_filename)
    return cache.filename_to_downloading_filename(vcon.uuid + "." + audio_extension)

def cache_vcon_audio(vcon, sftp):
    source_filename = get_filename(vcon)
    dest_filename = downloading_filename(vcon)
    download(source_filename, dest_filename, sftp)

def cache_vcon_audio_many(vcons, sftp):
    with ThreadPoolExecutor(max_workers=settings.max_download_threads) as executor:
        for vcon in vcons:
            executor.submit(cache_vcon_audio, vcon, sftp)

def processing_filename(vcon):
    return cache.filename_to_processing_filename(vcon.uuid)

def mark_vcon_as_invalid(vcon):
    collection = get_collection()
    collection.update_one({"uuid": vcon.uuid}, {"$set": {"corrupt": True, "done": True}})
    
def remove_vcon_from_processing(vcon):
    os.remove(processing_filename(vcon))

def process_invalids(vcons):
    vcons_valid = []
    for vcon in vcons:
        if not audio.is_valid(vcon):
            mark_vcon_as_invalid(vcon)
            remove_vcon_from_processing(vcon)
        else:
            vcons_valid.append(vcon)
    return vcons_valid
    
def unmarked_all_reserved():
    collection = get_collection()
    collection.update_many({"processed_by": hostname(), "done": {"$ne": True}}, {"$unset": {"processed_by": ""}})

def insert_many_maybe(vcons):
    collection = get_collection()
    collection.insert_many(vcons)

def get_by_filename(filename):
    collection = get_collection()
    return collection.find_one({"dialogs.0.filename": filename})

def exists_by_filename(filename):
    # Find a vcon where the first dialog's filename matches the given filename
    return get_by_filename(filename) is not None

def discover(url):
    sftp = connect(url)
    vcons = []
    for filename in sftp_utils.get_all_filenames(url, sftp):
        if not exists_by_filename(filename):
            vcon = create(filename, sftp.file_size(filename, sftp))
            vcons.append(vcon)
    insert_many_maybe(vcons)

def start_discover(url):
    discover_thread = threading.Thread(target=discover, args=(url,), daemon=True)
    discover_thread.start()
    return discover_thread

def audio_data(vcon):
    return vcon["dialogs"][0]["body"]

def set_audio_data(vcon, audio_data):
    vcon["dialogs"][0]["body"] = audio_data

def load_processing_into_ram(vcons):
    for vcon in vcons:
        filename = processing_filename(vcon)
        audio_data, sample_rate = audio.load_to_cpu(filename)
        set_audio_data(vcon, audio_data)
        vcon["sample_rate"] = sample_rate
    return vcons

def is_mono(vcon):
    return audio.is_mono(audio_data(vcon))

def convert_to_mono_maybe(vcon):
    if not is_mono(vcon):
        audio_data = audio.convert_to_mono(audio_data(vcon))
        set_audio_data(vcon, audio_data)
    return vcon

def convert_to_mono_many(vcons):
    vcons_mono = []
    vcons_mono_futures = []
    with ThreadPoolExecutor(max_workers=audio.cpu_cores_for_mono_conversion()) as executor:
        for vcon in vcons:
            if not is_mono(vcon):
                vcons_mono_futures.append(executor.submit(convert_to_mono_maybe, vcon))
        for future in as_completed(vcons_mono_futures):
            vcons_mono.append(future.result())
    return vcons_mono

def apply_vad_many(vcons):
    vcons_vad = []
    vcons_vad_futures = []
    vad = torchaudio.transforms.Vad(sample_rate=settings.sample_rate, trigger_level=0.5)
    with ThreadPoolExecutor(max_workers=audio.cpu_cores_for_vad()) as executor:
        for vcon in vcons:
            vcons_vad_futures.append(executor.submit(vad, vcon))
        for future in as_completed(vcons_vad_futures):
            vcons_vad.append(future.result())
    return vcons_vad

def move_to_gpu_maybe(vcon):
    vcon_audio_data = audio_data(vcon)
    audio.move_to_gpu_maybe(vcon_audio_data)
    set_audio_data(vcon, vcon_audio_data)
    return vcon

def move_to_gpu_many(vcons):
    vcons_on_gpu = []
    for vcon in vcons:
        move_to_gpu_maybe(vcon)
        vcons_on_gpu.append(vcon)
    return vcons_on_gpu

def split_by_language(vcons):
    vcons_en = []
    vcons_non_en = []
    for vcon in vcons:
        if languages(vcon) == ["en"]:
            vcons_en.append(vcon)
        else:
            vcons_non_en.append(vcon)
    return vcons_en, vcons_non_en

def set_done(vcon):
    vcon["done"] = True
    return vcon

def mark_vcons_as_done(vcons):
    for vcon in vcons:
        set_done(vcon)

def update_vcons_on_db(vcons):
    collection = get_collection()
    operations = []
    for vcon in vcons:
        vcon_uuid = uuid(vcon)
        operations.append(ReplaceOne({"uuid": vcon_uuid}, 
                                     vcon,
                                     upsert=True))
    thread = threading.Thread(target=collection.bulk_write, args=(operations,))
    thread.start()
