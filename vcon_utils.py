# This module should be imported as vcon.
import datetime
import logging
import mimetypes
import time
import os
from pprint import pprint
import threading
import binpacking
import torchaudio
from concurrent.futures import ThreadPoolExecutor, as_completed
from pymongo import MongoClient, ReplaceOne
import audio
import cache
import mongo_utils
import secrets_utils
import settings
from settings import hostname
import sftp as sftp_utils
from sftp import parse_url
import gpu
from utils import extension, suppress_output, wait_for_all_threads_to_finish, wait_for_one_thread_to_finish
from vcon import Vcon
from vcon.dialog import Dialog
from vcon.party import Party

def get_size(vcon):
    return vcon["size"]

def get_filename(vcon):
    return vcon["dialog"][0]["filename"]

def set_filename(vcon, filename):
    vcon["dialog"][0]["filename"] = filename

def get_audio(vcon):
    return vcon["dialog"][0]["body"]

def set_audio(vcon, audio):
    vcon["dialog"][0]["body"] = audio

def get_transcript_text(vcon):
    return vcon["dialog"][0]["transcript"]["text"]

def set_transcript_text(vcon, text):
    vcon["dialog"][0]["transcript"]["text"] = text

def get_languages(vcon):
    return vcon["dialog"][0]["transcript"]["languages"]

def set_languages(vcon, langs):
    vcon["dialog"][0]["transcript"]["languages"] = langs

def get_collection_name():
    return secrets_utils.secrets.get('mongo_db', {}).get('vcons_collection')

def get_collection():
    collection_name = get_collection_name()
    return mongo_utils.get_collection(collection_name)

def set_transcript(vcon, transcript):
    vcon["dialog"][0]["transcript"]["text"] = transcript
    return vcon

def set_transcript_dict(vcon, transcript_dict):
    vcon["dialog"][0]["transcript"] = transcript_dict
    return vcon

def set_done(vcon):
    vcon["done"] = True
    return vcon

def is_mono(vcon):
    return audio.is_mono(get_audio(vcon))

def convert_to_mono_maybe(vcon):
    if not is_mono(vcon):
        audio_data_val = audio.convert_to_mono(get_audio(vcon))
        set_audio(vcon, audio_data_val)
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

def batch_to_audio_data(batch):
    audio_data_list = []
    for vcon in batch:
        audio_data_val = get_audio(vcon)
        audio_data_list.append(audio_data_val)
    return audio_data_list

def make_batches(vcons):
    return binpacking.to_constant_volume(vcons, gpu.batch_bytes(), key=get_size)

def resample_vcon_one(vcon):
    audio_data_val = get_audio(vcon)
    sample_rate_val = vcon["sample_rate"]
    resampled_audio_data = audio.resample(audio_data_val, sample_rate_val)
    set_audio(vcon, resampled_audio_data)
    return vcon

def resample_many(vcons):
    futures = []
    resampled_vcons = []
    with ThreadPoolExecutor() as executor:
        for vcon in vcons:
            futures.append(executor.submit(resample_vcon_one, vcon))
        for future in as_completed(futures):
            vcon_val = future.result()
            resampled_vcons.append(vcon_val)
    return resampled_vcons

def downloading_filename(vcon):
    vcon_filename = get_filename(vcon)
    audio_extension = extension(vcon_filename)
    return cache.filename_to_downloading_filename(vcon["uuid"] + "." + audio_extension)

def cache_vcon_audio(vcon, sftp):
    source_filename = get_filename(vcon)
    dest_filename = downloading_filename(vcon)
    sftp_utils.download(source_filename, dest_filename, sftp)

def cache_vcon_audio_many(vcons, sftp):
    futures = []
    with ThreadPoolExecutor(max_workers=settings.max_download_threads) as executor:
        for vcon in vcons:
            futures.append(executor.submit(cache_vcon_audio, vcon, sftp))
        for future in as_completed(futures):
            future.result()

def processing_filename(vcon):
    return cache.filename_to_processing_filename(vcon["uuid"])

def mark_vcon_as_invalid(vcon):
    collection = get_collection()
    collection.update_one({"uuid": vcon["uuid"]}, {"$set": {"corrupt": True, "done": True}})

def remove_vcon_from_processing(vcon):
    os.remove(processing_filename(vcon))

def is_audio_valid(vcon):
    audio_data = get_audio(vcon)
    return audio.is_valid(audio_data)

def process_invalids(vcons):
    vcons_valid = []
    for vcon in vcons:
        if not is_audio_valid(vcon):
            vcon["done"] = True
            vcon["corrupt"] = True
            mark_vcon_as_invalid(vcon)
            remove_vcon_from_processing(vcon)
        else:
            vcons_valid.append(vcon)
    return vcons_valid

def unmarked_all_reserved():
    collection = get_collection()
    collection.update_many({"processed_by": settings.hostname, "done": {"$ne": True}}, {"$unset": {"processed_by": ""}})

def insert_many_maybe(vcons):
    collection = get_collection()
    collection.insert_many(vcons)

def get_by_filename(filename):
    collection = get_collection()
    return collection.find_one({"dialog.0.filename": filename})

def exists_by_filename(filename):
    return get_by_filename(filename) is not None

def create(url, file_size=None):
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
        vcon = vcon_obj.to_dict()
        vcon["size"] = file_size
        set_transcript_dict(vcon, {})
        return vcon

def discover(url):
    sftp = sftp_utils.connect(url)
    parsed = sftp_utils.parse_url(url)
    path = parsed["path"]
    print(parsed)
    vcons = []
    for filename in sftp_utils.get_all_filenames(path, sftp):
        if not exists_by_filename(filename):
            vcon = create(filename, sftp_utils.file_size(filename, sftp))
            vcons.append(vcon)
    insert_many_maybe(vcons)

def start_discover(url):
    discover_thread = threading.Thread(target=discover, args=(url,), daemon=True)
    discover_thread.start()
    return discover_thread

def load_processing_into_ram(vcons):
    for vcon in vcons:
        filename = processing_filename(vcon)
        audio_data_val, sample_rate_val = audio.load_to_cpu(filename)
        set_audio(vcon, audio_data_val)
        vcon["sample_rate"] = sample_rate_val
    return vcons

def apply_vad_many(vcons):
    vcons_vad = []
    vcons_vad_futures = []
    vad = torchaudio.transforms.Vad(sample_rate=settings.sample_rate, trigger_level=0.5)
    with ThreadPoolExecutor(max_workers=audio.cpu_cores_for_vad()) as executor:
        for vcon in vcons:
            audio = get_audio(vcon)
            vcons_vad_futures.append(executor.submit(vad, audio))
        for future in as_completed(vcons_vad_futures):
            vcons_vad.append(future.result())
    return vcons_vad

def move_to_gpu_maybe(vcon):
    vcon_audio_data = get_audio(vcon)
    vcon_audio_data = gpu.move_to_gpu_maybe(vcon_audio_data)
    set_audio(vcon, vcon_audio_data)
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
        if get_languages(vcon) == ["en"]:
            vcons_en.append(vcon)
        else:
            vcons_non_en.append(vcon)
    return vcons_en, vcons_non_en

def mark_vcons_as_done(vcons):
    for vcon in vcons:
        set_done(vcon)

def update_vcons_on_db(vcons):
    collection = get_collection()
    operations = []
    for vcon in vcons:
        vcon_uuid_val = vcon["uuid"]
        operations.append(ReplaceOne({"uuid": vcon_uuid_val}, 
                                     vcon,
                                     upsert=True))
    thread = threading.Thread(target=collection.bulk_write, args=(operations,))
    thread.start()

def delete_all():
    collection = get_collection()
    result = collection.delete_many({})
    print(f"Deleted {result.deleted_count} documents from the collection.")

def load_and_print_all():
    collection = get_collection()
    pprint(collection.find())

def all_urls():
    collection = get_collection()
    return [get_filename(vcon) for vcon in collection.find({"dialog.0.filename": 1})]

def find_and_reserve():
    collection = get_collection()
    result = collection.find_one_and_update(
        {"done": {"$ne": True}},
        {"$set": {"processed_by": settings.hostname}},
        return_document=True
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
        total_bytes += get_size(some_vcon)
    return reserved
