import gc
import logging
import threading
import time
from multiprocessing import Queue
from queue import Empty
from time import perf_counter
from typing import List

import cupy as cp
import numpy as np
import torch
from nemo.collections.asr.models import EncDecSpeakerLabelModel

import settings
import stats
from gpu import move_to_gpu_maybe, gpu_ram_free_bytes, gc_collect_maybe, we_have_a_gpu
from process import ShutdownException
from stats import with_blocking_time
from utils import suppress_output
from vcon_class import Vcon
from vcon_queue import VconQueue
from vcon_utils import batch_to_audio_data, is_english

def load():
    model_name = "langid_ambernet"
    # with suppress_output(should_suppress=True):
    with suppress_output(should_suppress=False):
        langid_model = EncDecSpeakerLabelModel.from_pretrained(model_name="langid_ambernet")
    langid_model = move_to_gpu_maybe(langid_model)
    langid_model.eval()
    return langid_model

def numpy_to_cupy_maybe(audio_data):
    """Convert numpy array to cupy array if GPU is available and data is not already on GPU"""
    if we_have_a_gpu() and isinstance(audio_data, np.ndarray):
        return cp.asarray(audio_data)
    return audio_data

def cupy_to_numpy_maybe(audio_data):
    """Convert cupy array to numpy array if needed"""
    if hasattr(audio_data, 'get'):  # cupy array
        return audio_data.get()
    return audio_data

def process_audio_batch_gpu(audio_data_batch):
    """Process audio data using GPU acceleration with CuPy - assumes data is already on GPU"""
    if not we_have_a_gpu():
        return audio_data_batch
    
    # Work directly with GPU data - no conversion needed since preprocess already put it on GPU
    gpu_audio_batch = []
    for audio_data in audio_data_batch:
        if hasattr(audio_data, 'get'):  # Already a cupy array
            # Ensure contiguous memory layout for optimal GPU performance
            gpu_audio_data = cp.ascontiguousarray(audio_data, dtype=cp.float32)
            gpu_audio_batch.append(gpu_audio_data)
        elif isinstance(audio_data, torch.Tensor) and audio_data.is_cuda:
            # Convert from PyTorch GPU tensor to CuPy array
            gpu_audio_data = cp.asarray(audio_data.detach().cpu().numpy())
            gpu_audio_data = cp.ascontiguousarray(gpu_audio_data, dtype=cp.float32)
            gpu_audio_batch.append(gpu_audio_data)
        elif isinstance(audio_data, np.ndarray):  
            # Fallback: convert numpy to cupy if somehow still on CPU
            gpu_audio_data = cp.asarray(audio_data, dtype=cp.float32)
            gpu_audio_data = cp.ascontiguousarray(gpu_audio_data)
            gpu_audio_batch.append(gpu_audio_data)
        else:
            # Keep as-is if unknown format
            gpu_audio_batch.append(audio_data)
    
    return gpu_audio_batch

def identify_languages(batch: List[Vcon], model):
    """
    Identify languages using NVIDIA AmberNet model with GPU acceleration.
    Returns vcons with standard language codes like ["en", "es", etc.]
    """
    language_map = None
    if hasattr(model, 'cfg'):
        if hasattr(model.cfg, 'train_ds') and hasattr(model.cfg.train_ds, 'labels'):
            language_map = model.cfg.train_ds.labels
        elif hasattr(model.cfg, 'validation_ds') and hasattr(model.cfg.validation_ds, 'labels'):
            language_map = model.cfg.validation_ds.labels
        elif hasattr(model.cfg, 'test_ds') and hasattr(model.cfg.test_ds, 'labels'):
            language_map = model.cfg.test_ds.labels
        elif hasattr(model.cfg, 'labels'):
            language_map = model.cfg.labels

    if language_map is None and hasattr(model, 'decoder') and hasattr(model.decoder, 'vocabulary'):
        language_map = model.decoder.vocabulary

    if language_map is None:
        raise AttributeError(
            "Could not find language map in model. Tried model.cfg.train_ds.labels, "
            "model.cfg.validation_ds.labels, model.cfg.test_ds.labels, model.cfg.labels, "
            "and model.decoder.vocabulary."
        )

    #print(f"Using language map with {len(language_map)} languages: {language_map[:10]}...")
    
    print(f"identifying languages for {len(batch)} vcons (gpu memory: {gpu_ram_free_bytes()})")
    audio_data_batch = batch_to_audio_data(batch)
    
    # Process audio data on GPU using CuPy
    gpu_audio_batch = process_audio_batch_gpu(audio_data_batch)
    
    results = []
    vcons = []
    
    # Process each audio sample with GPU acceleration
    for i, audio_data in enumerate(gpu_audio_batch):
        try:
            # Work directly with GPU data - avoid CPU transfer when possible
            audio_data_for_inference = audio_data
            
            # Use the model's built-in inference method instead of direct forward()
            # This is more appropriate for NeMo models
            with torch.no_grad():
                # Try different inference methods that NeMo models typically have
                if hasattr(model, 'infer'):
                    # Some NeMo models have an infer method
                    # Convert CuPy to PyTorch tensor directly on GPU if possible
                    if hasattr(audio_data, 'get'):  # cupy array
                        # Use CuPy's __dlpack__ interface for zero-copy transfer to PyTorch
                        try:
                            audio_tensor = torch.as_tensor(cp.asarray(audio_data), device='cuda')
                            result = model.infer([audio_tensor.cpu().numpy()])
                        except:
                            # Fallback to CPU conversion
                            audio_data_cpu = audio_data.get()
                            result = model.infer([audio_data_cpu])
                    else:
                        result = model.infer([audio_data_for_inference])
                    
                    if isinstance(result, list):
                        pred_label_idx = result[0] if len(result) > 0 else 0
                    else:
                        pred_label_idx = result
                        
                elif hasattr(model, 'predict'):
                    # Some have predict method
                    if hasattr(audio_data, 'get'):  # cupy array
                        try:
                            audio_tensor = torch.as_tensor(cp.asarray(audio_data), device='cuda')
                            result = model.predict([audio_tensor.cpu().numpy()])
                        except:
                            audio_data_cpu = audio_data.get()
                            result = model.predict([audio_data_cpu])
                    else:
                        result = model.predict([audio_data_for_inference])
                    pred_label_idx = result[0] if isinstance(result, list) else result
                    
                else:
                    #print(f"Warning: No infer or predict method found for model. Using manual forward pass.")
                    # Fallback to manual forward pass with proper tensor handling
                    if hasattr(audio_data, 'get'):  # cupy array
                        # Use direct GPU tensor creation from CuPy
                        try:
                            audio_tensor = torch.as_tensor(cp.asarray(audio_data), device='cuda').unsqueeze(0)
                        except:
                            # Fallback method
                            audio_data_cpu = audio_data.get()
                            audio_tensor = torch.from_numpy(audio_data_cpu).unsqueeze(0)
                            audio_tensor = move_to_gpu_maybe(audio_tensor)
                    else:
                        audio_tensor = torch.from_numpy(audio_data_for_inference).unsqueeze(0)
                        audio_tensor = move_to_gpu_maybe(audio_tensor)
                    
                    input_signal_length = torch.tensor([audio_tensor.shape[1]], dtype=torch.long)
                    input_signal_length = move_to_gpu_maybe(input_signal_length)
                    
                    logits = model(input_signal=audio_tensor, input_signal_length=input_signal_length)
                    
                    # Handle different possible output formats
                    if isinstance(logits, tuple):
                        logits = logits[0]  # Take first element if tuple
                    
                    pred_label_idx = logits.argmax().item()
            
            # Map index to language code
            if isinstance(pred_label_idx, (torch.Tensor, np.ndarray)):
                pred_label_idx = pred_label_idx.item()
            elif hasattr(pred_label_idx, 'get'):  # cupy array
                pred_label_idx = pred_label_idx.get().item()
            
            if pred_label_idx < len(language_map):
                predicted_lang = language_map[pred_label_idx]
                #print(f"Predicted language index {pred_label_idx} -> {predicted_lang}")
            else:
                print(f"Warning: Predicted index {pred_label_idx} out of range (max: {len(language_map)-1}), defaulting to 'en'")
                predicted_lang = 'en'
            if predicted_lang != 'es':
                predicted_lang = "en" # assume spanish is correctly identified-- all else assert english
            results.append(predicted_lang)
            
        except Exception as e:
            #print(f"Error processing audio sample: {e}")
            results.append('en')  # Default fallback
    
    # Assign languages to vcons (one language per vcon)
    for i, vcon_cur in enumerate(batch):
        if i < len(results):
            detected_lang = results[i]
            # Always return as list for consistency (e.g., ["en"])
            language_list = [detected_lang]
            #print(f"Detected language for vcon {i}: {language_list}")
            vcon_cur.set_languages(language_list)
            vcons.append(vcon_cur)  # Fixed: append the vcon, not the list
        else:
            #print(f"Warning: No result for vcon {i}, defaulting to ['en']")
            vcon_cur.set_languages(['en'])
            vcons.append(vcon_cur)
    
    # Clear GPU memory
    if we_have_a_gpu():
        del gpu_audio_batch
        torch.cuda.empty_cache()
    
    gc_collect_maybe()
    
    print(f"Processed {len(vcons)} vcons for language identification")
    return vcons

def is_batch_ready(batch : List[Vcon], batch_start : float, total_size : int):
    time_passed = perf_counter() - batch_start
    if time_passed > settings.lang_detect_batch_ready:
        return True
    if len(batch) > settings.lang_detect_batch_max_len:
        return True
    if total_size > settings.lang_detect_batch_max_size:
        return True
    return False

def collect_vcons(preprocessed_vcons_queue : VconQueue, target_vcon: Vcon | None, stats_queue: Queue):
    try:
        if not target_vcon:
            with with_blocking_time(stats_queue):
                target_vcon = preprocessed_vcons_queue.get(timeout=settings.lang_detect_batch_timeout_seconds)
    except Empty:
        return [target_vcon], None
    
    batch_start : float = perf_counter()
    total_size : int = 0
    batch : List = [target_vcon]
        
    while not is_batch_ready(batch, batch_start, total_size):
        try:
            with with_blocking_time(stats_queue):
                cur_vcon = preprocessed_vcons_queue.get(timeout=settings.lang_detect_batch_timeout_seconds)
            if cur_vcon.size == target_vcon.size:
                # Audio data is already on GPU from preprocessing
                batch.append(cur_vcon)
                total_size += cur_vcon.size
            else: 
                return batch, cur_vcon
        except TimeoutError:
            return batch, target_vcon
    return batch, target_vcon


def lang_detect(preprocessed_vcons_queue: VconQueue,
                lang_detected_en_vcons_queue: VconQueue,
                lang_detected_non_en_vcons_queue: VconQueue,
                stats_queue: Queue):
    stats.add(stats_queue, "start_time", time.time())
    model = load()
    target_vcon : Vcon | None = None
    vcons_bytes : int = 0
    vcons_count : int = 0
    vcons_duration : int = 0
    try:
        while True: # just run thread forever
            batch, target_vcon = collect_vcons(preprocessed_vcons_queue, target_vcon, stats_queue)
            lang_detected_vcons = identify_languages(batch, model)
            for vcon_cur in lang_detected_vcons:                
                vcons_count += 1
                vcons_bytes += vcon_cur.size
                vcons_duration += vcon_cur.duration
                stats.add(stats_queue, "vcons_count", vcons_count)
                stats.add(stats_queue, "vcons_bytes", vcons_bytes)
                stats.add(stats_queue, "vcons_duration", vcons_duration)
                if is_english(vcon_cur):
                    with with_blocking_time(stats_queue):
                        lang_detected_en_vcons_queue.put(vcon_cur)
                else:
                    with with_blocking_time(stats_queue):
                        lang_detected_non_en_vcons_queue.put(vcon_cur)        
    except ShutdownException:
        pass

def start_thread(preprocessed_vcons_queue, lang_detected_en_vcons_queue, lang_detected_non_en_vcons_queue, stats_queue):
    thread = threading.Thread(target=lang_detect, args=(preprocessed_vcons_queue, lang_detected_en_vcons_queue, lang_detected_non_en_vcons_queue, stats_queue))
    thread.start()
    return thread
