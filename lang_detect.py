import gc
import logging
import multiprocessing
import threading
import time
from queue import Empty
from time import time
from typing import List

import numpy as np
import torch
import nemo.collections.asr as nemo_asr

import audio
import gpu
import process
import settings
import stats
import vcon_utils
from multiprocessing import Queue
from gpu import gpu_ram_free_bytes, move_to_gpu_maybe, we_have_a_gpu, gc_collect_maybe
from process import ShutdownException, setup_signal_handlers
from stats import with_blocking_time
from utils import let_other_threads_run, dump_thread_stacks, suppress_output
from vcon_class import Vcon
from vcon_queue import VconQueue
from vcon_utils import batch_to_audio_data, is_english
from nemo.collections.asr.models import EncDecSpeakerLabelModel

def load():
    model_name = "langid_ambernet"
    
    # Set NeMo logging to ERROR level to reduce verbosity
    nemo_logger = logging.getLogger('nemo_logger')
    original_level = nemo_logger.level
    nemo_logger.setLevel(logging.ERROR)
    
    # Also set other common NeMo-related loggers
    for logger_name in ['omegaconf', 'hydra', 'lightning', 'pytorch_lightning']:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    
    # NeMo models use refresh_cache parameter instead of force_download
    # Setting refresh_cache=False will use local cache if available
    print(f"Loading {model_name} (using local cache if available)...")
    langid_model = EncDecSpeakerLabelModel.from_pretrained(model_name=model_name, refresh_cache=False)
    
    # Restore original logging level
    nemo_logger.setLevel(original_level)
    for logger_name in ['omegaconf', 'hydra', 'lightning', 'pytorch_lightning']:
        logging.getLogger(logger_name).setLevel(logging.INFO)
        
    langid_model = move_to_gpu_maybe(langid_model)
    langid_model.eval()
    return langid_model

def numpy_to_gpu_maybe(audio_data):
    """Convert numpy array to GPU tensor if GPU is available"""
    if we_have_a_gpu() and isinstance(audio_data, np.ndarray):
        return torch.from_numpy(audio_data).cuda()
    elif isinstance(audio_data, np.ndarray):
        return torch.from_numpy(audio_data)
    return audio_data

def gpu_to_numpy_maybe(audio_data):
    """Convert GPU tensor to numpy array if needed"""
    if isinstance(audio_data, torch.Tensor):
        return audio_data.cpu().numpy()
    return audio_data

def process_audio_batch_gpu(audio_data_batch):
    """Process audio data using GPU acceleration with PyTorch"""
    if not we_have_a_gpu():
        return audio_data_batch
    
    # Work directly with GPU data - convert to PyTorch tensors
    gpu_audio_batch = []
    for audio_data in audio_data_batch:
        if isinstance(audio_data, torch.Tensor):
            # Already a tensor, ensure it's on GPU and contiguous
            if audio_data.is_cuda:
                gpu_audio_data = audio_data.contiguous().float()
            else:
                gpu_audio_data = audio_data.cuda().contiguous().float()
            gpu_audio_batch.append(gpu_audio_data)
        elif isinstance(audio_data, np.ndarray):  
            # Convert numpy to GPU tensor
            gpu_audio_data = torch.from_numpy(audio_data).cuda().contiguous().float()
            gpu_audio_batch.append(gpu_audio_data)
        else:
            # Keep as-is if unknown format
            gpu_audio_batch.append(audio_data)
    
    return gpu_audio_batch

def identify_languages(vcon_cur: Vcon, model):
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

    batch = [vcon_cur]
    audio_data_batch_unprocessed = batch_to_audio_data(batch)
    audio_data_batch = []
    for audio_data in audio_data_batch_unprocessed:
        audio_data = gpu_to_numpy_maybe(audio_data)
        audio_data_batch.append(audio_data)
    # Process audio data on GPU using PyTorch
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
                    # Convert tensor to numpy for inference
                    if isinstance(audio_data, torch.Tensor):
                        try:
                            result = model.infer([audio_data.cpu().numpy()])
                        except:
                            # Fallback to CPU conversion
                            audio_data_cpu = audio_data.cpu().numpy()
                            result = model.infer([audio_data_cpu])
                    else:
                        result = model.infer([audio_data_for_inference])
                    
                    if isinstance(result, list):
                        pred_label_idx = result[0] if len(result) > 0 else 0
                    else:
                        pred_label_idx = result
                        
                elif hasattr(model, 'predict'):
                    # Some have predict method
                    if isinstance(audio_data, torch.Tensor):
                        try:
                            result = model.predict([audio_data.cpu().numpy()])
                        except:
                            audio_data_cpu = audio_data.cpu().numpy()
                            result = model.predict([audio_data_cpu])
                    else:
                        result = model.predict([audio_data_for_inference])
                    pred_label_idx = result[0] if isinstance(result, list) else result
                    
                else:
                    # Fallback to manual forward pass with proper tensor handling
                    if isinstance(audio_data, torch.Tensor):
                        # Use GPU tensor directly
                        try:
                            audio_tensor = audio_data.unsqueeze(0)
                        except:
                            # Fallback method
                            audio_data_cpu = audio_data.cpu().numpy()
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
            
            if pred_label_idx < len(language_map):
                predicted_lang = language_map[pred_label_idx]
            else:
                print(f"Warning: Predicted index {pred_label_idx} out of range (max: {len(language_map)-1}), defaulting to 'en'")
                predicted_lang = 'en'
            if predicted_lang != 'es':
                predicted_lang = "en" # assume spanish is correctly identified-- all else assert english
            results.append(predicted_lang)
            
        except Exception as e:
            results.append('en')  # Default fallback
    
    # Assign languages to vcons (one language per vcon)
    for i, vcon_cur in enumerate(batch):
        if i < len(results):
            detected_lang = results[i]
            # Always return as list for consistency (e.g., ["en"])
            language_list = [detected_lang]
            vcon_cur.set_languages(language_list)
            vcons.append(vcon_cur)  # Fixed: append the vcon, not the list
        else:
            vcon_cur.set_languages(['en'])
            vcons.append(vcon_cur)
    
    # Clear GPU memory
    if we_have_a_gpu():
        del gpu_audio_batch
        torch.cuda.empty_cache()
    
    gc_collect_maybe()
    
    return vcons

def is_batch_ready(batch : List[Vcon], batch_start : float, total_size : int):
    time_passed = time() - batch_start
    if time_passed > settings.lang_detect_batch_ready:
        return True
    if len(batch) > settings.lang_detect_batch_max_len:
        return True
    if total_size > settings.lang_detect_batch_max_size:
        return True
    return False

def lang_detect(preprocessed_vcons_queue: Queue,
                lang_detected_en_vcons_queue: Queue,
                lang_detected_non_en_vcons_queue: Queue,
                stats_queue: Queue):
    stats.start(stats_queue)
    model = None
    vcons_in_memory = []
    try:
        setup_signal_handlers()
        
        model = load()
        
        vcon_cur : Vcon | None = None
        
        while True: # just run thread forever
            with with_blocking_time(stats_queue):
                vcon_cur = preprocessed_vcons_queue.get()
            vcons_in_memory.append(vcon_cur)
            
            vcon_cur = identify_languages(vcon_cur, model)[0]
            stats.count(stats_queue)
            stats.bytes(stats_queue, vcon_cur.size)
            stats.duration(stats_queue, vcon_cur.duration)
            if is_english(vcon_cur):
                with with_blocking_time(stats_queue):
                    lang_detected_en_vcons_queue.put(vcon_cur)
            else:
                with with_blocking_time(stats_queue):
                    lang_detected_non_en_vcons_queue.put(vcon_cur)
            
            # Remove from our tracking list once processed
            if vcon_cur in vcons_in_memory:
                vcons_in_memory.remove(vcon_cur)
                
    except ShutdownException:
        print("LANG_DETECT: Received shutdown signal, cleaning up...")
        # Clean up model
        if model is not None:
            gpu.cleanup_model(model, "lang_detect_model")
            model = None
        
        # Clean up vcons in memory
        if vcons_in_memory:
            for vcon in vcons_in_memory:
                if hasattr(vcon, 'audio') and vcon.audio is not None:
                    vcon.audio = None
            vcons_in_memory.clear()
        
        # Clean up GPU memory
        gpu.exit_cleanup()
        pass
    except Exception as e:
        print(f"LANG_DETECT: FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up on error
        if model is not None:
            gpu.cleanup_model(model, "lang_detect_model")
        
        if vcons_in_memory:
            for vcon in vcons_in_memory:
                if hasattr(vcon, 'audio') and vcon.audio is not None:
                    vcon.audio = None
            vcons_in_memory.clear()
        
        gpu.exit_cleanup()
    finally:
        stats.stop(stats_queue)
    

def start_process(preprocessed_vcons_queue: Queue,
                  lang_detected_en_vcons_queue: Queue,
                  lang_detected_non_en_vcons_queue: Queue,
                  stats_queue: Queue):
    """Spawn the language detection worker as a separate process (instead of a thread)."""
    return process.start_process(
        target=lang_detect,
        args=(preprocessed_vcons_queue,
              lang_detected_en_vcons_queue,
              lang_detected_non_en_vcons_queue,
              stats_queue)
    )
