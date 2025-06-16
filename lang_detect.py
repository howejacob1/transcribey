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

def identify_languages_batch(vcon_batch: List[Vcon], model):
    """
    Identify languages for a batch of vcons using NVIDIA AmberNet model with GPU acceleration.
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

    # Extract audio data from the batch
    audio_data_batch_unprocessed = batch_to_audio_data(vcon_batch)
    audio_data_batch = []
    for audio_data in audio_data_batch_unprocessed:
        audio_data = gpu_to_numpy_maybe(audio_data)
        audio_data_batch.append(audio_data)
    
    # Process audio data on GPU using PyTorch
    gpu_audio_batch = process_audio_batch_gpu(audio_data_batch)
    
    results = []
    
    # Process the entire batch at once for efficiency
    try:
        with torch.no_grad():
            # Prepare batch tensors for inference
            batch_tensors = []
            batch_lengths = []
            
            for audio_data in gpu_audio_batch:
                if isinstance(audio_data, torch.Tensor):
                    batch_tensors.append(audio_data)
                    batch_lengths.append(audio_data.shape[0])
                else:
                    tensor = torch.from_numpy(audio_data)
                    tensor = move_to_gpu_maybe(tensor)
                    batch_tensors.append(tensor)
                    batch_lengths.append(tensor.shape[0])
            
            # Pad tensors to same length for batch processing
            max_length = max(batch_lengths)
            padded_tensors = []
            for tensor in batch_tensors:
                if tensor.shape[0] < max_length:
                    padding = max_length - tensor.shape[0]
                    padded_tensor = torch.nn.functional.pad(tensor, (0, padding))
                else:
                    padded_tensor = tensor
                padded_tensors.append(padded_tensor)
            
            # Stack into batch tensor
            batch_tensor = torch.stack(padded_tensors)
            input_signal_length = torch.tensor(batch_lengths, dtype=torch.long)
            input_signal_length = move_to_gpu_maybe(input_signal_length)
            
            # Try different inference methods
            if hasattr(model, 'infer'):
                # Convert to numpy for inference
                batch_numpy = batch_tensor.cpu().numpy()
                batch_results = model.infer(batch_numpy.tolist())
                if not isinstance(batch_results, list):
                    batch_results = [batch_results]
                results = batch_results
            elif hasattr(model, 'predict'):
                batch_numpy = batch_tensor.cpu().numpy()
                batch_results = model.predict(batch_numpy.tolist())
                if not isinstance(batch_results, list):
                    batch_results = [batch_results]
                results = batch_results
            else:
                # Fallback to manual forward pass
                logits = model(input_signal=batch_tensor, input_signal_length=input_signal_length)
                
                # Handle different possible output formats
                if isinstance(logits, tuple):
                    logits = logits[0]  # Take first element if tuple
                
                # Get predictions for the batch
                pred_indices = logits.argmax(dim=-1).cpu().numpy()
                results = pred_indices.tolist()
            
    except Exception as e:
        print(f"Batch processing failed, falling back to individual processing: {e}")
        # Fallback to individual processing
        results = []
        for audio_data in gpu_audio_batch:
            try:
                if hasattr(model, 'infer'):
                    if isinstance(audio_data, torch.Tensor):
                        result = model.infer([audio_data.cpu().numpy()])
                    else:
                        result = model.infer([audio_data])
                    pred_label_idx = result[0] if isinstance(result, list) else result
                elif hasattr(model, 'predict'):
                    if isinstance(audio_data, torch.Tensor):
                        result = model.predict([audio_data.cpu().numpy()])
                    else:
                        result = model.predict([audio_data])
                    pred_label_idx = result[0] if isinstance(result, list) else result
                else:
                    audio_tensor = audio_data.unsqueeze(0) if isinstance(audio_data, torch.Tensor) else torch.from_numpy(audio_data).unsqueeze(0)
                    audio_tensor = move_to_gpu_maybe(audio_tensor)
                    input_signal_length = torch.tensor([audio_tensor.shape[1]], dtype=torch.long)
                    input_signal_length = move_to_gpu_maybe(input_signal_length)
                    logits = model(input_signal=audio_tensor, input_signal_length=input_signal_length)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    pred_label_idx = logits.argmax().item()
                
                results.append(pred_label_idx)
            except Exception as e:
                results.append(0)  # Default fallback
    
    # Process results and assign languages to vcons
    vcons_with_languages = []
    for i, vcon_cur in enumerate(vcon_batch):
        if i < len(results):
            pred_label_idx = results[i]
            if isinstance(pred_label_idx, (torch.Tensor, np.ndarray)):
                pred_label_idx = pred_label_idx.item()
            
            if pred_label_idx < len(language_map):
                predicted_lang = language_map[pred_label_idx]
            else:
                print(f"Warning: Predicted index {pred_label_idx} out of range (max: {len(language_map)-1}), defaulting to 'en'")
                predicted_lang = 'en'
                
            if predicted_lang != 'es':
                predicted_lang = "en" # assume spanish is correctly identified-- all else assert english
                
            language_list = [predicted_lang]
            vcon_cur.set_languages(language_list)
            vcons_with_languages.append(vcon_cur)
        else:
            vcon_cur.set_languages(['en'])
            vcons_with_languages.append(vcon_cur)
    
    # Clear GPU memory
    if we_have_a_gpu():
        del gpu_audio_batch
        torch.cuda.empty_cache()
    
    gc_collect_maybe()
    
    return vcons_with_languages

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
        
        while True: # just run thread forever
            # Get a batch from the preprocessing queue
            with with_blocking_time(stats_queue):
                vcon_batch = preprocessed_vcons_queue.get()
            
            if not isinstance(vcon_batch, list):
                # Handle backward compatibility in case single vcons are still sent
                vcon_batch = [vcon_batch]
            
            vcons_in_memory.extend(vcon_batch)
            
            # Process the entire batch for language detection
            vcons_with_languages = identify_languages_batch(vcon_batch, model)
            
            # Split the batch into English and non-English
            en_vcons = []
            non_en_vcons = []
            
            for vcon_cur in vcons_with_languages:
                stats.count(stats_queue)
                stats.bytes(stats_queue, vcon_cur.size)
                stats.duration(stats_queue, vcon_cur.duration)
                
                if is_english(vcon_cur):
                    en_vcons.append(vcon_cur)
                else:
                    non_en_vcons.append(vcon_cur)
            
            # Put the English and non-English batches in their respective queues
            if en_vcons:
                with with_blocking_time(stats_queue):
                    lang_detected_en_vcons_queue.put(en_vcons)
            
            if non_en_vcons:
                with with_blocking_time(stats_queue):
                    lang_detected_non_en_vcons_queue.put(non_en_vcons)
            
            # Remove from our tracking list once processed
            for vcon_cur in vcon_batch:
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
