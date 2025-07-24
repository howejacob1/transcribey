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
from torch.multiprocessing import Queue
from gpu import gpu_ram_free_bytes, move_to_gpu_maybe, we_have_a_gpu, gc_collect_maybe
from process import ShutdownException, setup_signal_handlers
from stats import with_blocking_time
from utils import let_other_threads_run, dump_thread_stacks, suppress_output, flatten
from vcon_class import Vcon

from vcon_utils import batch_to_audio_data, is_english, mark_vcon_as_invalid, remove_vcon_from_processing
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
    """Convert GPU tensor to numpy if needed"""
    if isinstance(audio_data, torch.Tensor):
        if audio_data.is_cuda:
            audio_data = audio_data.cpu()
        audio_data = audio_data.numpy()
    return audio_data

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
    
    # Convert to numpy arrays for consistent processing and filter out None values
    audio_data_batch = []
    valid_indices = []  # Track which vcons have valid audio
    
    for i, audio_data in enumerate(audio_data_batch_unprocessed):
        if audio_data is None:
            print(f"Warning: vcon {i} has no audio data, defaulting to 'en'")
            continue
            
        if isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.detach().cpu().numpy()
        elif not isinstance(audio_data, np.ndarray):
            print(f"Warning: vcon {i} has invalid audio data type {type(audio_data)}, defaulting to 'en'")
            continue
            
        audio_data_batch.append(audio_data)
        valid_indices.append(i)
    
    # Clear the unprocessed batch to free memory immediately
    del audio_data_batch_unprocessed
    
    results = []
    
    # Process each audio sample individually to avoid large tensor accumulation
    for i, audio_data in enumerate(audio_data_batch):
        try:
            with torch.no_grad():
                # Create tensor on GPU only when needed
                if isinstance(audio_data, np.ndarray):
                    audio_tensor = torch.from_numpy(audio_data).float()
                    if we_have_a_gpu():
                        audio_tensor = audio_tensor.cuda()
                else:
                    audio_tensor = audio_data.float()
                    if we_have_a_gpu() and not audio_tensor.is_cuda:
                        audio_tensor = audio_tensor.cuda()
                
                # Add batch dimension
                audio_tensor = audio_tensor.unsqueeze(0)
                input_signal_length = torch.tensor([audio_tensor.shape[1]], dtype=torch.long)
                if we_have_a_gpu():
                    input_signal_length = input_signal_length.cuda()
                
                # Try different inference methods
                if hasattr(model, 'infer'):
                    # Use CPU numpy for inference to avoid GPU memory accumulation
                    audio_numpy = audio_tensor.cpu().numpy().squeeze(0)
                    result = model.infer([audio_numpy])
                    pred_label_idx = result[0] if isinstance(result, list) else result
                elif hasattr(model, 'predict'):
                    # Use CPU numpy for predict to avoid GPU memory accumulation
                    audio_numpy = audio_tensor.cpu().numpy().squeeze(0)
                    result = model.predict([audio_numpy])
                    pred_label_idx = result[0] if isinstance(result, list) else result
                else:
                    # Manual forward pass
                    logits = model(input_signal=audio_tensor, input_signal_length=input_signal_length)
                    
                    # Handle different possible output formats
                    if isinstance(logits, tuple):
                        logits = logits[0]  # Take first element if tuple
                    
                    # Get prediction
                    pred_label_idx = logits.argmax(dim=-1).cpu().item()
                
                # Clean up tensors immediately after use
                del audio_tensor, input_signal_length
                if 'logits' in locals():
                    del logits
                    
                results.append(pred_label_idx)
                
        except Exception as e:
            print(f"Error processing audio sample {i}: {e}")
            results.append(0)  # Default fallback
            
        # Force garbage collection and GPU cleanup every few samples
        if (i + 1) % 5 == 0:
            if we_have_a_gpu():
                torch.cuda.empty_cache()
            gc_collect_maybe()

    # Clean up batch data
    del audio_data_batch
    
    # Process results and assign languages to vcons
    vcons_with_languages = []
    result_idx = 0
    
    for i, vcon_cur in enumerate(vcon_batch):
        if i in valid_indices:
            # This vcon had valid audio and was processed
            if result_idx < len(results):
                pred_label_idx = results[result_idx]
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
                vcon_cur.languages = language_list
                result_idx += 1
            else:
                # Fallback if no result available
                vcon_cur.languages = ['en']
        else:
            # This vcon had no valid audio, default to English
            vcon_cur.languages = ['en']
            
        vcons_with_languages.append(vcon_cur)
    
    # Final cleanup
    if we_have_a_gpu():
        torch.cuda.empty_cache()
    gc_collect_maybe()
    
    return vcons_with_languages

def lang_detect(preprocessed_vcons_queue: Queue,
                lang_detected_en_vcons_queue: Queue,
                lang_detected_non_en_vcons_queue: Queue,
                stats_queue: Queue):
    # Set process title for identification in nvidia-smi and ps
    try:
        from setproctitle import setproctitle
        import os
        setproctitle("transcribey-lang_detect")
        print(f"[PID {os.getpid()}] Set process title to: transcribey-lang_detect")
    except ImportError:
        print("setproctitle not available for lang_detect process")
    
    stats.start(stats_queue)
    model = None
    vcons_in_memory = []
    batch_count = 0
    
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
            vcon_batch = flatten(vcon_batch)
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
                if is_english(vcon_cur) or settings.put_all_vcons_into_english_queue:
                    en_vcons.append(vcon_cur)
                else:
                    non_en_vcons.append(vcon_cur)

            # Put the English and non-English batches in their respective queues
            if en_vcons:
                with with_blocking_time(stats_queue):
                    lang_detected_en_vcons_queue.put(en_vcons)
            
            if non_en_vcons:
                if settings.mark_non_english_as_corrupt:
                    for vcon in non_en_vcons:
                        mark_vcon_as_invalid(vcon)
                else:
                    with with_blocking_time(stats_queue):
                        lang_detected_non_en_vcons_queue.put(non_en_vcons)
            
            # Remove from our tracking list once processed
            # NOTE: Do NOT clear audio data here - it's still needed by transcription!
            for vcon_cur in vcon_batch:
                if vcon_cur in vcons_in_memory:
                    vcons_in_memory.remove(vcon_cur)
            
            # Clear batch references
            del vcon_batch, vcons_with_languages, en_vcons, non_en_vcons
            
            # Increment batch counter and perform periodic cleanup
            batch_count += 1
            if batch_count % 10 == 0:  # Every 10 batches
                print(f"Lang detect processed {batch_count} batches, performing memory cleanup...")
                if we_have_a_gpu():
                    torch.cuda.empty_cache()
                gc_collect_maybe()
                
                # Print memory stats for monitoring
                if we_have_a_gpu():
                    memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                    memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024    # MB
                    print(f"GPU Memory - Allocated: {memory_allocated:.1f}MB, Reserved: {memory_reserved:.1f}MB")
                
    except ShutdownException:
        print("LANG_DETECT: Received shutdown signal, cleaning up...")
        # Clean up model
        if model is not None:
            gpu.cleanup_model(model, "lang_detect_model")
            model = None
        
        # Clear our tracking list but don't clear audio data from vcons
        # as they might still be in transcription queues
        if vcons_in_memory:
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
        
        # Clear our tracking list but don't clear audio data from vcons
        # as they might still be in transcription queues
        if vcons_in_memory:
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
