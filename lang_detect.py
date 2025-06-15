import gc
from typing import List
from vcon_class import Vcon
from time import perf_counter
from multiprocessing import Queue
from process import ShutdownException
from vcon_queue import VconQueue
import stats
import threading
import time
#import gpu
# import torch
from nemo.collections.asr.models import EncDecSpeakerLabelModel
import logging
#import process
# from utils import gpu_ram_free_bytes, gc_collect_maybe
#import vcon_utils
import settings
#from utils import move_to_gpu_maybe, suppress_output

def load():
    model_name = "langid_ambernet"
    # with suppress_output(should_suppress=True):
    langid_model = EncDecSpeakerLabelModel.from_pretrained(model_name="langid_ambernet")
    # langid_model = move_to_gpu_maybe(langid_model)
    langid_model.eval()
    return langid_model

# def identify_languages(all_vcons_batched, model):
#     """
#     Identify languages using NVIDIA AmberNet model.
#     Returns vcons with standard language codes like ["en", "es", etc.]
#     """
#     language_map = None
#     if hasattr(model, 'cfg'):
#         if hasattr(model.cfg, 'train_ds') and hasattr(model.cfg.train_ds, 'labels'):
#             language_map = model.cfg.train_ds.labels
#         elif hasattr(model.cfg, 'validation_ds') and hasattr(model.cfg.validation_ds, 'labels'):
#             language_map = model.cfg.validation_ds.labels
#         elif hasattr(model.cfg, 'test_ds') and hasattr(model.cfg.test_ds, 'labels'):
#             language_map = model.cfg.test_ds.labels
#         elif hasattr(model.cfg, 'labels'):
#             language_map = model.cfg.labels

#     if language_map is None and hasattr(model, 'decoder') and hasattr(model.decoder, 'vocabulary'):
#         language_map = model.decoder.vocabulary

#     if language_map is None:
#         raise AttributeError(
#             "Could not find language map in model. Tried model.cfg.train_ds.labels, "
#             "model.cfg.validation_ds.labels, model.cfg.test_ds.labels, model.cfg.labels, "
#             "and model.decoder.vocabulary."
#         )

#     #print(f"Using language map with {len(language_map)} languages: {language_map[:10]}...")
    
#     vcons = []
#     for vcon_batch in all_vcons_batched:
#         print(f"identifying languages for {len(vcon_batch)} vcons (gpu memory: {gpu_ram_free_bytes()})")
#         audio_data_batch = vcon_utils.batch_to_audio_data(vcon_batch)
#         results = []
        
#         # Process each audio sample
#         for audio_data in audio_data_batch:
#             try:
#                 # Ensure audio is numpy array and convert to proper format
#                 if isinstance(audio_data, torch.Tensor):
#                     audio_data = audio_data.cpu().numpy()
                
#                 # Ensure audio is 1D and float32
#                 audio_data = audio_data.squeeze()
#                 if audio_data.ndim != 1:
#                     # If still not 1D, take the first channel (assume mono)
#                     audio_data = audio_data[0]
#                 audio_data = audio_data.astype(np.float32)
                
#                 # Check if audio is long enough (minimum duration check)
#                 min_samples = 1024  # Minimum samples for reliable language detection
#                 if len(audio_data) < min_samples:
#                     #print(f"Audio too short ({len(audio_data)} samples), padding to {min_samples}")
#                     # Pad with zeros or repeat the audio
#                     padding_needed = min_samples - len(audio_data)
#                     audio_data = np.pad(audio_data, (0, padding_needed), mode='constant')
                
#                 # Use the model's built-in inference method instead of direct forward()
#                 # This is more appropriate for NeMo models
#                 with torch.no_grad():
#                     # Try different inference methods that NeMo models typically have
#                     if hasattr(model, 'infer'):
#                         # Some NeMo models have an infer method
#                         result = model.infer([audio_data])
#                         if isinstance(result, list):
#                             pred_label_idx = result[0] if len(result) > 0 else 0
#                         else:
#                             pred_label_idx = result
#                     elif hasattr(model, 'predict'):
#                         # Some have predict method
#                         result = model.predict([audio_data])
#                         pred_label_idx = result[0] if isinstance(result, list) else result
#                     else:
#                         #print(f"Warning: No infer or predict method found for model. Using manual forward pass.")
#                         # Fallback to manual forward pass with proper tensor handling
#                         audio_tensor = torch.from_numpy(audio_data).unsqueeze(0).to(model.device)
#                         input_signal_length = torch.tensor([audio_tensor.shape[1]], dtype=torch.long, device=model.device)
                        
#                         logits = model(input_signal=audio_tensor, input_signal_length=input_signal_length)
                        
#                         # Handle different possible output formats
#                         if isinstance(logits, tuple):
#                             logits = logits[0]  # Take first element if tuple
                        
#                         pred_label_idx = logits.argmax().item()
                
#                 # Map index to language code
#                 if isinstance(pred_label_idx, (torch.Tensor, np.ndarray)):
#                     pred_label_idx = pred_label_idx.item()
                
#                 if pred_label_idx < len(language_map):
#                     predicted_lang = language_map[pred_label_idx]
#                     #print(f"Predicted language index {pred_label_idx} -> {predicted_lang}")
#                 else:
#                     print(f"Warning: Predicted index {pred_label_idx} out of range (max: {len(language_map)-1}), defaulting to 'en'")
#                     predicted_lang = 'en'
#                 if predicted_lang != 'es':
#                     predicted_lang = "en" # assume spanish is correctly identified-- all else assert english
#                 results.append(predicted_lang)
                
#             except Exception as e:
#                 #print(f"Error processing audio sample: {e}")
#                 results.append('en')  # Default fallback
        
#         # Assign languages to vcons (one language per vcon)
#         for i, vcon_cur in enumerate(vcon_batch):
#             if i < len(results):
#                 detected_lang = results[i]
#                 # Always return as list for consistency (e.g., ["en"])
#                 language_list = [detected_lang]
#                 #print(f"Detected language for vcon {i}: {language_list}")
#                 vcon_utils.set_languages(vcon_cur, language_list)
#                 vcons.append(vcon_cur)  # Fixed: append the vcon, not the list
#             else:
#                 #print(f"Warning: No result for vcon {i}, defaulting to ['en']")
#                 vcon_utils.set_languages(vcon_cur, ['en'])
#                 vcons.append(vcon_cur)
        
#         gc_collect_maybe()
    
#     print(f"Processed {len(vcons)} vcons for language identification")
#     return vcons

def is_batch_ready(batch : List[Vcon], batch_start : float):
    time_passed = perf_counter() - batch_start
    if time_passed > settings.lang_detect_batch_ready

def collect_vcons(preprocessed_vcons_queue : VconQueue):
    batch_start : float = perf_counter()
    target_vcon : Vcon = preprocessed_vcons_queue.get()
    batch : List = [target_vcon]
    while !is_batch_ready(batch, batch_start):

def lang_detect(preprocessed_vcons_queue: VconQueue,
                lang_detected_en_vcons_queue: VconQueue,
                lang_detected_non_en_vcons_queue: VconQueue,
                stats_queue: Queue):
    stats.add(stats_queue, "start_time", time.time())
    model = load()
    target_vcon : Vcon = preprocessed_vcons_queue.get()
    colected_vcons = [target_vcon]
    batch_started_collecting = perf_counter()
    try:
        while True: # just run thread forever
            vcons = collect_vcons(preprocessed_vcons_queue, stats_queue)
            lang_detected_en_vcons, lang_detected_non_en_vcons = identify_languages(vcons, model)

            for vcon_cur in lang_detected_en_vcons:
                lang_detected_en_vcons_queue.put(vcon_cur)

            for vcon_cur in lang_detected_en_vcons:
                lang_detected_non_en_vcons_queue.put(vcon_cur)
    except ShutdownException:
        pass

def start_thread(preprocessed_vcons_queue, lang_detected_en_vcons_queue, lang_detected_non_en_vcons_queue, stats_queue):
    thread = threading.Thread(target=lang_detect, args=(preprocessed_vcons_queue, lang_detected_en_vcons_queue, lang_detected_non_en_vcons_queue, stats_queue))
    thread.start()
    return thread
