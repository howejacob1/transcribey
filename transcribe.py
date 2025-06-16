import time
from multiprocessing import Queue
from typing import List
import logging

import torch
import whisper
from nemo.collections.asr.models import ASRModel

import gpu
import process
import settings
import stats
import vcon_utils
from process import ShutdownException
from process import setup_signal_handlers
from stats import with_blocking_time
from utils import suppress_output
from vcon_class import Vcon

def load_nvidia(model_name):
    """Load nvidia model using NVIDIA NeMo."""
    total_start_time = time.time()
    print(f"Loading model {model_name}.")
    try:
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
        model = ASRModel.from_pretrained(model_name=model_name, refresh_cache=False)
        
        # Restore original logging level
        nemo_logger.setLevel(original_level)
        for logger_name in ['omegaconf', 'hydra', 'lightning', 'pytorch_lightning']:
            logging.getLogger(logger_name).setLevel(logging.INFO)
            
        model.eval()
        print(f"Model {model_name} loaded in {time.time() - total_start_time:.2f} seconds total")
        return model
    except Exception as e:
        print(f"ERROR: Failed to load model {model_name}: {e}")
        raise

def transcribe(lang_detected_queue: Queue,
                transcribed_queue: Queue,
                stats_queue: Queue,
                model_name: str,
                language: str):
    print(f"TRANSCRIBE PROCESS STARTING: language={language}, model={model_name}")
    stats.start(stats_queue)
    model = None
    vcons_in_memory = []
    try:
        setup_signal_handlers()
        model = load_nvidia(model_name)
        
        config = {"batch_size": 32}
        if language != "en" and language != "auto":
            config.update({
                "source_lang": language,
                "target_lang": language, 
                "task": "asr",
                "pnc": "yes"
            })
        elif language == "auto":
            # For auto language detection, don't specify source/target lang
            config.update({
                "task": "asr",
                "pnc": "yes"
            })

        vcon_cur : Vcon | None = None
        
        while True: # just run thread forever
            with with_blocking_time(stats_queue):
                vcon_cur = lang_detected_queue.get()
            vcons_in_memory.append(vcon_cur)

            # Transcribe the batch
            with torch.no_grad():
                with suppress_output(should_suppress=True):
                    transcription = model.transcribe(vcon_cur.audio, **config)
            
            # Handle different types of transcription results from NeMo
            if isinstance(transcription, list) and len(transcription) > 0:
                # If it's a list, take the first result
                transcription_obj = transcription[0]
                if hasattr(transcription_obj, 'text'):
                    text = transcription_obj.text
                else:
                    text = str(transcription_obj)
            elif hasattr(transcription, 'text'):
                # Single object with text attribute (most common case)
                text = transcription.text
            elif isinstance(transcription, str):
                # Already a string
                text = transcription
            else:
                # Fallback - convert to string but this shouldn't happen
                text = str(transcription)
                
            vcon_cur.transcript_text = text

            # Clean up audio data after transcription - no longer needed and prevents JSON serialization errors
            if hasattr(vcon_cur, 'audio') and vcon_cur.audio is not None:
                vcon_cur.audio = None

            stats.count(stats_queue)
            stats.bytes(stats_queue, vcon_cur.size)
            stats.duration(stats_queue, vcon_cur.duration)
            with with_blocking_time(stats_queue):
                transcribed_queue.put(vcon_cur)
            
            # Remove from our tracking list once processed
            if vcon_cur in vcons_in_memory:
                vcons_in_memory.remove(vcon_cur)
            
    except ShutdownException:
        print(f"TRANSCRIBE {language}: Received shutdown signal, cleaning up...")
        # Clean up model
        if model is not None:
            gpu.cleanup_model(model, f"transcribe_model_{language}")
            model = None
        
        # Clean up vcons in memory
        if vcons_in_memory:
            for vcon in vcons_in_memory:
                if hasattr(vcon, 'audio') and vcon.audio is not None:
                    vcon.audio = None
            vcons_in_memory.clear()
        
        # Clean up GPU memory
        gpu.exit_cleanup()
        stats.stop(stats_queue)
        exit()
    except Exception as e:
        print(f"TRANSCRIBE {language}: FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up on error
        if model is not None:
            gpu.cleanup_model(model, f"transcribe_model_{language}")
        
        if vcons_in_memory:
            for vcon in vcons_in_memory:
                if hasattr(vcon, 'audio') and vcon.audio is not None:
                    vcon.audio = None
            vcons_in_memory.clear()
        
        gpu.exit_cleanup()
        stats.stop(stats_queue)
        exit(1)

def start_process(lang_detected_queue: Queue,
                 transcribed_queue: Queue,
                 stats_queue: Queue,
                 language: str):
    """Start transcription process for specified language"""
    if language == "en":
        model_name = settings.en_model_name
    else:
        model_name = settings.non_en_model_name
    return process.start_process(target=transcribe, args=(lang_detected_queue, transcribed_queue, stats_queue, model_name, language))
