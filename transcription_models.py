# transcription_models.py
"""
This module provides functions to load and manage all ASR models used in the project.
"""

# Example placeholder for model loading functions
# Actual implementation will depend on the model frameworks/APIs used (e.g., HuggingFace, NVIDIA, Microsoft, etc.)

import importlib
from utils import suppress_output
import time
import logging
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torch

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {}

def load_nvidia_parakeet_tdt_06b_v2():
    """Load nvidia/parakeet-tdt-0.6b-v2 model using NVIDIA NeMo."""
    logger.info("Starting to load nvidia/parakeet-tdt-0.6b-v2 ...")
    start_time = time.time()
    try:
        with suppress_output():
            nemo_asr = importlib.import_module("nemo.collections.asr")
            model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
    except ImportError as e:
        raise ImportError("nemo_toolkit is required to load this model. Install with: pip install nemo_toolkit['all']") from e
    elapsed = time.time() - start_time
    logger.info(f"Finished loading nvidia/parakeet-tdt-0.6b-v2 in {elapsed:.2f} seconds.")
    return model


def load_nvidia_parakeet_tdt_ctc_110m():
    """Load nvidia/parakeet-tdt_ctc-110m model using NVIDIA NeMo."""
    logger.info("Starting to load nvidia/parakeet-tdt_ctc-110m ...")
    start_time = time.time()
    try:
        with suppress_output():
            nemo_asr = importlib.import_module("nemo.collections.asr")
            model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt_ctc-110m")
    except ImportError as e:
        raise ImportError("nemo_toolkit is required to load this model. Install with: pip install nemo_toolkit['all']") from e
    elapsed = time.time() - start_time
    logger.info(f"Finished loading nvidia/parakeet-tdt_ctc-110m in {elapsed:.2f} seconds.")
    return model


def load_nvidia_canary_1b_flash():
    """Load nvidia/canary-1b-flash model using NVIDIA NeMo."""
    logger.info("Starting to load nvidia/canary-1b-flash ...")
    start_time = time.time()
    try:
        with suppress_output():
            nemo_asr = importlib.import_module("nemo.collections.asr")
            model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/canary-1b-flash")
    except ImportError as e:
        raise ImportError("nemo_toolkit is required to load this model. Install with: pip install nemo_toolkit['all']") from e
    elapsed = time.time() - start_time
    logger.info(f"Finished loading nvidia/canary-1b-flash in {elapsed:.2f} seconds.")
    return model


def load_openai_whisper_large_v2():
    """Load openai/whisper-large-v2 model and processor from HuggingFace for direct wav analysis."""
    logger.info("Starting to load openai/whisper-large-v2 ...")
    start_time = time.time()
    try:
        with suppress_output():
            model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v2")
            processor = AutoProcessor.from_pretrained("openai/whisper-large-v2")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
    except ImportError as e:
        raise ImportError("transformers and torch are required to load this model. Install with: pip install transformers torch") from e
    elapsed = time.time() - start_time
    logger.info(f"Finished loading openai/whisper-large-v2 in {elapsed:.2f} seconds.")
    return model, processor, device

def load_openai_whisper_tiny():
    """Load openai/whisper-tiny model and processor from HuggingFace for direct wav analysis."""
    logger.info("Starting to load openai/whisper-tiny ...")
    start_time = time.time()
    try:
        with suppress_output():
            model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-tiny")
            processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
    except ImportError as e:
        raise ImportError("transformers and torch are required to load this model. Install with: pip install transformers torch") from e
    elapsed = time.time() - start_time
    logger.info(f"Finished loading openai/whisper-tiny in {elapsed:.2f} seconds.")
    return model, processor, device

if __name__ == "__main__":
    import sys
    import traceback