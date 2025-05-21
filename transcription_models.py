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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_REGISTRY = {}

def load_nvidia_parakeet_tdt_06b_v2():
    """Load nvidia/parakeet-tdt-0.6b-v2 model using NVIDIA NeMo."""
    logger.info("Starting to load nvidia/parakeet-tdt-0.6b-v2 ...")
    start_time = time.time()
    nemo_asr = importlib.import_module("nemo.collections.asr")
    model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
    model.to("cuda")
    elapsed = time.time() - start_time
    logger.info(f"Finished loading nvidia/parakeet-tdt-0.6b-v2 in {elapsed:.2f} seconds.")
    return model

def load_nvidia_canary_1b_flash():
    """Load nvidia/canary-1b-flash model using NVIDIA NeMo."""
    logger.info("Starting to load nvidia/canary-1b-flash ...")
    start_time = time.time()
    nemo_asr = importlib.import_module("nemo.collections.asr")
    model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/canary-1b-flash")
    model.to("cuda")
    elapsed = time.time() - start_time
    logger.info(f"Finished loading nvidia/canary-1b-flash in {elapsed:.2f} seconds.")
    return model

def load_openai_whisper_tiny():
    """
    Load the OpenAI Whisper tiny model and processor.
    Returns (model, processor, device)
    """
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    import torch
    model_name = "openai/whisper-tiny"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model, processor, device

def load_model_by_name(model_name):
    if model_name == "nvidia/parakeet-tdt-0.6b-v2":
        return load_nvidia_parakeet_tdt_06b_v2()
    elif model_name == "nvidia/canary-1b-flash":
        return load_nvidia_canary_1b_flash()
    else:
        raise ValueError(f"Unknown model name: {model_name}")

if __name__ == "__main__":
    import sys
    import traceback