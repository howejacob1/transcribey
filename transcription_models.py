# transcription_models.py
"""
This module provides functions to load and manage all ASR models used in the project.
"""

# Example placeholder for model loading functions
# Actual implementation will depend on the model frameworks/APIs used (e.g., HuggingFace, NVIDIA, Microsoft, etc.)

import importlib

MODEL_REGISTRY = {}

def load_nvidia_parakeet_tdt_06b_v2():
    """Load nvidia/parakeet-tdt-0.6b-v2 model using NVIDIA NeMo."""
    try:
        nemo_asr = importlib.import_module("nemo.collections.asr")
    except ImportError as e:
        raise ImportError("nemo_toolkit is required to load this model. Install with: pip install nemo_toolkit['all']") from e
    return nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")


def load_nvidia_parakeet_tdt_ctc_110m():
    """Load nvidia/parakeet-tdt_ctc-110m model using NVIDIA NeMo."""
    try:
        nemo_asr = importlib.import_module("nemo.collections.asr")
    except ImportError as e:
        raise ImportError("nemo_toolkit is required to load this model. Install with: pip install nemo_toolkit['all']") from e
    return nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt_ctc-110m")


def load_nvidia_canary_1b_flash():
    """Load nvidia/canary-1b-flash model using NVIDIA NeMo."""
    try:
        nemo_asr = importlib.import_module("nemo.collections.asr")
    except ImportError as e:
        raise ImportError("nemo_toolkit is required to load this model. Install with: pip install nemo_toolkit['all']") from e
    return nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/canary-1b-flash")


def load_openai_whisper_large_v2():
    """Load openai/whisper-large-v2 model and processor from HuggingFace for direct wav analysis."""
    try:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        import torch
    except ImportError as e:
        raise ImportError("transformers and torch are required to load this model. Install with: pip install transformers torch") from e
    model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v2")
    processor = AutoProcessor.from_pretrained("openai/whisper-large-v2")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return model, processor, device


def load_distil_whisper_distil_large_v3():
    """Load distil-whisper/distil-large-v3 model and processor from HuggingFace for direct wav analysis."""
    try:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        import torch
    except ImportError as e:
        raise ImportError("transformers and torch are required to load this model. Install with: pip install transformers torch") from e
    model = AutoModelForSpeechSeq2Seq.from_pretrained("distil-whisper/distil-large-v3")
    processor = AutoProcessor.from_pretrained("distil-whisper/distil-large-v3")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return model, processor, device


def load_openai_whisper_large():
    """Load openai/whisper-large model and processor from HuggingFace for direct wav analysis."""
    try:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        import torch
    except ImportError as e:
        raise ImportError("transformers and torch are required to load this model. Install with: pip install transformers torch") from e
    model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large")
    processor = AutoProcessor.from_pretrained("openai/whisper-large")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return model, processor, device


def load_openai_whisper_medium():
    """Load openai/whisper-medium model and processor from HuggingFace for direct wav analysis."""
    try:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        import torch
    except ImportError as e:
        raise ImportError("transformers and torch are required to load this model. Install with: pip install transformers torch") from e
    model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-medium")
    processor = AutoProcessor.from_pretrained("openai/whisper-medium")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return model, processor, device


def load_openai_whisper_medium_en():
    """Load openai/whisper-medium.en model and processor from HuggingFace for direct wav analysis."""
    try:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        import torch
    except ImportError as e:
        raise ImportError("transformers and torch are required to load this model. Install with: pip install transformers torch") from e
    model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-medium.en")
    processor = AutoProcessor.from_pretrained("openai/whisper-medium.en")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return model, processor, device


def load_openai_whisper_small():
    """Load openai/whisper-small model and processor from HuggingFace for direct wav analysis."""
    try:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        import torch
    except ImportError as e:
        raise ImportError("transformers and torch are required to load this model. Install with: pip install transformers torch") from e
    model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-small")
    processor = AutoProcessor.from_pretrained("openai/whisper-small")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return model, processor, device


def load_openai_whisper_small_en():
    """Load openai/whisper-small.en model and processor from HuggingFace for direct wav analysis."""
    try:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        import torch
    except ImportError as e:
        raise ImportError("transformers and torch are required to load this model. Install with: pip install transformers torch") from e
    model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-small.en")
    processor = AutoProcessor.from_pretrained("openai/whisper-small.en")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return model, processor, device


def load_openai_whisper_base():
    """Load openai/whisper-base model and processor from HuggingFace for direct wav analysis."""
    try:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        import torch
    except ImportError as e:
        raise ImportError("transformers and torch are required to load this model. Install with: pip install transformers torch") from e
    model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-base")
    processor = AutoProcessor.from_pretrained("openai/whisper-base")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return model, processor, device


def load_openai_whisper_base_en():
    """Load openai/whisper-base.en model and processor from HuggingFace for direct wav analysis."""
    try:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        import torch
    except ImportError as e:
        raise ImportError("transformers and torch are required to load this model. Install with: pip install transformers torch") from e
    model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-base.en")
    processor = AutoProcessor.from_pretrained("openai/whisper-base.en")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return model, processor, device


def load_openai_whisper_tiny():
    """Load openai/whisper-tiny model and processor from HuggingFace for direct wav analysis."""
    try:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        import torch
    except ImportError as e:
        raise ImportError("transformers and torch are required to load this model. Install with: pip install transformers torch") from e
    model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-tiny")
    processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return model, processor, device


def load_openai_whisper_tiny_en():
    """Load openai/whisper-tiny.en model and processor from HuggingFace for direct wav analysis."""
    try:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        import torch
    except ImportError as e:
        raise ImportError("transformers and torch are required to load this model. Install with: pip install transformers torch") from e
    model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-tiny.en")
    processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return model, processor, device

if __name__ == "__main__":
    import sys
    import traceback