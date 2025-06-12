import gpu
import time
import nemo_asr
from utils import suppress_output, move_to_gpu_maybe

def load_nvidia_raw(model_name):
    with suppress_output(should_suppress=True):
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
        return model

def load_nvidia(model_name):
    """Load nvidia model using NVIDIA NeMo."""
    total_start_time = time.time()
    print(f"Loading model {model_name}.")
    model = load_nvidia_raw(model_name)
    model = move_to_gpu_maybe(model)
    print(f"Model {model_name} loaded in {time.time() - total_start_time:.2f} seconds total")
    return model