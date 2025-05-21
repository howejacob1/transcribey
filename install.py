import subprocess
import sys

def install_packages():
    packages = [
        ["numpy", "<2.0.0"],
        ["torch", "--index-url", "https://download.pytorch.org/whl/cu128"],
        ["torchvision", "--index-url", "https://download.pytorch.org/whl/cu128"],
        ["torchaudio", "--index-url", "https://download.pytorch.org/whl/cu128"],
        ["nemo_toolkit[all]"],
        ["openai"],
        ["toml"],
        ["librosa"],
        ["soundfile"],
        ["transformers"],
        ["matplotlib"],
        ["scikit-learn"],
        ["sentencepiece"],
        ["huggingface_hub[hf_xet]"],
        ["pydub"],
        ["ffmpeg-python"],
        ["jiwer"],
        ["speechbrain"],
        ["pyannote-audio"],
        ["pymongo"],
        ["vcon"],
        ["cuda-python"],
        ["nemo_toolkit['all']"],
        ["lightning"],
        ["omegaconf"],
        ["hydra-core"],
        ["openai-whisper"],
    ]

    for pkg in packages:
        print(f"Installing: {pkg[0]}")
        cmd = [sys.executable, "-m", "pip", "install"] + pkg
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    install_packages()

    import transcription_models
    import torch

    # Cache all models by loading and deleting them in order
    model_names = [
        "nvidia/parakeet-tdt-0.6b-v2",
        "nvidia/canary-1b-flash",
    ]
    for name in model_names:
        print(f"Caching model: {name}")
        try:
            model = transcription_models.load_model_by_name(name)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Failed to cache model {name}: {e}")

    # Cache OpenAI Whisper tiny model
    print("Caching model: openai/whisper-tiny")
    try:
        model, processor, device = transcription_models.load_openai_whisper_tiny()
        del model
        del processor
        del device
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"Failed to cache model openai/whisper-tiny: {e}") 