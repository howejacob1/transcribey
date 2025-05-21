import subprocess
import sys
import transcription_models
import torch

def install_packages():
    packages = [
        ["openai", "--break-system-packages", "--force-reinstall"],
        ["toml", "--break-system-packages", "--force-reinstall"],
        ["librosa", "--break-system-packages", "--force-reinstall"],
        ["numpy", "--break-system-packages", "--force-reinstall"],
        ["soundfile", "--break-system-packages", "--force-reinstall"],
        ["transformers", "--break-system-packages", "--force-reinstall"],
        ["matplotlib", "--break-system-packages", "--force-reinstall"],
        ["scikit-learn", "--break-system-packages", "--force-reinstall"],
        ["sentencepiece", "--break-system-packages", "--force-reinstall"],
        ["nemo_toolkit[all]", "--break-system-packages", "--force-reinstall"],
        ["huggingface_hub[hf_xet]", "--break-system-packages", "--force-reinstall"],
        ["pydub", "--break-system-packages", "--force-reinstall"],
        ["ffmpeg-python", "--break-system-packages", "--force-reinstall"],
        ["jiwer", "--break-system-packages", "--force-reinstall"],
        ["speechbrain", "--break-system-packages", "--force-reinstall"],
        ["pyannote-audio", "--break-system-packages", "--force-reinstall"],
        ["pymongo", "--break-system-packages", "--force-reinstall"],
        ["vcon", "--break-system-packages", "--force-reinstall"],
        ["cuda-python", "--break-system-packages", "--force-reinstall"],
        ["nemo_toolkit['all']", "--break-system-packages", "--force-reinstall"],
        ["torch", "--index-url", "https://download.pytorch.org/whl/cu128", "--force-reinstall", "--break-system-packages"],
        ["torchvision", "--index-url", "https://download.pytorch.org/whl/cu128", "--force-reinstall", "--break-system-packages"],
        ["torchaudio", "--index-url", "https://download.pytorch.org/whl/cu128", "--force-reinstall", "--break-system-packages"],
        ["lightning", "--break-system-packages", "--force-reinstall"],
        ["omegaconf", "--break-system-packages", "--force-reinstall"],
        ["hydra-core", "--break-system-packages", "--force-reinstall"],
        ["openai-whisper", "--break-system-packages", "--force-reinstall"],
    ]

    for pkg in packages:
        print(f"Installing: {pkg[0]}")
        cmd = [sys.executable, "-m", "pip", "install"] + pkg
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    install_packages()

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