import subprocess
import sys
import ai

def install_packages():
    packages = [
        ["numpy<2.0.0"],
        ["binpacking"],
        ["paramiko"],
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
        ["lightning"],
        ["omegaconf"],
        ["hydra-core"],
        ["openai-whisper"],
    ]

    for pkg in packages:
        print(f"Installing: {pkg[0]}")
        cmd = [sys.executable, "-m", "pip", "install", "--break-system-packages"] + pkg
        subprocess.run(cmd, check=True)

def preinstall_all_models():
    ai_model = AIModel()
    ai_model.load(transcribe_english_model_name)
    ai_model.load(transcribe_nonenglish_model_name)
    ai_model.load(identify_languages_model_name)
    ai_model.unload()

if __name__ == "__main__":
    install_packages()
    import torch
    import ai
    preinstall_all_models()