import subprocess
import sys

def install_packages():
    packages = [
        ["torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu128"],
        ["openai", 
         "cupy-cuda12x",
         "toml", 
         "librosa", 
         "soundfile", 
         "transformers", 
         "matplotlib", 
         "scikit-learn",
         "sentencepiece", 
         "huggingface_hub[hf_xet]", 
         "pydub", 
         "ffmpeg-python", 
         "jiwer", 
         "speechbrain", 
         "pyannote-audio",
         "pymongo", 
         "vcon", 
         "cuda-python",
         "lightning",
         "omegaconf",
         "hydra-core",
         "openai-whisper",
         "GPUtil", 
         "numpy<2.0.0",
         "binpacking", 
         "paramiko", 
         "nemo_toolkit[all]"]
    ]

    #cmd = [sys.executable, "source", ".venv/bin/activate"]
    #subprocess.run(cmd, check=True)
    for pkg in packages:
        print(f"Installing: {pkg[0]}")
        cmd = [sys.executable, "-m", "pip", "install"] + pkg
        subprocess.run(cmd, check=True)

#def preinstall_all_models():
#    ai_model = AIModel()
#    ai_model.load(transcribe_english_model_name)
#    ai_model.load(transcribe_nonenglish_model_name)
#    ai_model.load(identify_languages_model_name)
#    ai_model.unload()

if __name__ == "__main__":
    install_packages()
    import lang_detect
    import transcribe_en
    import transcribe_non_en

    print("Loading models...")
    lang_detect.load()
    transcribe_en.load()
    transcribe_non_en.load()
    # import torch
    # preinstall_all_models()