from transcription_models import load_nvidia_parakeet_tdt_ctc_110m

# Picked sample wav file
wav_path = "fake_wavs/level0000/copy_94200076.wav"

print(f"Loading model...")
asr_model = load_nvidia_parakeet_tdt_ctc_110m()
print(f"Model loaded. Transcribing {wav_path} ...")

# Run transcription
result = asr_model.transcribe([wav_path])
print("Transcription result:")
print(result) 