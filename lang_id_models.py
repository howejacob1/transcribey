import torchaudio


#VoxLingua107, ECAPA-TDNN, Rep-ECAPA-TDNN, pyannote-audio

def test_speechbrain_voxlingua107_ecapa(wav_path):
    """Test the speechbrain/lang-id-voxlingua107-ecapa model on a given wav file and print the prediction."""
    from speechbrain.inference.classifiers import EncoderClassifier
    language_id = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp")
    signal = language_id.load_audio(wav_path)
    prediction = language_id.classify_batch(signal)
    print(prediction)

def test_speechbrain_commonlanguage_ecapa(wav_path):
    """Test the speechbrain/lang-id-commonlanguage_ecapa model on a given wav file and print the prediction."""
    from speechbrain.inference.classifiers import EncoderClassifier
    language_id = EncoderClassifier.from_hparams(source="speechbrain/lang-id-commonlanguage_ecapa", savedir="tmp")
    signal = language_id.load_audio(wav_path)
    prediction = language_id.classify_batch(signal)
    print(prediction)

def test_whisper_tiny_lang_id(wav_path):
    """Test the Whisper Tiny model for language identification on a given wav file and print the prediction."""
    import whisper
    # Load the tiny Whisper model
    model = whisper.load_model('tiny')
    # Load and preprocess the audio
    audio = whisper.load_audio(wav_path)
    mel = whisper.log_mel_spectrogram(audio)
    mel = whisper.pad_or_trim(mel, 3000)  # 3000 is N_FRAMES for Whisper
    # Run language detection
    _, probs = model.detect_language(mel)
    language = max(probs, key=probs.get)
    print({"language": language, "probs": probs})