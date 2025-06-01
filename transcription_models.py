# transcription_models.py
"""
This module provides functions to load and manage all ASR models used in the project.
"""

# Example placeholder for model loading functions
# Actual implementation will depend on the model frameworks/APIs used (e.g., HuggingFace, NVIDIA, Microsoft, etc.)

import importlib
from utils import suppress_output, get_device
import time
import logging
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torch
import torchaudio
from settings import lang_detect_batch_size, lang_detect_threshold

def load_whisper_tiny():
    """
    Load the OpenAI Whisper tiny model and processor.
    Returns (model, processor, device)
    """
    print("Loading whisper tiny")
    with suppress_output(should_suppress=True):
        model_name = "openai/whisper-tiny"
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
        model = model.to(get_device())
        return (model, processor)

def load_nvidia(model_name):
    """Load nvidia model using NVIDIA NeMo."""
    print(f"Loading model {model_name}")
    with suppress_output(should_suppress=True):
        nemo_asr = importlib.import_module("nemo.collections.asr")
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
        model.to(get_device())
        return model

transcribe_english_model_name = "nvidia/parakeet-tdt-0.6b-v2"
transcribe_nonenglish_model_name = "nvidia/canary-1b-flash"
identify_languages_model_name = "openai/whisper-tiny"

def load_model(model_name):
    if model_name == "openai/whisper-tiny":
        return load_whisper_tiny()
    else:
        return load_nvidia(model_name)

def preinstall_all_models():
    ai_model = AIModel()
    ai_model.load(transcribe_english_model_name)
    ai_model.load(transcribe_nonenglish_model_name)
    ai_model.load(identify_languages_model_name)
    ai_model.unload()

class AIModel:
    def __init__(self):
        self.model = None
        self.model_name = None

    def unload(self):
        del self.model
        del self.model_name
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model = None
        self.model_name = None

    def load(self, model_to_load):
        if model_to_load != self.model_name:
            self.unload()
            self.model = load_model(model_to_load)
            self.model_name = model_to_load

    def load_en_transcription(self):
        self.load(transcribe_english_model_name)
    
    def load_non_en_transcription(self):
        self.load(transcribe_nonenglish_model_name)
    
    def load_lang_detect(self):
        self.load(identify_languages_model_name)

    def identify_languages(self, wav_files):
        self.load(identify_languages_model_name)
        return identify_languages(self.model, wav_files)

    def transcribe(self, wav_files, english_only=False):
        if english_only:
            self.load(transcribe_english_model_name)
        else:
            self.load(transcribe_nonenglish_model_name)
        return self.transcribe(wav_files)

    def loaded_model_mode(self):
        if self.model_name == transcribe_english_model_name:
            return "en"
        elif self.model_name == transcribe_nonenglish_model_name:
            return "non_en"
        elif self.model_name == identify_languages_model_name:
            return "lang_detect"
        
    def load_by_mode(self, mode):
        if mode == "en":
            self.load_en_transcription()
        elif mode == "non_en":
            self.load_non_en_transcription()
        elif mode == "lang_detect":
            self.load_lang_detect()

def resample_wav_maybe(wav, sample_rate, target_sample_rate=16000):
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        wav = resampler(wav)
        wav = wav.squeeze().numpy()
    return wav

def load_and_resample_wavs(all_wav_paths, target_sample_rate=16000):
    """
    Loads and resamples a list of wav files to the target sample rate.
    Returns a list of numpy arrays (waveforms) and a list of valid indices.
    """
    wavs = []
    for wav_path in all_wav_paths:
        raw_wav, sample_rate = torchaudio.load(wav_path)
        wav = resample_wav_maybe(raw_wav, sample_rate)
        wavs.append(wav)
    
    return wavs

def split_wavs_into_batches(wavs, batch_size):
    """
    Splits a list of wavs into batches of size batch_size.
    """
    return [wavs[i:i + batch_size] for i in range(0, len(wavs), batch_size)]

#language_tokens = [t for t in tokenizer.additional_special_tokens if len(t) == 6]
#language_token_ids = tokenizer.convert_tokens_to_ids(language_tokens)
whisper_tokens = ['en', 'zh', 'de', 'es', 'ru', 'ko', 'fr', 'ja', 'pt', 'tr', 'pl', 'ca', 'nl', 'ar', 'sv', 'it', 'id', 'hi', 'fi', 'vi', 'he', 'uk', 'el', 'ms', 'cs', 'ro', 'da', 'hu', 'ta', 'no', 'th', 'ur', 'hr', 'bg', 'lt', 'la', 'mi', 'ml', 'cy', 'sk', 'te', 'fa', 'lv', 'bn', 'sr', 'az', 'sl', 'kn', 'et', 'mk', 'br', 'eu', 'is', 'hy', 'ne', 'mn', 'bs', 'kk', 'sq', 'sw', 'gl', 'mr', 'pa', 'si', 'km', 'sn', 'yo', 'so', 'af', 'oc', 'ka', 'be', 'tg', 'sd', 'gu', 'am', 'yi', 'lo', 'uz', 'fo', 'ht', 'ps', 'tk', 'nn', 'mt', 'sa', 'lb', 'my', 'bo', 'tl', 'mg', 'as', 'tt', 'haw', 'ln', 'ha', 'ba', 'jw', 'su']
whisper_token_languages = ['<|en|>', '<|zh|>', '<|de|>', '<|es|>', '<|ru|>', '<|ko|>', '<|fr|>', '<|ja|>', '<|pt|>', '<|tr|>', '<|pl|>', '<|ca|>', '<|nl|>', '<|ar|>', '<|sv|>', '<|it|>', '<|id|>', '<|hi|>', '<|fi|>', '<|vi|>', '<|he|>', '<|uk|>', '<|el|>', '<|ms|>', '<|cs|>', '<|ro|>', '<|da|>', '<|hu|>', '<|ta|>', '<|no|>', '<|th|>', '<|ur|>', '<|hr|>', '<|bg|>', '<|lt|>', '<|la|>', '<|mi|>', '<|ml|>', '<|cy|>', '<|sk|>', '<|te|>', '<|fa|>', '<|lv|>', '<|bn|>', '<|sr|>', '<|az|>', '<|sl|>', '<|kn|>', '<|et|>', '<|mk|>', '<|br|>', '<|eu|>', '<|is|>', '<|hy|>', '<|ne|>', '<|mn|>', '<|bs|>', '<|kk|>', '<|sq|>', '<|sw|>', '<|gl|>', '<|mr|>', '<|pa|>', '<|si|>', '<|km|>', '<|sn|>', '<|yo|>', '<|so|>', '<|af|>', '<|oc|>', '<|ka|>', '<|be|>', '<|tg|>', '<|sd|>', '<|gu|>', '<|am|>', '<|yi|>', '<|lo|>', '<|uz|>', '<|fo|>', '<|ht|>', '<|ps|>', '<|tk|>', '<|nn|>', '<|mt|>', '<|sa|>', '<|lb|>', '<|my|>', '<|bo|>', '<|tl|>', '<|mg|>', '<|as|>', '<|tt|>', '<|haw|>', '<|ln|>', '<|ha|>', '<|ba|>', '<|jw|>', '<|su|>']
whisper_token_ids = [50259, 50260, 50261, 50262, 50263, 50264, 50265, 50266, 50267, 50268, 50269, 50270, 50271, 50272, 50273, 50274, 50275, 50276, 50277, 50278, 50279, 50280, 50281, 50282, 50283, 50284, 50285, 50286, 50287, 50288, 50289, 50290, 50291, 50292, 50293, 50294, 50295, 50296, 50297, 50298, 50299, 50300, 50301, 50302, 50303, 50304, 50305, 50306, 50307, 50308, 50309, 50310, 50311, 50312, 50313, 50314, 50315, 50316, 50317, 50318, 50319, 50320, 50321, 50322, 50323, 50324, 50325, 50326, 50327, 50328, 50329, 50330, 50331, 50332, 50333, 50334, 50335, 50336, 50337, 50338, 50339, 50340, 50341, 50342, 50343, 50344, 50345, 50346, 50347, 50348, 50349, 50350, 50351, 50352, 50353, 50354, 50355, 50356, 50357]
whisper_start_transcription_token_id = 50258

def whisper_token_to_language(token):
    return whisper_token_languages[whisper_token_ids.index(token)]

def identify_languages(all_wav_paths, model_and_processor, threshold=None):
    model, processor = model_and_processor
    if threshold is None:
        threshold = lang_detect_threshold
    device = get_device()

    all_wav_paths_batched = split_wavs_into_batches(all_wav_paths, lang_detect_batch_size)

    all_wav_languages_detected = []
    for wav_paths in all_wav_paths_batched:
        wavs = load_and_resample_wavs(wav_paths, target_sample_rate=16000)
        input_features = processor(wavs, sampling_rate=16000, return_tensors="pt").input_features.to(device)

        with torch.no_grad():
            logits = model(input_features, decoder_input_ids=torch.tensor([[whisper_start_transcription_token_id]] * len(wavs), device=device)).logits
        logits = logits[:, 0, :]  # (batch, vocab_size)

        for index in range(len(wavs)):
            lang_logits = logits[index, whisper_token_ids]
            lang_probs = torch.softmax(lang_logits, dim=-1).cpu().numpy()
            wav_languages_detected = [whisper_token_languages[j] for j, prob in enumerate(lang_probs) if prob >= threshold]
            all_wav_languages_detected.append(wav_languages_detected)
    assert len(all_wav_languages_detected) == len(all_wav_paths)
    return all_wav_languages_detected

def transcribe_vcons(collection, model, vcons_to_transcribe, batch_size):
    """
    Transcribe a list of vCons using the specified model.
    
    Args:
        collection: MongoDB collection containing vCons
        model: Loaded model instance (should have a .transcribe method)
        vcons_to_transcribe: List of (vcon_id, file_path) tuples to transcribe
        batch_size: Number of files to process in each batch
    """

    if not vcons_to_transcribe:
        logging.info(f"No vCons to transcribe with {getattr(model, 'name', getattr(model, '__class__', type(model)).__name__)}")
        return
        
    total = len(vcons_to_transcribe)
    model_name = getattr(model, 'name', getattr(model, '__class__', type(model)).__name__)
    logging.info(f"Starting transcription of {total} vCons with {model_name}")
    start_time = time.time()
    processed = 0
    file_paths_only = [file_path for _, file_path in vcons_to_transcribe]
    total_bytes = get_total_wav_size(file_paths_only)
    total_audio_seconds = get_total_wav_duration(file_paths_only)
    all_files = file_paths_only
    for i in range(0, total, batch_size):
        batch = vcons_to_transcribe[i:min(i+batch_size, total)]
        batch_ids = [vcon_id for vcon_id, _ in batch]
        batch_files = [file_path for _, file_path in batch]
        valid_indices = []
        valid_files = []
        valid_ids = []
        for idx, file_path in enumerate(batch_files):
            if os.path.exists(file_path) and os.path.isfile(file_path):
                valid_indices.append(idx)
                valid_files.append(file_path)
                valid_ids.append(batch_ids[idx])
            else:
                logging.warning(f"File not found, not accessible, or invalid: {file_path}")
        if not valid_files:
            logging.warning(f"No valid files in batch {i//batch_size + 1}")
            continue
        batch_start = time.time()
        try:
            print(f"Files to transcribe: {valid_files}")
            transcriptions = model.transcribe(valid_files)
            batch_elapsed = time.time() - batch_start
            logging.info(f"Transcribed batch of {len(valid_files)} files with {model_name} in {batch_elapsed:.2f} seconds")
            for vcon_id, file_path, transcription in zip(valid_ids, valid_files, transcriptions):
                analysis = {
                    "type": "transcription",
                    "dialog": [0],
                    "vendor": model_name,
                    "body": transcription.text if hasattr(transcription, 'text') else str(transcription),
                    "encoding": "none"
                }
                try:
                    collection.update_one(
                        {"_id": vcon_id},
                        {"$push": {"analysis": analysis}}
                    )
                    logging.info(f"Transcribed {os.path.basename(file_path)} with {model_name}: {analysis['body']}")
                    processed += 1
                except Exception as e:
                    logging.error(f"Error updating transcription for vCon {vcon_id}: {str(e)}")
        except RuntimeError as e:
            if 'CUDA error' in str(e):
                logging.error(f"CUDA error during transcription: {str(e)}. Clearing CUDA cache and continuing.")
                torch.cuda.empty_cache()
            else:
                logging.error(f"Error transcribing batch with {model_name}: {str(e)}")
        except Exception as e:
            logging.error(f"Error transcribing batch with {model_name}: {str(e)}")
        if processed % 100 == 0 or processed == total:
            logging.info(f"Progress: {processed}/{total} vCons transcribed with {model_name}")
    total_elapsed = time.time() - start_time
    rtf = None
    if total_elapsed > 0 and total_audio_seconds > 0:
        rtf = total_audio_seconds / total_elapsed
    logging.info(f"Completed transcription of {processed} vCons with {model_name} in {total_elapsed:.2f} seconds")
    logging.info(f"Total data: {total_bytes / (1024**3):.2f} GB, total length: {total_audio_seconds:.2f} seconds, real time factor: {rtf:.1f}x")
    del model
    torch.cuda.empty_cache()
    return {
        'rtf': rtf,
        'duration': total_audio_seconds,
        'size': total_bytes,
        'batch_size': batch_size
    }

if __name__ == "__main__":
    import sys
    import traceback
    preinstall_all_models()