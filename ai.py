import importlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cupy as np
import nemo.collections.asr as nemo_asr
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

import audio
import settings
import utils
import vcon_utils
from gpu import gpu_ram_free_bytes, move_to_gpu_maybe, gc_collect_maybe
from utils import suppress_output

def whisper_token_to_language(token):
    return whisper_token_languages[whisper_token_ids.index(token)]

def load_whisper_tiny_raw():
    with suppress_output(should_suppress=True):
        model_name = "openai/whisper-tiny"
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
        return (model, processor)

def load_whisper_tiny():
    """
    Load the OpenAI Whisper tiny model and processor.
    Returns (model, processor, device)
    """
    total_start_time = time.time()
    print("Loading whisper tiny.")
    model, processor = load_whisper_tiny_raw()
    model = move_to_gpu_maybe(model)
    print(f"Whisper tiny loaded in {time.time() - total_start_time:.2f} seconds total")
    return (model, processor)





#language_tokens = [t for t in tokenizer.additional_special_tokens if len(t) == 6]
#language_token_ids = tokenizer.convert_tokens_to_ids(language_tokens)
whisper_tokens = ['en', 'zh', 'de', 'es', 'ru', 'ko', 'fr', 'ja', 'pt', 'tr', 'pl', 'ca', 'nl', 'ar', 'sv', 'it', 'id', 'hi', 'fi', 'vi', 'he', 'uk', 'el', 'ms', 'cs', 'ro', 'da', 'hu', 'ta', 'no', 'th', 'ur', 'hr', 'bg', 'lt', 'la', 'mi', 'ml', 'cy', 'sk', 'te', 'fa', 'lv', 'bn', 'sr', 'az', 'sl', 'kn', 'et', 'mk', 'br', 'eu', 'is', 'hy', 'ne', 'mn', 'bs', 'kk', 'sq', 'sw', 'gl', 'mr', 'pa', 'si', 'km', 'sn', 'yo', 'so', 'af', 'oc', 'ka', 'be', 'tg', 'sd', 'gu', 'am', 'yi', 'lo', 'uz', 'fo', 'ht', 'ps', 'tk', 'nn', 'mt', 'sa', 'lb', 'my', 'bo', 'tl', 'mg', 'as', 'tt', 'haw', 'ln', 'ha', 'ba', 'jw', 'su']
whisper_token_languages = ['<|en|>', '<|zh|>', '<|de|>', '<|es|>', '<|ru|>', '<|ko|>', '<|fr|>', '<|ja|>', '<|pt|>', '<|tr|>', '<|pl|>', '<|ca|>', '<|nl|>', '<|ar|>', '<|sv|>', '<|it|>', '<|id|>', '<|hi|>', '<|fi|>', '<|vi|>', '<|he|>', '<|uk|>', '<|el|>', '<|ms|>', '<|cs|>', '<|ro|>', '<|da|>', '<|hu|>', '<|ta|>', '<|no|>', '<|th|>', '<|ur|>', '<|hr|>', '<|bg|>', '<|lt|>', '<|la|>', '<|mi|>', '<|ml|>', '<|cy|>', '<|sk|>', '<|te|>', '<|fa|>', '<|lv|>', '<|bn|>', '<|sr|>', '<|az|>', '<|sl|>', '<|kn|>', '<|et|>', '<|mk|>', '<|br|>', '<|eu|>', '<|is|>', '<|hy|>', '<|ne|>', '<|mn|>', '<|bs|>', '<|kk|>', '<|sq|>', '<|sw|>', '<|gl|>', '<|mr|>', '<|pa|>', '<|si|>', '<|km|>', '<|sn|>', '<|yo|>', '<|so|>', '<|af|>', '<|oc|>', '<|ka|>', '<|be|>', '<|tg|>', '<|sd|>', '<|gu|>', '<|am|>', '<|yi|>', '<|lo|>', '<|uz|>', '<|fo|>', '<|ht|>', '<|ps|>', '<|tk|>', '<|nn|>', '<|mt|>', '<|sa|>', '<|lb|>', '<|my|>', '<|bo|>', '<|tl|>', '<|mg|>', '<|as|>', '<|tt|>', '<|haw|>', '<|ln|>', '<|ha|>', '<|ba|>', '<|jw|>', '<|su|>']
whisper_token_ids = [50259, 50260, 50261, 50262, 50263, 50264, 50265, 50266, 50267, 50268, 50269, 50270, 50271, 50272, 50273, 50274, 50275, 50276, 50277, 50278, 50279, 50280, 50281, 50282, 50283, 50284, 50285, 50286, 50287, 50288, 50289, 50290, 50291, 50292, 50293, 50294, 50295, 50296, 50297, 50298, 50299, 50300, 50301, 50302, 50303, 50304, 50305, 50306, 50307, 50308, 50309, 50310, 50311, 50312, 50313, 50314, 50315, 50316, 50317, 50318, 50319, 50320, 50321, 50322, 50323, 50324, 50325, 50326, 50327, 50328, 50329, 50330, 50331, 50332, 50333, 50334, 50335, 50336, 50337, 50338, 50339, 50340, 50341, 50342, 50343, 50344, 50345, 50346, 50347, 50348, 50349, 50350, 50351, 50352, 50353, 50354, 50355, 50356, 50357]
whisper_start_transcription_token_id = 50258    

# def preprocess_identify_languages(vcon_batch, audio_data_batch, processor):
#     audio_data_batch = vcon_utils.batch_to_audio_data(vcon_batch)
#     inputs = processor(audio_data_batch, sampling_rate=settings.sample_rate, return_tensors="pt", padding="max_length")
#     return vcon_batch, audio_data_batch, inputs


def transcribe_batch(vcon_batch, model, language="en", config={}):
    audio_data_batch = vcon_utils.batch_to_audio_data(vcon_batch)
    # audio_data_processed = []
    # for audio_data in audio_data_batch:
    #     audio_data = audio_data.squeeze().numpy().astype(np.float32)
    #     audio_data_processed.append(audio_data)
    all_transcriptions = model.transcribe(audio_data_batch, **config)
    vcons = []
    for vcon_obj, transcription in zip(vcon_batch, all_transcriptions):
        text = transcription.text
        vcon_obj = vcon_utils.set_transcript(vcon_obj, text)
        vcons.append(vcon_obj)
    return vcons

def transcribe_many(vcons_batched, model, language="en"):
    vcons = []
    config = {"batch_size": 52}
    if language != "en":
        config = {"source_lang": language,
                    "target_lang": language,
                    "task": "asr",
                    "pnc": "yes"}
    #print(f"gpu_ram_free_bytes: {gpu_ram_free_bytes()}")
    for vcon_batch in vcons_batched:
        vcons.extend(transcribe_batch(vcon_batch, model, language, config))
        gc_collect_maybe()
    return vcons
