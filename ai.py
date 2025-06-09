from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib
import time
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import settings
from gpu import move_to_gpu_maybe, gc_collect_maybe
from utils import suppress_output
import vcon_utils
import audio
import numpy as np
from gpu import gpu_ram_free_bytes

def whisper_token_to_language(token):
    return whisper_token_languages[whisper_token_ids.index(token)]

def load_whisper_tiny_raw():
    with suppress_output(should_suppress=False):
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

def load_nvidia_raw(model_name):
    with suppress_output(should_suppress=False):
        nemo_asr = importlib.import_module("nemo.collections.asr")
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
        model.to(torch.float16)
        return model

def load_nvidia(model_name):
    """Load nvidia model using NVIDIA NeMo."""
    total_start_time = time.time()
    print(f"Loading model {model_name}.")
    model = load_nvidia_raw(model_name)
    model = move_to_gpu_maybe(model)
    print(f"Model {model_name} loaded in {time.time() - total_start_time:.2f} seconds total")
    return model

#language_tokens = [t for t in tokenizer.additional_special_tokens if len(t) == 6]
#language_token_ids = tokenizer.convert_tokens_to_ids(language_tokens)
whisper_tokens = ['en', 'zh', 'de', 'es', 'ru', 'ko', 'fr', 'ja', 'pt', 'tr', 'pl', 'ca', 'nl', 'ar', 'sv', 'it', 'id', 'hi', 'fi', 'vi', 'he', 'uk', 'el', 'ms', 'cs', 'ro', 'da', 'hu', 'ta', 'no', 'th', 'ur', 'hr', 'bg', 'lt', 'la', 'mi', 'ml', 'cy', 'sk', 'te', 'fa', 'lv', 'bn', 'sr', 'az', 'sl', 'kn', 'et', 'mk', 'br', 'eu', 'is', 'hy', 'ne', 'mn', 'bs', 'kk', 'sq', 'sw', 'gl', 'mr', 'pa', 'si', 'km', 'sn', 'yo', 'so', 'af', 'oc', 'ka', 'be', 'tg', 'sd', 'gu', 'am', 'yi', 'lo', 'uz', 'fo', 'ht', 'ps', 'tk', 'nn', 'mt', 'sa', 'lb', 'my', 'bo', 'tl', 'mg', 'as', 'tt', 'haw', 'ln', 'ha', 'ba', 'jw', 'su']
whisper_token_languages = ['<|en|>', '<|zh|>', '<|de|>', '<|es|>', '<|ru|>', '<|ko|>', '<|fr|>', '<|ja|>', '<|pt|>', '<|tr|>', '<|pl|>', '<|ca|>', '<|nl|>', '<|ar|>', '<|sv|>', '<|it|>', '<|id|>', '<|hi|>', '<|fi|>', '<|vi|>', '<|he|>', '<|uk|>', '<|el|>', '<|ms|>', '<|cs|>', '<|ro|>', '<|da|>', '<|hu|>', '<|ta|>', '<|no|>', '<|th|>', '<|ur|>', '<|hr|>', '<|bg|>', '<|lt|>', '<|la|>', '<|mi|>', '<|ml|>', '<|cy|>', '<|sk|>', '<|te|>', '<|fa|>', '<|lv|>', '<|bn|>', '<|sr|>', '<|az|>', '<|sl|>', '<|kn|>', '<|et|>', '<|mk|>', '<|br|>', '<|eu|>', '<|is|>', '<|hy|>', '<|ne|>', '<|mn|>', '<|bs|>', '<|kk|>', '<|sq|>', '<|sw|>', '<|gl|>', '<|mr|>', '<|pa|>', '<|si|>', '<|km|>', '<|sn|>', '<|yo|>', '<|so|>', '<|af|>', '<|oc|>', '<|ka|>', '<|be|>', '<|tg|>', '<|sd|>', '<|gu|>', '<|am|>', '<|yi|>', '<|lo|>', '<|uz|>', '<|fo|>', '<|ht|>', '<|ps|>', '<|tk|>', '<|nn|>', '<|mt|>', '<|sa|>', '<|lb|>', '<|my|>', '<|bo|>', '<|tl|>', '<|mg|>', '<|as|>', '<|tt|>', '<|haw|>', '<|ln|>', '<|ha|>', '<|ba|>', '<|jw|>', '<|su|>']
whisper_token_ids = [50259, 50260, 50261, 50262, 50263, 50264, 50265, 50266, 50267, 50268, 50269, 50270, 50271, 50272, 50273, 50274, 50275, 50276, 50277, 50278, 50279, 50280, 50281, 50282, 50283, 50284, 50285, 50286, 50287, 50288, 50289, 50290, 50291, 50292, 50293, 50294, 50295, 50296, 50297, 50298, 50299, 50300, 50301, 50302, 50303, 50304, 50305, 50306, 50307, 50308, 50309, 50310, 50311, 50312, 50313, 50314, 50315, 50316, 50317, 50318, 50319, 50320, 50321, 50322, 50323, 50324, 50325, 50326, 50327, 50328, 50329, 50330, 50331, 50332, 50333, 50334, 50335, 50336, 50337, 50338, 50339, 50340, 50341, 50342, 50343, 50344, 50345, 50346, 50347, 50348, 50349, 50350, 50351, 50352, 50353, 50354, 50355, 50356, 50357]
whisper_start_transcription_token_id = 50258

def identify_language_batch(vcon_batch, audio_data_batch, inputs, model):
    vcons = []
    with torch.no_grad():
        #vcon_utils.print_audio_duration_many(vcon_utils.unbatch(all_vcons_batched))
        # audio_data_processed = []
        # for audio_data in audio_data_batch:
        #     audio_data = audio_data.squeeze().cpu().numpy().astype(np.float32)
        #     audio_data_processed.append(audio_data)
        # # Ensure all audio is a 1D, float32, mono torch tensor
        # audio_data_squeezed = []
        # for i, audio_data in enumerate(audio_data_batch):
        #     # Convert numpy arrays to torch tensors
        #     if isinstance(audio_data, np.ndarray):
        #         audio_data = torch.tensor(audio_data)
        #     # Squeeze and flatten to 1D
        #     audio_data = audio_data.squeeze()
        #     if audio_data.ndim != 1:
        #         # If still not 1D, take the first channel (assume mono)
        #         audio_data = audio_data[0]
        #     # Ensure float32
        #     audio_data = audio_data.to(torch.float32)
        #     print(f"audio_data[{i}] type: {type(audio_data)}, shape: {getattr(audio_data, 'shape', None)}, dtype: {getattr(audio_data, 'dtype', None)}")
        #     audio_data_squeezed.append(audio_data)
        # Batch process
        #print(f"before processor gpu_ram_free_bytes: {gpu_ram_free_bytes()}")
        
        #print(f"after processor, before move_to_gpu_maybe(inputs) gpu_ram_free_bytes: {gpu_ram_free_bytes()}")
        inputs = move_to_gpu_maybe(inputs)
        #print(f"after move_to_gpu_maybe, before move_to_gpu_maybe(input_features) gpu_ram_free_bytes: {gpu_ram_free_bytes()}")
        input_features = inputs.input_features
        input_features = move_to_gpu_maybe(input_features)
        #print(f"after move_to_gpu_maybe(input_features), before move_to_gpu_maybe(decoder_input_ids) gpu_ram_free_bytes: {gpu_ram_free_bytes()}")
        decoder_input_ids = torch.tensor([[whisper_start_transcription_token_id]] * len(audio_data_batch))
        #print(f"after decoder_input_ids, before move_to_gpu_maybe(decoder_input_ids) gpu_ram_free_bytes: {gpu_ram_free_bytes()}")
        decoder_input_ids = move_to_gpu_maybe(decoder_input_ids)
        #print(f"after move_to_gpu_maybe(decoder_input_ids), before model(input_features, decoder_input_ids=decoder_input_ids) gpu_ram_free_bytes: {gpu_ram_free_bytes()}")
        
        model_output = model(input_features, decoder_input_ids=decoder_input_ids)
        #print(f"after model(input_features, decoder_input_ids=decoder_input_ids), before logits gpu_ram_free_bytes: {gpu_ram_free_bytes()}")
        logits = model_output.logits
        #print(f"after logits, before logits[:, 0, :] gpu_ram_free_bytes: {gpu_ram_free_bytes()}")
        logits = logits[:, 0, :]
        #print(f"after logits[:, 0, :], before for i, vcon in enumerate(vcon_batch) gpu_ram_free_bytes: {gpu_ram_free_bytes()}")
        for i, vcon in enumerate(vcon_batch):
            #print(f"after for i, vcon in enumerate(vcon_batch), before lang_logits gpu_ram_free_bytes: {gpu_ram_free_bytes()}")
            lang_logits = logits[i, whisper_token_ids]
            #print(f"after lang_logits, before lang_probs gpu_ram_free_bytes: {gpu_ram_free_bytes()}")
            lang_probs = torch.softmax(lang_logits, dim=-1).cpu().detach().numpy()
            #print(f"after lang_probs, before audio_languages_detected gpu_ram_free_bytes: {gpu_ram_free_bytes()}")
            audio_languages_detected = [whisper_tokens[j] for j, prob in enumerate(lang_probs) if prob >= settings.lang_detect_threshold]
            #print(f"after audio_languages_detected, before audio_languages_detected gpu_ram_free_bytes: {gpu_ram_free_bytes()}")
            if audio_languages_detected:
                languages = audio_languages_detected
            else:
                max_prob_idx = lang_probs.argmax()
                #print(f"gpu_ram_free_bytes: {gpu_ram_free_bytes()}")
                languages = [whisper_tokens[max_prob_idx]]
                #print(f"gpu_ram_free_bytes: {gpu_ram_free_bytes()}")
            vcon_utils.set_languages(vcon, languages)
            #print(f"gpu_ram_free_bytes: {gpu_ram_free_bytes()}")
            vcons.append(vcon)
        
    return vcons
    

def preprocess_identify_languages(vcon_batch, audio_data_batch, processor):
    audio_data_batch = vcon_utils.batch_to_audio_data(vcon_batch)
    inputs = processor(audio_data_batch, sampling_rate=settings.sample_rate, return_tensors="pt", padding="max_length")
    return vcon_batch, audio_data_batch, inputs

def identify_languages(all_vcons_batched, model, processor):
    vcons = []
    futures = []
    with ThreadPoolExecutor(max_workers=audio.cpu_cores_for_preprocessing()) as executor:
        for vcon_batch in all_vcons_batched:
            audio_data_batch = vcon_utils.batch_to_audio_data(vcon_batch)
            futures.append(executor.submit(preprocess_identify_languages, vcon_batch, audio_data_batch, processor))
        for future in as_completed(futures):
            vcon_batch, audio_data_batch, inputs = future.result()
            vcons.extend(identify_language_batch(vcon_batch, audio_data_batch, inputs, model))
        gc_collect_maybe()
    return vcons

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
    config = {}
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
