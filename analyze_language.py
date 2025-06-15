import os
import sys

import cupy as np
import torch

import ai
import audio
import settings
import vcon_utils

def print_audio_duration_many_patch(vcons):
    for vcon in vcons:
        audio_data = vcon_utils.get_audio(vcon)
        if isinstance(audio_data, np.ndarray):
            if audio_data.ndim == 2:
                print(f"duration: {audio_data.shape[1]}")
                print(f"channels: {audio_data.shape[0]}")
            elif audio_data.ndim == 1:
                print(f"duration: {audio_data.shape[0]}")
                print(f"channels: 1")
        elif isinstance(audio_data, torch.Tensor):
            if audio_data.ndim == 2:
                print(f"duration: {audio_data.shape[1]}")
                print(f"channels: {audio_data.shape[0]}")
            elif audio_data.ndim == 1:
                print(f"duration: {audio_data.shape[0]}")
                print(f"channels: 1")
        else:
            print(f"Unknown audio data type: {type(audio_data)}")

vcon_utils.print_audio_duration_many = print_audio_duration_many_patch

def batch_to_audio_data_patch(batch):
    audio_data_list = []
    for vcon in batch:
        audio_data_val = vcon_utils.get_audio(vcon)
        # Always return 1D numpy array (float32)
        if isinstance(audio_data_val, torch.Tensor):
            arr = audio_data_val.squeeze().cpu().numpy().astype(np.float32)
        elif isinstance(audio_data_val, np.ndarray):
            arr = audio_data_val.squeeze().astype(np.float32)
        else:
            raise ValueError(f"Unsupported audio type: {type(audio_data_val)}")
        audio_data_list.append(arr)
    return audio_data_list

vcon_utils.batch_to_audio_data = batch_to_audio_data_patch

def identify_languages_patch(all_vcons_batched, model, processor):
    vcons = []
    for vcon_batch in all_vcons_batched:
        vcon_utils.print_audio_duration_many(vcon_utils.unbatch(all_vcons_batched))
        audio_data_batch = vcon_utils.batch_to_audio_data(vcon_batch)
        # Ensure all audio is a 1D numpy array (float32)
        audio_data_squeezed = []
        for i, audio_data in enumerate(audio_data_batch):
            if isinstance(audio_data, torch.Tensor):
                audio_data = audio_data.squeeze().cpu().numpy().astype(np.float32)
            elif isinstance(audio_data, np.ndarray):
                audio_data = audio_data.squeeze().astype(np.float32)
            else:
                raise ValueError(f"Unsupported audio type: {type(audio_data)}")
            print(f"identify_languages_patch: audio_data[{i}] type: {type(audio_data)}, shape: {getattr(audio_data, 'shape', None)}, dtype: {getattr(audio_data, 'dtype', None)}")
            audio_data_squeezed.append(audio_data)
        # Batch process with correct padding
        inputs = processor(
            audio_data_squeezed,
            sampling_rate=settings.sample_rate,
            return_tensors="pt",
            padding="max_length"
        )
        inputs = ai.move_to_gpu_maybe(inputs)
        input_features = inputs.input_features
        input_features = ai.move_to_gpu_maybe(input_features)
        decoder_input_ids = torch.tensor([[ai.whisper_start_transcription_token_id]] * len(audio_data_squeezed))
        decoder_input_ids = ai.move_to_gpu_maybe(decoder_input_ids)
        model_output = model(input_features, decoder_input_ids=decoder_input_ids)
        logits = model_output.logits
        logits = logits[:, 0, :]
        for i, vcon in enumerate(vcon_batch):
            lang_logits = logits[i, ai.whisper_token_ids]
            lang_probs = torch.softmax(lang_logits, dim=-1).cpu().detach().numpy()
            audio_languages_detected = [ai.whisper_tokens[j] for j, prob in enumerate(lang_probs) if prob >= settings.lang_detect_threshold]
            if audio_languages_detected:
                languages = audio_languages_detected
            else:
                max_prob_idx = lang_probs.argmax()
                languages = [ai.whisper_tokens[max_prob_idx]]
            vcon_utils.set_languages(vcon, languages)
            vcons.append(vcon)
        ai.gc_collect_maybe()
    return vcons

ai.identify_languages = identify_languages_patch

def main():
    # Path to a random audio file from openslr-12
    audio_file = "/home/bantaim/conserver/openslr-12/LibriSpeech/dev-other/6267/53049/6267-53049-0022.flac"
    if not os.path.exists(audio_file):
        print(f"Audio file not found: {audio_file}")
        sys.exit(1)

    # Create a vcon object for the audio file
    vcon = vcon_utils.create(audio_file)
    audio_data, sample_rate = audio.load_to_cpu(audio_file)
    # Resample if needed
    audio_data = audio.resample_audio(audio_data, sample_rate)
    # Force to 1D numpy array (float32)
    if isinstance(audio_data, (torch.Tensor,)):
        audio_data_np = audio_data.squeeze().cpu().numpy().astype(np.float32)
    elif isinstance(audio_data, np.ndarray):
        audio_data_np = audio_data.squeeze().astype(np.float32)
    else:
        raise ValueError(f"Unsupported audio type: {type(audio_data)}")
    # Pad to 480000 samples (30s at 16kHz)
    target_len = 480000
    if audio_data_np.shape[0] < target_len:
        audio_data_np = np.pad(audio_data_np, (0, target_len - audio_data_np.shape[0]), mode='constant')
    vcon_utils.set_audio(vcon, audio_data_np)
    vcon["sample_rate"] = settings.sample_rate
    # For size, use torch tensor
    audio_data_tensor = torch.from_numpy(audio_data_np)
    vcon["size"] = audio.get_size(audio_data_tensor)

    # Batch for identify_languages expects a list of batches (list of list of vcons)
    vcons_batched = [[vcon]]

    # Print debug info for batch
    print("DEBUG: batch_to_audio_data output just before identify_languages:")
    for batch in vcons_batched:
        for v in batch:
            arr = vcon_utils.get_audio(v)
            print(f"  type: {type(arr)}, shape: {getattr(arr, 'shape', None)}, dtype: {getattr(arr, 'dtype', None)}")

    # Load whisper-tiny model and processor
    model, processor = ai.load_whisper_tiny()

    # Detect language
    vcons_with_lang = ai.identify_languages(vcons_batched, model, processor)
    detected_langs = vcon_utils.get_languages(vcons_with_lang[0])
    print(f"Detected language(s): {detected_langs}")

if __name__ == "__main__":
    main() 