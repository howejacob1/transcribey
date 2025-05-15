import openai
import toml
import os
import time
import random
import string
import concurrent.futures
import transcription_models

# Load OpenAI API key from .secrets.toml
secrets = toml.load(".secrets.toml")
openai.api_key = secrets["openai"]["api_key"]

VOICES = [
    "nova", "shimmer", "echo", "onyx", "fable", "alloy", "ash", "sage", "coral"
]

def generate_conversation():
    prompt = (
        "Generate a short, random topic conversation between two people. "
        "Maybe Include pauses in the conversation, denoted by one or more periods (e.g., '....'). "
        "Do not use speaker labels. The conversation should be about 50-100 words."
        "Choose a random language for the conversation. 95%% are english,"
        "4%% are spanish, 1%% are french, and 1%% are german. If the conversation"
        "is in spanish, then include a mixture of english and spanish words."
        "also make the conversation double the length."
        "Do not use emojis. Do not end the conversation early if it goes"
        " on longer than 100 words."
    )
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=1.0,
    )
    return response.choices[0].message.content.strip()

def synthesize_wav(text, output_path, voice="onyx"):
    response = openai.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text,
        response_format="wav"
    )
    with open(output_path, "wb") as f:
        f.write(response.content)
    print(f"WAV file saved to {output_path}")

def generate_one_wav(i):
    conversation = generate_conversation()
    print(f"Conversation {i+1}:\n{conversation}\n")
    timestamp = int(time.time() * 1000)
    rand_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    voice = random.choice(VOICES)
    output_path = f"fake_wavs/openai_fake_conversation_{timestamp}_{rand_suffix}.wav"
    print(f"Using voice: {voice}")
    while True:
        try:
            synthesize_wav(conversation, output_path, voice=voice)
            break
        except Exception as e:
            print(f"Error generating wav: {e}. Retrying...")
            time.sleep(2)

def main():
    os.makedirs("fake_wavs", exist_ok=True)
    print("Loading nvidia/parakeet-tdt_ctc-110m ...")
    parakeet_model = transcription_models.load_nvidia_parakeet_tdt_ctc_110m()
    print("Loaded nvidia/parakeet-tdt_ctc-110m.")
    print("Loading nvidia/canary-1b-flash ...")
    canary_model = transcription_models.load_nvidia_canary_1b_flash()
    print("Loaded nvidia/canary-1b-flash.")
    print("Loading openai/whisper-tiny ...")
    whisper_tiny_model, whisper_tiny_processor, whisper_tiny_device = transcription_models.load_openai_whisper_tiny()
    print("Loaded openai/whisper-tiny.")
    total = 100
    batch_size = 20
    for batch_start in range(0, total, batch_size):
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = [executor.submit(generate_one_wav, i) for i in range(batch_start, min(batch_start+batch_size, total))]
            concurrent.futures.wait(futures)

if __name__ == "__main__":
    main() 