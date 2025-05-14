import openai
import toml
import os
import time
import random
import string

# Load OpenAI API key from secrets.toml
secrets = toml.load("secrets.toml")
openai.api_key = secrets["openai"]["api_key"]

VOICES = [
    "nova", "shimmer", "echo", "onyx", "fable", "alloy", "ash", "sage", "coral"
]

def generate_conversation():
    prompt = (
        "Generate a short, natural-sounding conversation between two people. "
        "Include pauses in the conversation, denoted by one or more periods (e.g., '....'). "
        "Do not use speaker labels. The conversation should be about 50-100 words."
        "Choose a random language for the conversation. 95%% are english,"
        "4%% are spanish, 1%% are french, and 1%% are german."
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

def main():
    os.makedirs("fake_wavs", exist_ok=True)
    for i in range(100):
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

if __name__ == "__main__":
    main() 