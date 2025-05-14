import os
import random
from deepseek_infer import load_model, infer

def generate_fake_wav(language_code, output_dir, filename):
    """
    Generate a fake wav file for the given language code.
    This is a placeholder function. Replace with actual TTS or audio generation logic.
    """
    path = os.path.join(output_dir, filename)
    # For now, just create an empty file as a placeholder
    with open(path, 'wb') as f:
        f.write(b'')
    print(f"Generated fake wav for {language_code}: {path}")


def main():
    output_dir = 'fake_wavs'
    os.makedirs(output_dir, exist_ok=True)
    languages = ['en', 'es', 'de', 'fr', 'zh']
    for lang in languages:
        filename = f"sample_{lang}.wav"
        generate_fake_wav(lang, output_dir, filename)


def random_duration():
    """
    Generate a random number between 1 and 100, with the vast majority around 20.
    Uses a skewed distribution (e.g., exponential or normal with clipping).
    """
    # Use a normal distribution centered at 20, but clip to [1, 100]
    value = int(random.normalvariate(20, 5))
    return max(1, min(100, value))


def generate_fake_conversation(language_code, duration_seconds):
    """
    Generate a fake conversation between two people using the DeepSeek model.
    The conversation should last approximately duration_seconds, assuming 100 words per minute.
    Include long pauses in the text (e.g., [pause], ...). Do not use speaker labels.
    """
    prompt = (
        f"Generate a realistic conversation in {language_code} between two people. "
        f"The conversation should be about {duration_seconds} seconds long, perhaps with pauses (write \"...\") or "
        f" multiple long pauses. Assume the conversation is about 100 words per minute. It should be natural."
        f"Do not use speaker labels. Only use text, not emojis. Do not end the conversation early if it goes"
        f" on longer than {duration_seconds} seconds. Do not share your thoughts on the conversation."
    )
    model, tokenizer = load_model()
    conversation = infer(prompt, model, tokenizer, max_new_tokens=words_needed * 2)
    return conversation


def weighted_random_language():
    """
    Returns a language code based on the following probabilities:
    95% English ('en'), 4% Spanish ('es'), 1% for each of the rest (de, fr, zh).
    """
    r = random.random()
    if r < 0.95:
        return 'en'
    elif r < 0.99:
        return 'es'
    else:
        return random.choice(['de', 'fr', 'zh'])


if __name__ == "__main__":
    main()

    # Parameter: number of conversations to generate
    X = 5  # Change this value as needed
    print(f"\n--- Generating {X} fake conversations (weighted languages) ---\n")
    for i in range(X):
        lang = weighted_random_language()
        duration = random_duration()
        print(f"\nConversation {i+1} ({lang}, ~{duration}s):")
        conversation = generate_fake_conversation(lang, duration)
        print(conversation) 