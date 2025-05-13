import os
import random

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


if __name__ == "__main__":
    main() 