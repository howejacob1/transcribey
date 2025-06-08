import os
from gtts import gTTS
import numpy as np
import soundfile as sf

# Directory to save wav files
output_dir = os.path.expanduser('~/conserver/fake_wavs_cute')
os.makedirs(output_dir, exist_ok=True)

# 10 different Spanish sentences (can be anything cute or random)
sentences = [
    "Hola, ¿cómo estás? Espero que tengas un día maravilloso.",
    "Los gatitos pequeños juegan en el jardín bajo el sol.",
    "Me encanta escuchar música mientras camino por el parque.",
    "Las estrellas brillan en el cielo cada noche.",
    "Hoy es un buen día para aprender algo nuevo.",
    "Las flores en primavera tienen colores muy bonitos.",
    "Un abrazo grande para ti, amigo mío.",
    "El chocolate caliente es perfecto en días fríos.",
    "Los pájaros cantan canciones alegres por la mañana.",
    "Nunca dejes de soñar y sonreír cada día."
]

# Target duration in seconds
TARGET_DURATION = 20
SAMPLE_RATE = 22050  # gTTS default sample rate is 22050 Hz

for i, text in enumerate(sentences):
    tts = gTTS(text=text, lang='es')
    mp3_path = os.path.join(output_dir, f'spanish_{i+1}.mp3')
    wav_path = os.path.join(output_dir, f'spanish_{i+1}.wav')
    tts.save(mp3_path)

    # Convert mp3 to wav using soundfile (via numpy)
    # gTTS saves as mp3, so we need to decode it
    # Use pydub for mp3 to wav conversion
    from pydub import AudioSegment
    audio = AudioSegment.from_mp3(mp3_path)
    audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1)
    audio.export(wav_path, format="wav")

    # Load wav and pad if needed
    data, sr = sf.read(wav_path)
    current_duration = len(data) / sr
    if current_duration < TARGET_DURATION:
        pad_length = int((TARGET_DURATION - current_duration) * sr)
        data = np.concatenate([data, np.zeros(pad_length, dtype=data.dtype)])
        sf.write(wav_path, data, sr)
    # Optionally, remove the mp3 file
    os.remove(mp3_path)

print(f"Generated 10 Spanish wav files in {output_dir}") 