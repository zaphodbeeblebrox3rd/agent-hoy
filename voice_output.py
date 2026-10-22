# Spoken replies. Playback uses PyAudio, which the app already uses for the mic.

import io
import re
import wave

TTS_MODEL = "gpt-4o-mini-tts"
# Rollback: change these two constants. coral is the previous flat voice.
TTS_VOICE = "marin"
TTS_INSTRUCTIONS = (
    "Speak as a calm, intelligent person in a real conversation. "
    "Use a natural pace and ordinary intonation. "
    "Sound attentive and clear, not like an announcer or a character."
)
MAX_SPOKEN_CHARS = 1200


def text_for_speech(text: str) -> str:
    """Strip markup so the voice does not read formatting characters."""
    without_ticks = text.replace("`", " ")
    without_markup = re.sub(r"[#*_>\[\]]+", " ", without_ticks)
    collapsed = re.sub(r"\s+", " ", without_markup).strip()
    if len(collapsed) <= MAX_SPOKEN_CHARS:
        return collapsed
    clipped = collapsed[:MAX_SPOKEN_CHARS]
    sentence_end = max(clipped.rfind(". "), clipped.rfind("? "), clipped.rfind("! "))
    if sentence_end > 200:
        return clipped[: sentence_end + 1].strip()
    return clipped.strip()


def synthesize_speech(client, text: str) -> bytes:
    """Return WAV bytes for text. Raises if the speech API fails."""
    spoken = text_for_speech(text)
    if not spoken:
        return b""

    response = client.audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=spoken,
        instructions=TTS_INSTRUCTIONS,
        response_format="wav",
    )
    return response.read()


def play_wav_bytes(wav_bytes: bytes, should_stop) -> None:
    """Play WAV audio. should_stop() cuts playback off between chunks."""
    if not wav_bytes:
        return

    import pyaudio

    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        player = pyaudio.PyAudio()
        stream = None
        try:
            stream = player.open(
                format=player.get_format_from_width(wav_file.getsampwidth()),
                channels=wav_file.getnchannels(),
                rate=wav_file.getframerate(),
                output=True,
            )
            chunk = 1024
            frames = wav_file.readframes(chunk)
            while frames:
                if should_stop():
                    break
                stream.write(frames)
                frames = wav_file.readframes(chunk)
        finally:
            if stream is not None:
                stream.stop_stream()
                stream.close()
            player.terminate()
