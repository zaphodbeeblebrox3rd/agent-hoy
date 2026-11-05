# Spoken replies. Playback uses PyAudio, which the app already uses for the mic.

import io
import re
import time
import wave

TTS_MODEL = "gpt-4o-mini-tts"
# Rollback: set False to send speech through the same client as the selected chat model.
TTS_USES_OPENAI = True
# Rollback: change these two constants. coral is the previous flat voice.
TTS_VOICE = "marin"
TTS_INSTRUCTIONS = (
    "Speak as a calm, intelligent person in a real conversation. "
    "Use a natural pace and ordinary intonation. "
    "Sound attentive and clear, not like an announcer or a character."
)
MAX_SPOKEN_CHARS = 1200
# Rollback: set False to speak the whole on-screen reply.
SPEAK_BRIEF_ANSWER = True
# A further sentence is spoken only when the opening still fits in this length.
BRIEF_SPOKEN_CHARS = 120
BRIEF_SPOKEN_SENTENCES = 1
# Rollback: raise this to speak a longer line as its own part.
BRIEF_PART_CHARS = 80
# Rollback: set 0 to speak the parts with no gap.
PART_PAUSE_SECONDS = 2.0


def _clean_markup(text: str) -> str:
    without_ticks = text.replace("`", " ")
    without_markup = re.sub(r"[#*_>\[\]]+", " ", without_ticks)
    return re.sub(r"\s+", " ", without_markup).strip()


def _complete_sentences(text: str) -> list[str]:
    parts = re.findall(r".+?(?:[.!?]+|$)", text)
    return [part.strip() for part in parts if part.strip()]


def text_for_speech(text: str) -> str:
    """Strip markup so the voice does not read formatting characters."""
    collapsed = _clean_markup(text)
    if len(collapsed) <= MAX_SPOKEN_CHARS:
        return collapsed
    clipped = collapsed[:MAX_SPOKEN_CHARS]
    sentence_end = max(clipped.rfind(". "), clipped.rfind("? "), clipped.rfind("! "))
    if sentence_end > 200:
        return clipped[: sentence_end + 1].strip()
    return clipped.strip()


def brief_for_speech(text: str) -> str:
    """The opening answer, in whole sentences. Later paragraphs stay on screen."""
    if not SPEAK_BRIEF_ANSWER:
        return text_for_speech(text)
    raw = (text or "").replace("\r\n", "\n").strip()
    first = raw.split("\n\n", 1)[0]
    sentences = _complete_sentences(_clean_markup(first))
    chosen: list[str] = []
    for sentence in sentences:
        if not chosen:
            chosen.append(sentence)
            continue
        if len(chosen) >= BRIEF_SPOKEN_SENTENCES:
            break
        if len(" ".join(chosen)) + 1 + len(sentence) > BRIEF_SPOKEN_CHARS:
            break
        chosen.append(sentence)
    return " ".join(chosen).strip()


def _numbered_parts(text: str) -> list[str]:
    matches = list(re.finditer(r"(?:^|\s)(\d{1,2}[.)]\s+)", text))
    if len(matches) < 2:
        return []
    parts = []
    for index, match in enumerate(matches):
        start = match.start(1)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        piece = text[start:end].strip()
        if piece:
            parts.append(piece)
    return parts


def speech_parts(text: str) -> list[str]:
    """Pieces to speak. Several lines or numbered items are separate parts."""
    if not text or not str(text).strip():
        return []
    if not SPEAK_BRIEF_ANSWER:
        spoken = text_for_speech(text)
        return [spoken] if spoken else []
    raw = str(text).replace("\r\n", "\n").strip()
    first = raw.split("\n\n", 1)[0]
    lines = [_clean_markup(line) for line in first.split("\n")]
    lines = [line for line in lines if line]
    if len(lines) >= 2:
        if all(len(line) <= BRIEF_PART_CHARS for line in lines):
            return lines
        short_answers = []
        for line in lines:
            if len(line) > BRIEF_PART_CHARS:
                break
            short_answers.append(line)
        if short_answers:
            return short_answers
    numbered = _numbered_parts(lines[0] if lines else "")
    if len(numbered) >= 2 and all(len(part) <= BRIEF_PART_CHARS for part in numbered):
        return numbered
    brief = brief_for_speech(text)
    return [brief] if brief else []


def wait_between_parts(seconds: float, should_stop) -> bool:
    """True when playback should stop during the gap."""
    if seconds <= 0:
        return False
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if should_stop():
            return True
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
    return bool(should_stop())


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
