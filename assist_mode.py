# Assist controls for the original analysis, spoken replies, and screen sharing.
#
# Rollback: set ASSIST_MODES_ENABLED to False. The controls disappear and the
# app stays in transcription-only behavior.

import os
from dataclasses import dataclass

ASSIST_MODES_ENABLED = True

VISION_OFF = "off"
VISION_WINDOW = "window"
VISION_SCREEN = "screen"

VISION_CHOICES = (
    (VISION_OFF, "Off"),
    (VISION_WINDOW, "Window"),
    (VISION_SCREEN, "Screen"),
)

DEFAULT_TRANSCRIPTION = True
DEFAULT_VOICE = False
DEFAULT_VISION = VISION_OFF

GUIDE_INSTRUCTIONS = (
    "You guide a person who is looking at their own computer. "
    "The reply will be read aloud, so use short spoken sentences. "
    "Give only the next step. Name the control you can see. "
    "Then stop and wait for them to do it. "
    "You cannot click, type, or move the pointer. "
    "If a screenshot is included, base the step on what is visible. "
    "If capture failed, say so in one sentence and answer from their words."
)

GUIDE_MAX_OUTPUT_TOKENS = 400
GUIDE_TRANSCRIPT_CHARS = 2000

CONF_DIR = "conf"
ASSIST_MODE_FILE = "assist_mode.conf"


@dataclass(frozen=True)
class AssistSettings:
    transcription: bool = DEFAULT_TRANSCRIPTION
    voice: bool = DEFAULT_VOICE
    vision: str = DEFAULT_VISION


def speaks(settings: AssistSettings) -> bool:
    """Voice reads the guide reply aloud."""
    return ASSIST_MODES_ENABLED and settings.voice


def captures(settings: AssistSettings) -> bool:
    """Window and Screen attach a screenshot."""
    return ASSIST_MODES_ENABLED and settings.vision in (VISION_WINDOW, VISION_SCREEN)


def guide_active(settings: AssistSettings) -> bool:
    """A guide turn runs when a reply will be spoken or a screen is shared."""
    return ASSIST_MODES_ENABLED and (settings.voice or captures(settings))


def normalize_vision(vision: str | None) -> str:
    if vision in (VISION_OFF, VISION_WINDOW, VISION_SCREEN):
        return vision
    return DEFAULT_VISION


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def assist_mode_path(conf_dir: str = CONF_DIR) -> str:
    return os.path.join(conf_dir, ASSIST_MODE_FILE)


def load_assist_settings(conf_dir: str = CONF_DIR) -> AssistSettings:
    """Persisted controls. Missing values use the defaults."""
    if not ASSIST_MODES_ENABLED:
        return AssistSettings()

    path = assist_mode_path(conf_dir)
    if not os.path.exists(path):
        return AssistSettings()

    transcription = DEFAULT_TRANSCRIPTION
    voice = DEFAULT_VOICE
    vision = DEFAULT_VISION
    legacy_mode = None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, value = (part.strip() for part in stripped.split("=", 1))
                key = key.lower()
                if key == "transcription":
                    transcription = _parse_bool(value)
                elif key == "voice":
                    voice = _parse_bool(value)
                elif key == "vision":
                    vision = normalize_vision(value)
                elif key == "mode":
                    legacy_mode = value
    except OSError as exc:
        print(f"Error loading assist settings from {path}: {exc}")
        return AssistSettings()

    if legacy_mode and "transcription" not in _keys_in(path):
        transcription, voice, vision = _from_legacy_mode(legacy_mode)

    return AssistSettings(transcription, voice, normalize_vision(vision))


def _keys_in(path: str) -> set[str]:
    keys = set()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                keys.add(stripped.split("=", 1)[0].strip().lower())
    except OSError:
        return set()
    return keys


def _from_legacy_mode(mode: str) -> tuple[bool, bool, str]:
    """Old single radio. Voice and vision were exclusive with transcription."""
    if mode == "voice":
        return False, True, VISION_OFF
    if mode == VISION_WINDOW:
        return False, True, VISION_WINDOW
    if mode == VISION_SCREEN:
        return False, True, VISION_SCREEN
    return True, False, VISION_OFF


def save_assist_settings(settings: AssistSettings, conf_dir: str = CONF_DIR) -> None:
    """Save the checkboxes and vision radio. No-op when the feature is rolled back."""
    if not ASSIST_MODES_ENABLED:
        return

    os.makedirs(conf_dir, exist_ok=True)
    path = assist_mode_path(conf_dir)
    transcription = "true" if settings.transcription else "false"
    voice = "true" if settings.voice else "false"
    vision = normalize_vision(settings.vision)
    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("# Assist controls. Transcription and voice can both be on.\n")
            handle.write("# vision is one of: off, window, screen\n")
            handle.write(f"transcription = {transcription}\n")
            handle.write(f"voice = {voice}\n")
            handle.write(f"vision = {vision}\n")
    except OSError as exc:
        print(f"Error saving assist settings to {path}: {exc}")
