# Assist mode for voice replies and screen sharing.
#
# Rollback: set ASSIST_MODES_ENABLED to False. The radios disappear and the
# app stays in transcription-only behavior.

import os

ASSIST_MODES_ENABLED = True

MODE_TRANSCRIPTION = "transcription"
MODE_VOICE = "voice"
MODE_WINDOW = "window"
MODE_SCREEN = "screen"

DEFAULT_ASSIST_MODE = MODE_TRANSCRIPTION

ASSIST_MODES = (
    (MODE_TRANSCRIPTION, "Transcription"),
    (MODE_VOICE, "Voice"),
    (MODE_WINDOW, "Window"),
    (MODE_SCREEN, "Screen"),
)

VALID_ASSIST_MODES = {mode for mode, _label in ASSIST_MODES}

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


def speaks(mode: str) -> bool:
    """Voice, Window, and Screen read the reply aloud."""
    return ASSIST_MODES_ENABLED and mode in (MODE_VOICE, MODE_WINDOW, MODE_SCREEN)


def captures(mode: str) -> bool:
    """Window and Screen attach a screenshot to the reply."""
    return ASSIST_MODES_ENABLED and mode in (MODE_WINDOW, MODE_SCREEN)


def guide_active(mode: str) -> bool:
    return speaks(mode)


def normalize_assist_mode(mode: str | None) -> str:
    if not ASSIST_MODES_ENABLED:
        return MODE_TRANSCRIPTION
    if mode in VALID_ASSIST_MODES:
        return mode
    return DEFAULT_ASSIST_MODE


def assist_mode_path(conf_dir: str = CONF_DIR) -> str:
    return os.path.join(conf_dir, ASSIST_MODE_FILE)


def load_assist_mode(conf_dir: str = CONF_DIR) -> str:
    """Persisted radio selection. Missing or invalid values use the default."""
    if not ASSIST_MODES_ENABLED:
        return MODE_TRANSCRIPTION

    path = assist_mode_path(conf_dir)
    if not os.path.exists(path):
        return DEFAULT_ASSIST_MODE

    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, value = (part.strip() for part in stripped.split("=", 1))
                if key.lower() == "mode":
                    return normalize_assist_mode(value)
    except OSError as exc:
        print(f"Error loading assist mode from {path}: {exc}")

    return DEFAULT_ASSIST_MODE


def save_assist_mode(mode: str, conf_dir: str = CONF_DIR) -> None:
    """Save the radio selection. No-op when the feature is rolled back."""
    if not ASSIST_MODES_ENABLED:
        return

    normalized = normalize_assist_mode(mode)
    os.makedirs(conf_dir, exist_ok=True)
    path = assist_mode_path(conf_dir)
    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("# Assist mode radio selection.\n")
            handle.write("# mode is one of: transcription, voice, window, screen\n")
            handle.write(f"mode = {normalized}\n")
    except OSError as exc:
        print(f"Error saving assist mode to {path}: {exc}")
