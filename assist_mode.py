# Assist controls for the original analysis, spoken replies, and screen sharing.
#
# Rollback: set ASSIST_MODES_ENABLED to False. The controls disappear and the
# app stays in transcription-only behavior.

import os
from dataclasses import dataclass

ASSIST_MODES_ENABLED = True
# Rollback: set False to stop telling the guide how Live scroll already works.
DESCRIBE_APP_CONTROLS = True

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
DEFAULT_BROWSER_CONTROL = False

GUIDE_MAX_OUTPUT_TOKENS = 2000
GUIDE_TRANSCRIPT_CHARS = 2000
DIALOGUE_TURNS = 6


def guide_instructions(browser_control: bool = False, memory: bool = True, files: bool = True) -> str:
    """Spoken guide. You means the assistant. IT help is available, not assumed."""
    text = (
        "You are the assistant speaking with the person at this computer. "
        "When they say you, they mean you, the assistant. "
        "Reply as yourself in a conversation, in natural sentences that can be read aloud. "
        "Do not use a troubleshooting template, a numbered procedure, or a ticket format unless they ask for steps. "
        "Help with any subject. Technical troubleshooting is available when that is what they want. It is not the assumed topic. "
        "Use the recent conversation, what you remember, and the page or screenshot when those are present. "
        "If a screenshot or page is missing, say so in one sentence and answer from their words."
    )
    if DESCRIBE_APP_CONTROLS:
        text += (
            " This window already keeps older text and follows the newest line when Live scroll is checked. "
            "Unchecking Live scroll leaves the view where it is while new text is still added. "
            "When they ask how to use this window, answer from the controls you can see. "
            "Do not propose a new layout for something a visible control already does."
        )
    if memory:
        text += (
            " If you learn a durable fact about this person or how they like to work, end with a MEMORY: section. "
            "Put one fact on each following line. Never say MEMORY out loud."
        )
    if browser_control:
        text += (
            " You can operate an attached Edge or Chrome page. "
            "Only include an ACTIONS: section when the capture says a browser tab is attached and an action would help. "
            "One action per line, using click, type, press, scroll, or navigate. "
            "click takes the visible label, type takes the text to put in the focused field, "
            "press takes the key, scroll takes up or down, and navigate takes an http or https address. "
            "Never say ACTIONS out loud."
        )
    else:
        text += " You cannot click, type, or move the pointer. Tell them what you see and what they can do."
    if files:
        text += (
            " Attached files are context. Use them when they are present. "
            "When the user asks you to create, save, or download a file, say the file name in the reply "
            "and end with a FILE: block. The first line is FILE: followed by the file name, then the file body, "
            "then a line that is exactly ENDFILE. Do not read the file body aloud. "
            "Do not write a FILE: block unless they asked for a file."
        )
    return text


GUIDE_INSTRUCTIONS = guide_instructions(False, True)


def build_guide_prompt(
    transcript: str,
    capture_note: str,
    history: str = "",
    memory: str = "",
    page_text: str = "",
    attached: str = "",
) -> str:
    """New utterance plus recent turns, memory, attached files, and the page."""
    parts = []
    if history.strip():
        parts.append("Recent conversation:\n" + history.strip())
    parts.append("What the user just said:\n" + (transcript or "").strip())
    if memory.strip():
        parts.append("What you remember:\n" + memory.strip())
    if attached.strip():
        parts.append("Attached files:\n" + attached.strip())
    if page_text.strip():
        parts.append("Full page text:\n" + page_text.strip())
    parts.append("Capture:\n" + (capture_note or "").strip())
    return "\n\n".join(parts) + "\n"


def format_recent_turns(turns: list[tuple[str, str]]) -> str:
    """What the user said and what the assistant replied, oldest first."""
    lines = []
    for user_text, assistant_text in list(turns)[-DIALOGUE_TURNS:]:
        user_clip = (user_text or "").strip()[-GUIDE_TRANSCRIPT_CHARS:]
        assistant_clip = (assistant_text or "").strip()[-GUIDE_TRANSCRIPT_CHARS:]
        if user_clip:
            lines.append(f"User: {user_clip}")
        if assistant_clip:
            lines.append(f"Assistant: {assistant_clip}")
    return "\n".join(lines)


CONF_DIR = "conf"
ASSIST_MODE_FILE = "assist_mode.conf"


@dataclass(frozen=True)
class AssistSettings:
    transcription: bool = DEFAULT_TRANSCRIPTION
    voice: bool = DEFAULT_VOICE
    vision: str = DEFAULT_VISION
    browser_control: bool = DEFAULT_BROWSER_CONTROL


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
    browser_control = DEFAULT_BROWSER_CONTROL
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
                elif key == "browser_control":
                    browser_control = _parse_bool(value)
                elif key == "mode":
                    legacy_mode = value
    except OSError as exc:
        print(f"Error loading assist settings from {path}: {exc}")
        return AssistSettings()

    if legacy_mode and "transcription" not in _keys_in(path):
        transcription, voice, vision = _from_legacy_mode(legacy_mode)

    return AssistSettings(
        transcription,
        voice,
        normalize_vision(vision),
        browser_control,
    )


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
    browser_control = "true" if settings.browser_control else "false"
    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("# Assist controls. Transcription and voice can both be on.\n")
            handle.write("# vision is one of: off, window, screen\n")
            handle.write("# browser_control acts in an attached Edge or Chrome tab\n")
            handle.write(f"transcription = {transcription}\n")
            handle.write(f"voice = {voice}\n")
            handle.write(f"vision = {vision}\n")
            handle.write(f"browser_control = {browser_control}\n")
    except OSError as exc:
        print(f"Error saving assist settings to {path}: {exc}")
