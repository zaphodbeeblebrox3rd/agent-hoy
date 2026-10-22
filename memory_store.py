# Live memory for the spoken assistant.
#
# Rollback: set MEMORY_ENABLED to False. Reading and writing stop.
# The memory file is left on disk.

import os

MEMORY_ENABLED = True
MEMORY_DIR = "memory"
MEMORY_FILE = "agent_memory.md"
MAX_MEMORY_LINES = 100
MAX_MEMORY_LINE_CHARS = 240
MAX_ACTIONS = 8

_ACTION_VERBS = ("click", "type", "press", "scroll", "navigate")
_HEADER = (
    "# Live assistant memory. This folder is gitignored.\n"
    "# One fact per bullet. The assistant reloads this file on the next turn.\n\n"
)


def memory_path(directory: str = MEMORY_DIR) -> str:
    return os.path.join(directory, MEMORY_FILE)


def load_memory(directory: str = MEMORY_DIR) -> str:
    """Bullet list for the next prompt. Empty when memory is off or missing."""
    if not MEMORY_ENABLED:
        return ""
    lines = _read_lines(directory)
    if not lines:
        return ""
    return "\n".join(f"- {line}" for line in lines)


def remember(lines: list[str], directory: str = MEMORY_DIR) -> None:
    """Append new facts. Duplicates are skipped and the file stays capped."""
    if not MEMORY_ENABLED or not lines:
        return

    existing = _read_lines(directory)
    seen = {item.casefold() for item in existing}
    for line in lines:
        cleaned = _clean_fact(line)
        if not cleaned or cleaned.casefold() in seen:
            continue
        existing.append(cleaned)
        seen.add(cleaned.casefold())

    kept = existing[-MAX_MEMORY_LINES:]
    os.makedirs(directory, exist_ok=True)
    path = memory_path(directory)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(_HEADER)
        for line in kept:
            handle.write(f"- {line}\n")


def split_guide_reply(
    text: str,
) -> tuple[str, list[str], list[tuple[str, str]], list[tuple[str, str]]]:
    """Separate the spoken reply from MEMORY, ACTIONS, and FILE blocks.

    Those blocks are saved or run later. They are not shown or spoken.
    """
    from file_exchange import MAX_OUTPUT_FILES

    spoken_lines = []
    memory = []
    actions = []
    files = []
    file_name = ""
    file_lines = []
    mode = "spoken"

    def close_file():
        nonlocal file_name
        if file_name and len(files) < MAX_OUTPUT_FILES:
            body = "\n".join(file_lines).strip("\n")
            if body:
                files.append((file_name, body))
        file_name = ""
        file_lines.clear()

    for line in (text or "").splitlines():
        raw = line.strip()
        if raw.startswith("```"):
            continue
        marker = raw.upper()
        if marker == "MEMORY:" or marker == "ACTIONS:" or marker == "ENDFILE" or marker.startswith("FILE:"):
            if mode == "file":
                close_file()
        if marker == "MEMORY:":
            mode = "memory"
            continue
        if marker == "ACTIONS:":
            mode = "actions"
            continue
        if marker == "ENDFILE":
            mode = "spoken"
            continue
        if marker.startswith("FILE:"):
            name = raw.split(":", 1)[1].strip()
            if name:
                file_name = name
                mode = "file"
            continue
        if mode == "spoken":
            spoken_lines.append(line)
            continue
        if mode == "file":
            file_lines.append(line)
            continue
        if not raw:
            continue
        if raw[:1] in "-*":
            raw = raw[1:].strip()
        if not raw:
            continue
        if mode == "memory":
            fact = _clean_fact(raw)
            if fact:
                memory.append(fact)
            continue
        action = _parse_action(raw)
        if action and len(actions) < MAX_ACTIONS:
            actions.append(action)
    if mode == "file":
        close_file()
    spoken = "\n".join(spoken_lines).strip()
    return spoken, memory, actions, files


def visible_reply(text: str) -> str:
    """Reply text with memory, browser actions, and file bodies removed."""
    spoken, _memory, _actions, _files = split_guide_reply(text)
    return spoken


def _clean_fact(line: str) -> str:
    cleaned = (line or "").strip()
    if cleaned[:1] in "-*":
        cleaned = cleaned[1:].strip()
    cleaned = " ".join(cleaned.split())
    if not cleaned or cleaned.upper() in ("MEMORY:", "ACTIONS:", "ENDFILE") or cleaned.upper().startswith("FILE:"):
        return ""
    return cleaned[:MAX_MEMORY_LINE_CHARS]


def _parse_action(line: str) -> tuple[str, str] | None:
    if ":" not in line:
        return None
    verb, argument = line.split(":", 1)
    verb = verb.strip().lower()
    argument = argument.strip()
    if verb not in _ACTION_VERBS or not argument:
        return None
    return verb, argument


def _read_lines(directory: str) -> list[str]:
    path = memory_path(directory)
    if not os.path.exists(path):
        return []
    lines = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                fact = _clean_fact(line)
                if not fact or fact.startswith("#"):
                    continue
                lines.append(fact)
    except OSError as exc:
        print(f"Error loading memory from {path}: {exc}")
        return []
    return lines[-MAX_MEMORY_LINES:]
