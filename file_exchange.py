# Files the user attaches for context, and files the assistant writes on request.
#
# Rollback: set FILE_EXCHANGE_ENABLED to False. The buttons disappear,
# attached files are not sent, and FILE: blocks are not saved.

import os
import re

FILE_EXCHANGE_ENABLED = True
OUTPUT_DIR = "output"
MAX_ATTACHED_FILES = 8
MAX_UPLOAD_BYTES = 1_000_000
MAX_CONTEXT_CHARS = 16_000
MAX_CONTEXT_TOTAL = 48_000
MAX_OUTPUT_FILES = 3
MAX_OUTPUT_CHARS = 100_000

_NAME_CHARS = re.compile(r"[^A-Za-z0-9._ -]+")


def safe_output_name(name: str) -> str:
    """A single file name with no directory."""
    base = os.path.basename((name or "").replace("\\", "/")).strip().rstrip(".")
    if base in ("", ".", ".."):
        return ""
    cleaned = _NAME_CHARS.sub("_", base).strip()
    if cleaned in ("", ".", ".."):
        return ""
    return cleaned[:80]


def load_context_file(path: str) -> str:
    """Text of one attached file, or a short note when it cannot be read."""
    name = os.path.basename(path)
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        return f"File: {name}\nCould not be read ({exc})."
    if size > MAX_UPLOAD_BYTES:
        return f"File: {name}\nToo large to attach ({size} bytes)."
    try:
        with open(path, "rb") as handle:
            data = handle.read(MAX_UPLOAD_BYTES + 1)
    except OSError as exc:
        return f"File: {name}\nCould not be read ({exc})."
    if b"\x00" in data[:8000]:
        return f"File: {name}\nBinary file, not included as text."
    text = data.decode("utf-8", errors="replace")
    note = ""
    if len(text) > MAX_CONTEXT_CHARS:
        text = text[:MAX_CONTEXT_CHARS]
        note = "\n[truncated]"
    return f"File: {name}\n{text}{note}"


def format_attached_context(blocks: list[str]) -> str:
    if not FILE_EXCHANGE_ENABLED or not blocks:
        return ""
    kept = []
    total = 0
    for block in blocks[:MAX_ATTACHED_FILES]:
        room = MAX_CONTEXT_TOTAL - total
        if room <= 0:
            break
        piece = block[:room]
        kept.append(piece)
        total += len(piece)
    return "\n\n".join(kept)


def write_output_files(
    files: list[tuple[str, str]],
    directory: str = OUTPUT_DIR,
) -> list[str]:
    """Save requested files. Returns the paths that were written."""
    if not FILE_EXCHANGE_ENABLED or not files:
        return []
    os.makedirs(directory, exist_ok=True)
    written = []
    for name, body in files[:MAX_OUTPUT_FILES]:
        safe = safe_output_name(name)
        if not safe:
            continue
        text = body[:MAX_OUTPUT_CHARS]
        path = _unique_path(os.path.join(directory, safe))
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            if text and not text.endswith("\n"):
                handle.write("\n")
        written.append(path)
    return written


def _unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    number = 2
    while True:
        candidate = f"{root}-{number}{ext}"
        if not os.path.exists(candidate):
            return candidate
        number += 1
