"""Load keyword, question, and topic configuration from conf/*.conf files."""

from __future__ import annotations

import os
import re
from typing import Dict, List

CONF_DIR = "conf"


def _resolve_config_path(filename: str) -> str:
    """Prefer conf/<name>.conf override; fall back to conf/<name>.conf.example."""
    config_path = os.path.join(CONF_DIR, filename)
    if os.path.exists(config_path):
        return config_path
    example_path = f"{config_path}.example"
    if os.path.exists(example_path):
        return example_path
    return config_path


def _read_config_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.readlines()


def load_question_patterns(config_file: str = "question_patterns.conf") -> List[str]:
    """Load question-detection regex patterns (one per non-comment line)."""
    path = _resolve_config_path(config_file)
    patterns: List[str] = []

    if not os.path.exists(path):
        print(f"Warning: question patterns file not found: {path}")
        return patterns

    for line_num, line in enumerate(_read_config_lines(path), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        patterns.append(stripped)

    print(f"Loaded {len(patterns)} question patterns from {path}")
    return patterns


def load_question_type_patterns(
    config_file: str = "question_type_patterns.conf",
) -> Dict[str, List[str]]:
    """Load question-type regex patterns grouped by [section] headers."""
    path = _resolve_config_path(config_file)
    patterns: Dict[str, List[str]] = {}
    current_section: str | None = None

    if not os.path.exists(path):
        print(f"Warning: question type patterns file not found: {path}")
        return patterns

    for line_num, line in enumerate(_read_config_lines(path), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("[") and stripped.endswith("]"):
            current_section = stripped[1:-1].strip().lower()
            patterns.setdefault(current_section, [])
            continue

        if current_section is None:
            print(
                f"Warning: pattern outside section in {path} line {line_num}: {stripped}"
            )
            continue

        patterns[current_section].append(stripped)

    total = sum(len(values) for values in patterns.values())
    print(f"Loaded {total} question type patterns in {len(patterns)} categories from {path}")
    return patterns


def load_tech_keywords(config_file: str = "tech_keywords.conf") -> Dict[str, List[str]]:
    """Load tech keywords grouped by [section]; comma-separated or one per line."""
    path = _resolve_config_path(config_file)
    keywords: Dict[str, List[str]] = {}
    current_section: str | None = None

    if not os.path.exists(path):
        print(f"Warning: tech keywords file not found: {path}")
        return keywords

    for line_num, line in enumerate(_read_config_lines(path), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("[") and stripped.endswith("]"):
            current_section = stripped[1:-1].strip().lower()
            keywords.setdefault(current_section, [])
            continue

        if current_section is None:
            print(
                f"Warning: keyword outside section in {path} line {line_num}: {stripped}"
            )
            continue

        if "," in stripped:
            for item in stripped.split(","):
                item = item.strip()
                if item:
                    keywords[current_section].append(item)
        else:
            keywords[current_section].append(stripped)

    total = sum(len(values) for values in keywords.values())
    print(f"Loaded {total} tech keywords in {len(keywords)} categories from {path}")
    return keywords


_TOPIC_FIELD_RE = re.compile(r"^@([a-z_]+)\s*$", re.IGNORECASE)
_TOPIC_SECTION_RE = re.compile(r"^\[([a-z0-9_]+)\]\s*$", re.IGNORECASE)


def load_topic_explanations(
    config_file: str = "topic_explanations.conf",
) -> Dict[str, Dict[str, str]]:
    """
    Load topic explanations from a sectioned config file.

    Format:
        [category]
        @title
        Title text
        @summary
        Summary paragraph
        @challenges
        Bullet list...
        @commands
        Command list...
        @boot_process   (optional)
        Long-form content...
    """
    path = _resolve_config_path(config_file)
    topics: Dict[str, Dict[str, str]] = {}

    if not os.path.exists(path):
        print(f"Warning: topic explanations file not found: {path}")
        return topics

    current_category: str | None = None
    current_field: str | None = None
    field_lines: List[str] = []

    def flush_field() -> None:
        nonlocal current_field, field_lines
        if current_category and current_field and field_lines:
            topics[current_category][current_field] = "\n".join(field_lines).strip()
        current_field = None
        field_lines = []

    for line_num, raw_line in enumerate(_read_config_lines(path), 1):
        line = raw_line.rstrip("\n\r")
        stripped = line.strip()

        if not stripped or stripped.startswith("#"):
            continue

        section_match = _TOPIC_SECTION_RE.match(stripped)
        if section_match:
            flush_field()
            current_category = section_match.group(1).lower()
            topics.setdefault(current_category, {})
            continue

        field_match = _TOPIC_FIELD_RE.match(stripped)
        if field_match:
            flush_field()
            current_field = field_match.group(1).lower()
            continue

        if current_category is None or current_field is None:
            print(f"Warning: unexpected content in {path} line {line_num}: {stripped}")
            continue

        field_lines.append(line)

    flush_field()

    print(f"Loaded {len(topics)} topic explanations from {path}")
    return topics


def _default_transcription_corrections() -> Dict[str, str]:
    """Minimal fallback when conf/corrections.conf is unavailable."""
    return {
        "computer environment": "compute environment",
        "on premise": "on-premise",
        "camera": "containerized",
        "in Fennaband": "Infiniband",
        "Fennaband": "Infiniband",
        "created storage later": "federated storage layers",
        "storage later": "storage layers",
        "bare metal": "bare-metal",
        "split brain": "split-brain",
        "cloud burst": "cloud-burst",
        "open source": "open-source",
        "fault tolerant": "fault-tolerant",
        "Design of": "Design a",
        "for us": "for a",
        "or solution": "Your solution",
        "and show": "and ensure",
        "This includes": "for specialized hardware",
    }


def load_transcription_corrections(
    config_file: str = "corrections.conf",
) -> Dict[str, str]:
    """Load transcription corrections (original = corrected) from conf/."""
    path = _resolve_config_path(config_file)
    corrections: Dict[str, str] = {}

    try:
        if not os.path.exists(path):
            print(f"Corrections config file {path} not found, using default corrections")
            return _default_transcription_corrections()

        for line_num, line in enumerate(_read_config_lines(path), 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            if "=" not in stripped:
                print(f"Warning: Invalid correction format in {path} line {line_num}: {stripped}")
                continue

            original, corrected = (part.strip() for part in stripped.split("=", 1))
            if original and corrected:
                corrections[original] = corrected
            else:
                print(f"Warning: Invalid correction format in {path} line {line_num}: {stripped}")

        print(f"Loaded {len(corrections)} transcription corrections from {path}")
    except Exception as exc:
        print(f"Error loading corrections from {path}: {exc}")
        return _default_transcription_corrections()

    return corrections
