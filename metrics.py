"""Stage timings for slow turns.

Each finished turn appends one JSON line to metrics/turns.jsonl.
The line has a local timestamp, the total, and each stage in milliseconds.
Notes stay short: sizes and labels, not the words that were spoken or sent.

Rollback: set METRICS_ENABLED to False. Recording stops. The log file stays.
"""

import json
import os
import threading
import time
from datetime import datetime

METRICS_ENABLED = True
METRICS_DIR = "metrics"
METRICS_FILE = "turns.jsonl"
_MAX_NOTE = 160
_lock = threading.Lock()


class TurnTimer:
    """Collect stage durations for one guide, speech, transcription, or analysis turn."""

    def __init__(self, kind: str):
        self.kind = kind
        self._started = time.perf_counter()
        self._wall = datetime.now().astimezone().isoformat(timespec="seconds")
        self.stages: list[dict] = []
        self._name = ""
        self._mark = 0.0
        self._open = False
        self._finished = False

    def start(self, name: str) -> None:
        self.stop()
        self._name = name
        self._mark = time.perf_counter()
        self._open = True

    def stop(self, note: str = "") -> None:
        if not self._open:
            return
        elapsed_ms = int(round((time.perf_counter() - self._mark) * 1000))
        self.stages.append(
            {
                "stage": self._name,
                "ms": elapsed_ms,
                "note": _short_note(note),
            }
        )
        self._open = False

    def finish(self, note: str = "") -> None:
        """Write one line. A second call does not write again."""
        if self._finished:
            return
        self.stop()
        self._finished = True
        if not METRICS_ENABLED:
            return
        total_ms = int(round((time.perf_counter() - self._started) * 1000))
        record = {
            "at": self._wall,
            "kind": self.kind,
            "total_ms": total_ms,
            "note": _short_note(note),
            "stages": self.stages,
        }
        line = json.dumps(record, ensure_ascii=True)
        try:
            os.makedirs(METRICS_DIR, exist_ok=True)
            path = os.path.join(METRICS_DIR, METRICS_FILE)
            with _lock:
                with open(path, "a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
        except OSError as exc:
            print(f"Could not save timing: {exc}")
            return
        parts = " ".join(f"{item['stage']}={item['ms']}" for item in self.stages)
        print(f"Timing {self.kind} {total_ms}ms {parts}".rstrip())


def _short_note(note: str) -> str:
    return " ".join(str(note or "").split())[:_MAX_NOTE]
