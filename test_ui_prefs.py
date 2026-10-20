#!/usr/bin/env python3
"""Tests for toolbar collapse UI preferences."""

import os
import tempfile

from ui_prefs import (
    TOOLBAR_COLLAPSE_ENABLED,
    load_toolbar_collapsed,
    save_toolbar_collapsed,
)


def test_defaults_and_round_trip():
    assert TOOLBAR_COLLAPSE_ENABLED is True
    with tempfile.TemporaryDirectory() as tmp_dir:
        assert load_toolbar_collapsed(tmp_dir) is False
        save_toolbar_collapsed(True, tmp_dir)
        assert load_toolbar_collapsed(tmp_dir) is True
        save_toolbar_collapsed(False, tmp_dir)
        assert load_toolbar_collapsed(tmp_dir) is False
        path = os.path.join(tmp_dir, "ui.conf")
        assert os.path.exists(path)
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        assert "toolbar_collapsed = false" in text
    print("OK toolbar collapse prefs")


if __name__ == "__main__":
    test_defaults_and_round_trip()
    print("OK all ui_prefs tests")
