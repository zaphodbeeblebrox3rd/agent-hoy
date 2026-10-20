# UI preferences such as toolbar collapse.
#
# Rollback: set TOOLBAR_COLLAPSE_ENABLED to False. The toggle disappears and
# the full toolbar stays visible.

import os
from typing import Optional

CONF_DIR = "conf"
UI_CONFIG_FILE = "ui.conf"

# Rollback: set False to remove collapse/expand and always show the full toolbar.
TOOLBAR_COLLAPSE_ENABLED = True
DEFAULT_TOOLBAR_COLLAPSED = False


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def ui_config_path(conf_dir: str = CONF_DIR) -> str:
    return os.path.join(conf_dir, UI_CONFIG_FILE)


def _resolve_ui_config_path(conf_dir: str = CONF_DIR) -> str:
    path = ui_config_path(conf_dir)
    if os.path.exists(path):
        return path
    example_path = f"{path}.example"
    if os.path.exists(example_path):
        return example_path
    return path


def load_toolbar_collapsed(conf_dir: str = CONF_DIR) -> bool:
    """Load whether the toolbar starts collapsed."""
    if not TOOLBAR_COLLAPSE_ENABLED:
        return False

    path = _resolve_ui_config_path(conf_dir)
    if not os.path.exists(path):
        return DEFAULT_TOOLBAR_COLLAPSED

    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, value = (part.strip() for part in stripped.split("=", 1))
                if key.lower() == "toolbar_collapsed":
                    return _parse_bool(value)
    except OSError as exc:
        print(f"Error loading UI prefs from {path}: {exc}")

    return DEFAULT_TOOLBAR_COLLAPSED


def save_toolbar_collapsed(collapsed: bool, conf_dir: str = CONF_DIR) -> None:
    """Persist toolbar collapsed state to conf/ui.conf."""
    if not TOOLBAR_COLLAPSE_ENABLED:
        return

    os.makedirs(conf_dir, exist_ok=True)
    path = ui_config_path(conf_dir)
    existing: dict[str, str] = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#") or "=" not in stripped:
                        continue
                    key, value = (part.strip() for part in stripped.split("=", 1))
                    existing[key.lower()] = value
        except OSError:
            existing = {}

    existing["toolbar_collapsed"] = "true" if collapsed else "false"

    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("# UI preferences (gitignored). Copy from ui.conf.example to customize.\n")
            handle.write(f"toolbar_collapsed = {existing['toolbar_collapsed']}\n")
            for key, value in existing.items():
                if key == "toolbar_collapsed":
                    continue
                handle.write(f"{key} = {value}\n")
    except OSError as exc:
        print(f"Error saving UI prefs to {path}: {exc}")
