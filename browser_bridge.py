# Attach to the Edge or Chrome window the user is sharing.
#
# A browser that is already open cannot be attached. It has to be started
# with a remote debugging port. Rollback: set BROWSER_CONTROL_ENABLED to
# False. The Control browser checkbox disappears and actions are not run.

import base64
import contextlib
import hashlib
import json
import os
import re
import socket
import struct
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from image_payload import MAX_ENCODED_BYTES, PreparedImage, prepare_encoded

BROWSER_CONTROL_ENABLED = True
# Rollback: set False to ignore choose. Other browser actions stay.
CHOOSE_ENABLED = True
DEBUG_PORT = 9222
PAGE_TEXT_CHARS = 20_000
WEBSOCKET_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_MAX_FRAME_BYTES = 96 * 1024 * 1024
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
_BROWSER_SUFFIXES = (
    " - google chrome",
    " - microsoft edge",
    " — google chrome",
    " — microsoft edge",
    " – google chrome",
    " – microsoft edge",
)


class CdpError(Exception):
    """The local debugging connection failed."""


@dataclass
class PageView:
    attached: bool
    browser: str
    title: str
    url: str
    text: str
    status: str
    websocket_url: str
    relaunch_command: str
    prepared: PreparedImage | None = None
    image_too_large: bool = False


def browser_actions_allowed(browser_control: bool) -> bool:
    """Actions run only when the feature is on and the checkbox is checked."""
    return BROWSER_CONTROL_ENABLED and bool(browser_control)


def parse_choose(argument: str) -> tuple[str, str] | None:
    """Return the sentence anchor and the answer. None when the text is empty."""
    text = " ".join((argument or "").split()).strip()
    if not text:
        return None
    if "|" in text:
        anchor, answer = text.split("|", 1)
        anchor = " ".join(anchor.split()).strip()
        answer = " ".join(answer.split()).strip()
        if not anchor or not answer:
            return None
        return anchor, answer
    return "", text


def relaunch_command(exe: str, kind: str) -> str:
    """One command that starts this browser with a debugging port."""
    if exe:
        return f'"{exe}" --remote-debugging-port={DEBUG_PORT}'
    binary = "msedge" if kind == "edge" else "chrome"
    return f"{binary} --remote-debugging-port={DEBUG_PORT}"


def debugging_status(kind: str, command: str) -> str:
    name = "Edge" if kind == "edge" else "Chrome"
    return (
        f"{name} is open without a debugging port, so the full page is not available. "
        f"Close it and start it with: {command}"
    )


def strip_window_title(title: str) -> str:
    """Drop the browser and 'more pages' suffix from a window title."""
    import re

    cleaned = (title or "").strip()
    changed = True
    while changed and cleaned:
        changed = False
        folded = cleaned.casefold()
        for suffix in _BROWSER_SUFFIXES:
            if folded.endswith(suffix):
                cleaned = cleaned[: -len(suffix)].strip()
                changed = True
                break
    cleaned = re.sub(r"\s+and\s+\d+\s+more\s+pages?$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def titles_match(window_title: str, tab_title: str) -> bool:
    window = strip_window_title(window_title).casefold()
    tab = " ".join((tab_title or "").split()).casefold()
    if not window or not tab:
        return False
    if window == tab:
        return True
    if len(tab) >= 3 and (window.startswith(tab) or tab.startswith(window)):
        return True
    return False


def page_image_for_prompt(prepared: PreparedImage) -> tuple[PreparedImage | None, bool]:
    """Keep a full-page image only when it did not have to be shrunk."""
    if prepared.resized:
        return None, True
    return prepared, False


def look_at_browser(
    window_id: int | None,
    model: str,
    *,
    include_image: bool = True,
) -> PageView | None:
    """Read the shared Edge or Chrome tab. None when the window is not one of those."""
    if not window_id:
        return None
    title, exe = _window_identity(int(window_id))
    kind = browser_kind(exe, title)
    if not kind:
        return None

    command = relaunch_command(exe, kind)
    port = find_debugging_port(kind)
    if not port:
        return _unattached(kind, title, debugging_status(kind, command), command)

    try:
        targets = list_pages(port)
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
        print(f"Browser debugging port {port} did not answer: {exc}")
        return _unattached(kind, title, debugging_status(kind, command), command)

    target = match_tab(title, targets)
    if not target:
        name = "Edge" if kind == "edge" else "Chrome"
        status = f"No open {name} tab title matched this window."
        return _unattached(kind, title, status, command)

    websocket_url = target.get("webSocketDebuggerUrl") or ""
    if not websocket_url:
        return _unattached(kind, title, debugging_status(kind, command), command)

    try:
        text, url, png, shot_too_large = read_page(
            websocket_url,
            include_image=include_image,
        )
    except (CdpError, OSError, TimeoutError, ValueError) as exc:
        print(f"Could not read the browser page: {exc}")
        return _unattached(kind, title, f"Could not read the page ({exc}).", command)

    prepared = None
    too_large = shot_too_large
    if png and not too_large:
        try:
            if len(png) > MAX_ENCODED_BYTES:
                too_large = True
            else:
                prepared, too_large = page_image_for_prompt(prepare_encoded(png, model))
        except Exception as exc:
            print(f"Full-page image was not usable: {exc}")
            too_large = True
            prepared = None

    tab_title = target.get("title") or title
    note = [f"Attached {kind} tab {tab_title}."]
    if url:
        note.append(url)
    if text and len(text) >= PAGE_TEXT_CHARS:
        note.append("Page text was truncated.")
    if too_large:
        note.append("The full-page image was too large, so only the page text is included.")
    elif prepared is not None:
        note.append(f"Full page image {prepared.width}x{prepared.height}.")
    else:
        note.append("Full page text is included. No full-page image.")

    return PageView(
        attached=True,
        browser=kind,
        title=title,
        url=url,
        text=text,
        status=" ".join(note),
        websocket_url=websocket_url,
        relaunch_command=command,
        prepared=prepared,
        image_too_large=too_large,
    )


def run_actions(websocket_url: str, actions: list[tuple[str, str]]) -> str:
    """Run click, type, press, scroll, navigate, and choose on the attached tab."""
    if not BROWSER_CONTROL_ENABLED or not actions or not websocket_url:
        return ""
    notes = []
    entered = False
    skip_entries = False
    entry_verbs = ("choose", "type", "match", "move")
    failures = ("not found", "refused", "choose is off", "not placed")
    for verb, argument in actions:
        if skip_entries and verb in entry_verbs:
            continue
        try:
            session = _pooled_session(websocket_url)
            note = _run_one(session, verb, argument)
        except (CdpError, OSError, TimeoutError, ValueError) as exc:
            _drop_session(websocket_url)
            note = f"{verb} failed: {exc}"
        notes.append(note)
        if verb in entry_verbs:
            if note.casefold() in failures and not entered:
                # Nothing was entered - refuse to run a submit on a blank answer.
                notes.append("skipped remaining selections after a failed one")
                skip_entries = True
            elif note.casefold() not in failures and not note.startswith(
                f"{verb} failed"
            ):
                entered = True
    return "; ".join(note for note in notes if note)


def browser_kind(exe: str, title: str) -> str | None:
    name = os.path.basename(exe or "").casefold()
    if name in ("msedge.exe", "msedge", "microsoft-edge"):
        return "edge"
    if name in ("chrome.exe", "chrome", "google-chrome", "google-chrome-stable"):
        return "chrome"
    folded = (title or "").casefold()
    if "microsoft edge" in folded:
        return "edge"
    if "google chrome" in folded:
        return "chrome"
    return None


def debuggable_browser_window_ids() -> list[int]:
    """Visible Edge/Chrome windows on a machine where that browser is debuggable."""
    if sys.platform != "win32":
        return []
    import ctypes
    from ctypes import wintypes

    ports = {kind: find_debugging_port(kind) for kind in ("edge", "chrome")}
    if not any(ports.values()):
        return []
    user32 = ctypes.windll.user32
    found: list[int] = []
    enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def _collect(hwnd, _lparam):
        try:
            if user32.IsWindowVisible(hwnd) and not user32.IsIconic(hwnd):
                kind = browser_kind(_windows_exe(hwnd), _windows_title(hwnd))
                if kind and ports.get(kind):
                    found.append(int(hwnd))
        except Exception:
            pass
        return True

    user32.EnumWindows(enum_proc(_collect), 0)
    return found


def find_debugging_port(kind: str) -> int | None:
    """Port from DevToolsActivePort, or 9222 when that browser is already listening."""
    candidates = []
    from_file = _port_from_file(kind)
    if from_file:
        candidates.append(from_file)
    if DEBUG_PORT not in candidates:
        candidates.append(DEBUG_PORT)
    for port in candidates:
        if _port_is_browser(port, kind):
            return port
    return None


def list_pages(port: int) -> list[dict]:
    payload = _json_get(f"http://127.0.0.1:{port}/json/list", timeout=1.5)
    if not isinstance(payload, list):
        return []
    pages = []
    for target in payload:
        if not isinstance(target, dict) or target.get("type") != "page":
            continue
        url = str(target.get("url") or "")
        if url.startswith("devtools://"):
            continue
        pages.append(target)
    return pages


def match_tab(window_title: str, targets: list[dict]) -> dict | None:
    matches = [target for target in targets if titles_match(window_title, str(target.get("title") or ""))]
    if not matches:
        return None
    exact = strip_window_title(window_title).casefold()

    def rank(target: dict) -> tuple[int, int]:
        title = " ".join(str(target.get("title") or "").split()).casefold()
        exact_score = 0 if title == exact else 1
        return exact_score, -len(title)

    matches.sort(key=rank)
    return matches[0]


def read_page(
    websocket_url: str,
    *,
    include_image: bool = True,
) -> tuple[str, str, bytes | None, bool]:
    """Return page text, url, png bytes, and whether the screenshot was too large."""
    expression = (
        "(() => {"
        " const text = document.body ? (document.body.innerText || '') : '';"
        " const vis = (el) => !!(el.offsetParent || el.getClientRects().length);"
        " const parts = [];"
        " document.querySelectorAll('select').forEach(el => {"
        "   if (!vis(el)) return;"
        "   const opts = Array.from(el.options).map(o => (o.selected ? '[x] ' : '[ ] ') + o.text.trim());"
        "   if (opts.length) parts.push('Select: ' + opts.join(' | '));"
        " });"
        " document.querySelectorAll('input, textarea').forEach(el => {"
        "   if (!vis(el)) return;"
        "   const ty = (el.type || 'text').toLowerCase();"
        "   if (ty === 'radio' || ty === 'checkbox') {"
        "     const label = (el.labels && el.labels.length ? el.labels[0].innerText : (el.getAttribute('aria-label') || el.value || '')).trim();"
        "     parts.push((ty === 'checkbox' ? 'Checkbox' : 'Radio') + ' ' + (el.checked ? '[x] ' : '[ ] ') + label);"
        "   } else if (ty === 'button' || ty === 'submit') {"
        "     const v = (el.value || '').trim();"
        "     if (v) parts.push('Button: ' + v);"
        "   } else {"
        "     parts.push('Input: ' + ((el.value || el.placeholder || '').trim() || '(empty)'));"
        "   }"
        " });"
        " document.querySelectorAll('[contenteditable=true]').forEach(el => {"
        "   if (vis(el)) parts.push('Editable: ' + (el.innerText || '').trim().slice(0, 120));"
        " });"
        " const roles = '[role=\"option\"],[role=\"radio\"],[role=\"checkbox\"],[role=\"menuitemradio\"],[role=\"menuitemcheckbox\"]';"
        " document.querySelectorAll(roles).forEach(el => {"
        "   if (!vis(el)) return;"
        "   const st = el.getAttribute('aria-checked') || el.getAttribute('aria-selected');"
        "   const mark = st === 'true' ? '[x] ' : (st === 'false' ? '[ ] ' : '');"
        "   const tx = (el.innerText || el.getAttribute('aria-label') || '').trim();"
        "   if (tx) parts.push(mark + tx);"
        " });"
        " const countOf = (sel) => [...document.querySelectorAll(sel)].filter(vis).length;"
        " const cb = countOf('input[type=\"checkbox\"]') + countOf('[role=\"checkbox\"]');"
        " const rb = countOf('input[type=\"radio\"]') + countOf('[role=\"radio\"]');"
        " const sl = countOf('select');"
        " const fi = countOf('input.fitb-input, textarea.fitb-input');"
        " const ord = countOf('.choice-item[draggable=\"true\"], [data-drop-target-for-element]');"
        " const mrow = countOf('.match-row');"
        " const mtile = countOf('[draggable=\"true\"]');"
        " const kinds = [];"
        " if (mrow && mtile) kinds.push('matching - drag ' + mtile + ' items onto ' + mrow + ' labeled targets, use match: <item> -> <target> per pair');"
        " if (cb > 1) kinds.push('multiple select - ' + cb + ' checkbox options, choose ALL correct');"
        " if (rb > 1) kinds.push('single choice - ' + rb + ' radio options, choose exactly ONE');"
        " if (fi) kinds.push('fill in the blank - ' + fi + ' text field' + (fi > 1 ? 's' : ''));"
        " if (sl) kinds.push('dropdown - ' + sl + ' select' + (sl > 1 ? 's' : ''));"
        " if (ord && !mrow) kinds.push('ordering - arrange ' + ord + ' items top to bottom, use move: <item> -> <item at its slot> per step');"
        " const qtype = kinds.length ? 'Question type: ' + kinds.join('; ') + '.\\n\\n' : '';"
        f" const limit = {PAGE_TEXT_CHARS};"
        " const main = text.slice(0, limit);"
        " const controls = parts.join('\\n').slice(0, 4000);"
        " return {text: qtype + main + (controls ? '\\n\\nForm controls:\\n' + controls : ''), url: location.href};"
        "})()"
    )
    session = None
    for attempt in range(2):
        try:
            session = _pooled_session(websocket_url)
            evaluated = session.call(
                "Runtime.evaluate",
                {"expression": expression, "returnByValue": True},
            )
            break
        except (OSError, CdpError):
            _drop_session(websocket_url)
            session = None
            if attempt:
                raise
    if session is None:
        raise CdpError("Debugging socket unavailable")
    with contextlib.suppress(OSError):
        value = (evaluated.get("result") or {}).get("value") or {}
        if not isinstance(value, dict):
            value = {}
        text = str(value.get("text") or "")
        url = str(value.get("url") or "")
        png = None
        too_large = False
        if include_image:
            try:
                try:
                    session.call("Page.enable")
                except CdpError:
                    pass
                shot = session.call(
                    "Page.captureScreenshot",
                    {
                        "format": "png",
                        "captureBeyondViewport": True,
                        "fromSurface": True,
                    },
                    timeout=20,
                )
                encoded = shot.get("data") or ""
                if encoded:
                    png = base64.b64decode(encoded)
            except CdpError as exc:
                if "too large" in str(exc).lower():
                    too_large = True
                png = None
        return text, url, png, too_large


class CdpSession:
    """Minimal Chrome DevTools Protocol client for a local page socket."""

    def __init__(self, websocket_url: str, timeout: float = 10.0):
        from urllib.parse import urlparse

        parsed = urlparse(websocket_url)
        host = (parsed.hostname or "").strip("[]")
        if parsed.scheme not in ("ws", "wss") or host not in _LOCAL_HOSTS:
            raise CdpError("Refusing a debugging socket that is not local")
        port = parsed.port or (443 if parsed.scheme == "wss" else 80)
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"

        self._timeout = timeout
        self._next_id = 1
        self._buffer = b""
        self._sock = socket.create_connection((host, port), timeout=timeout)
        self._sock.settimeout(timeout)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n"
        )
        self._sock.sendall(request.encode("ascii"))
        raw = self._read_until(b"\r\n\r\n")
        head, _, leftover = raw.partition(b"\r\n\r\n")
        self._buffer = leftover
        header = head.decode("iso-8859-1", errors="replace")
        status_line = header.split("\r\n", 1)[0]
        if " 101 " not in status_line:
            self.close()
            raise CdpError("Debugging socket did not accept the connection")
        accept = base64.b64encode(
            hashlib.sha1((key + WEBSOCKET_GUID).encode("ascii")).digest()
        ).decode("ascii")
        if accept.lower() not in header.lower():
            self.close()
            raise CdpError("Debugging socket handshake did not match")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    def call(self, method: str, params: dict | None = None, timeout: float | None = None) -> dict:
        import time

        message_id = self._next_id
        self._next_id += 1
        payload = {"id": message_id, "method": method}
        if params:
            payload["params"] = params
        self._send_text(json.dumps(payload))
        deadline = time.time() + (timeout or self._timeout)
        while time.time() < deadline:
            message = self._recv_text(deadline)
            if message is None:
                break
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                continue
            if data.get("id") != message_id:
                continue
            if "error" in data:
                raise CdpError(str(data["error"]))
            result = data.get("result")
            return result if isinstance(result, dict) else {}
        raise CdpError(f"{method} timed out")

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass

    def _send_text(self, text: str) -> None:
        self._sock.sendall(encode_client_frame(text))

    def _read_until(self, marker: bytes) -> bytes:
        raw = self._buffer
        self._buffer = b""
        while marker not in raw:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise CdpError("Debugging socket closed during handshake")
            raw += chunk
            if len(raw) > 65536:
                raise CdpError("Debugging handshake was too large")
        return raw

    def _recv_exact(self, size: int, deadline: float) -> bytes:
        import time

        while len(self._buffer) < size:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise CdpError("Debugging socket timed out")
            self._sock.settimeout(remaining)
            try:
                chunk = self._sock.recv(max(4096, size - len(self._buffer)))
            except socket.timeout as exc:
                raise CdpError("Debugging socket timed out") from exc
            if not chunk:
                raise CdpError("Debugging socket closed")
            self._buffer += chunk
        data = self._buffer[:size]
        self._buffer = self._buffer[size:]
        return data

    def _recv_text(self, deadline: float) -> str | None:
        fragments = []
        while True:
            header = self._recv_exact(2, deadline)
            first, second = header[0], header[1]
            fin = first & 0x80
            opcode = first & 0x0F
            masked = second & 0x80
            length = second & 0x7F
            if length == 126:
                length = struct.unpack("!H", self._recv_exact(2, deadline))[0]
            elif length == 127:
                length = struct.unpack("!Q", self._recv_exact(8, deadline))[0]
            if length > _MAX_FRAME_BYTES:
                raise CdpError("Debugging frame was too large")
            mask_key = self._recv_exact(4, deadline) if masked else b""
            payload = self._recv_exact(length, deadline) if length else b""
            if masked and mask_key:
                payload = bytes(byte ^ mask_key[index % 4] for index, byte in enumerate(payload))
            if opcode == 0x8:
                return None
            if opcode == 0x9:
                self._sock.sendall(encode_client_frame(payload.decode("utf-8", errors="replace"), opcode=0xA))
                continue
            if opcode == 0xA:
                continue
            if opcode in (0x1, 0x0):
                fragments.append(payload)
                if fin:
                    return b"".join(fragments).decode("utf-8", errors="replace")
                continue
            if fin and not fragments:
                continue


_CDP_POOL: dict[str, "CdpSession"] = {}


def _pooled_session(websocket_url: str) -> "CdpSession":
    """Reuse one debugging socket per page. Fresh sockets per call exhaust them."""
    session = _CDP_POOL.get(websocket_url)
    if session is None:
        session = CdpSession(websocket_url)
        _CDP_POOL[websocket_url] = session
    return session


def _drop_session(websocket_url: str) -> None:
    session = _CDP_POOL.pop(websocket_url, None)
    if session is not None:
        session.close()


def encode_client_frame(text: str, opcode: int = 0x1) -> bytes:
    """Masked client frame. Servers send unmasked frames back."""
    payload = text.encode("utf-8")
    mask = os.urandom(4)
    masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
    header = bytearray([0x80 | (opcode & 0x0F)])
    length = len(payload)
    if length < 126:
        header.append(0x80 | length)
    elif length < 65536:
        header.append(0x80 | 126)
        header.extend(struct.pack("!H", length))
    else:
        header.append(0x80 | 127)
        header.extend(struct.pack("!Q", length))
    header.extend(mask)
    header.extend(masked)
    return bytes(header)


def _unattached(kind: str, title: str, status: str, command: str) -> PageView:
    return PageView(
        attached=False,
        browser=kind,
        title=title,
        url="",
        text="",
        status=status,
        websocket_url="",
        relaunch_command=command,
    )


def _window_identity(window_id: int) -> tuple[str, str]:
    if sys.platform != "win32":
        return "", ""
    return _windows_title(window_id), _windows_exe(window_id)


def _windows_title(window_id: int) -> str:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int
    length = user32.GetWindowTextLengthW(int(window_id))
    if length <= 0:
        return ""
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(int(window_id), buffer, length + 1)
    return buffer.value.strip()


def _windows_exe(window_id: int) -> str:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.QueryFullProcessImageNameW.argtypes = [
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL

    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(int(window_id), ctypes.byref(pid))
    if not pid.value:
        return ""
    handle = kernel32.OpenProcess(0x1000, False, pid.value)
    if not handle:
        return ""
    try:
        size = wintypes.DWORD(1024)
        buffer = ctypes.create_unicode_buffer(size.value)
        if not kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            return ""
        return buffer.value
    finally:
        kernel32.CloseHandle(handle)


def _port_from_file(kind: str) -> int | None:
    for path in _port_file_candidates(kind):
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                port = int(handle.readline().strip())
        except (OSError, ValueError):
            continue
        if 1 <= port <= 65535:
            return port
    return None


def _port_file_candidates(kind: str) -> list[str]:
    local = os.environ.get("LOCALAPPDATA", "")
    home = os.path.expanduser("~")
    if kind == "edge":
        names = (
            os.path.join(local, "Microsoft", "Edge", "User Data", "DevToolsActivePort"),
            os.path.join(home, "Library", "Application Support", "Microsoft Edge", "DevToolsActivePort"),
            os.path.join(home, ".config", "microsoft-edge", "DevToolsActivePort"),
        )
    else:
        names = (
            os.path.join(local, "Google", "Chrome", "User Data", "DevToolsActivePort"),
            os.path.join(home, "Library", "Application Support", "Google", "Chrome", "DevToolsActivePort"),
            os.path.join(home, ".config", "google-chrome", "DevToolsActivePort"),
        )
    return [path for path in names if path]


def _port_is_browser(port: int, kind: str) -> bool:
    try:
        payload = _json_get(f"http://127.0.0.1:{port}/json/version", timeout=0.8)
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
        return False
    browser = str(payload.get("Browser") or "") if isinstance(payload, dict) else ""
    if kind == "edge":
        return "Edg" in browser
    return "Chrome" in browser and "Edg" not in browser


def _json_get(url: str, timeout: float):
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _trusted_click(session: "CdpSession", x: float, y: float) -> None:
    """A real mouse click. Synthetic el.click() is ignored by many frameworks."""
    session.call("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": x, "y": y})
    session.call(
        "Input.dispatchMouseEvent",
        {"type": "mousePressed", "x": x, "y": y, "button": "left", "clickCount": 1},
    )
    session.call(
        "Input.dispatchMouseEvent",
        {"type": "mouseReleased", "x": x, "y": y, "button": "left", "clickCount": 1},
    )
    time.sleep(0.15)


def _trusted_drag(session: "CdpSession", x: float, y: float, x2: float, y2: float) -> None:
    """A real mouse drag. HTML5 drag-and-drop needs moved steps between press and release."""
    session.call("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": x, "y": y})
    time.sleep(0.1)
    session.call(
        "Input.dispatchMouseEvent",
        {"type": "mousePressed", "x": x, "y": y, "button": "left", "buttons": 1, "clickCount": 1},
    )
    time.sleep(0.15)
    steps = 12
    for i in range(1, steps + 1):
        mx = x + (x2 - x) * i / steps
        my = y + (y2 - y) * i / steps
        session.call(
            "Input.dispatchMouseEvent",
            {"type": "mouseMoved", "x": mx, "y": my, "button": "left", "buttons": 1},
        )
        time.sleep(0.05)
    session.call(
        "Input.dispatchMouseEvent",
        {"type": "mouseMoved", "x": x2, "y": y2, "button": "left", "buttons": 1},
    )
    time.sleep(0.3)
    session.call(
        "Input.dispatchMouseEvent",
        {"type": "mouseReleased", "x": x2, "y": y2, "button": "left", "buttons": 1, "clickCount": 1},
    )
    time.sleep(0.2)


def _run_plan(session: "CdpSession", raw: str) -> tuple[str, bool] | None:
    """Execute a JSON click/type plan the page script returned. None if not a plan."""
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("plan"), list):
        return None
    for step in data["plan"]:
        if not isinstance(step, dict):
            continue
        x, y = step.get("x"), step.get("y")
        if x is None or y is None:
            continue
        act = step.get("act")
        if act == "click":
            _trusted_click(session, x, y)
        elif act == "drag":
            x2, y2 = step.get("x2"), step.get("y2")
            if x2 is not None and y2 is not None:
                _trusted_drag(session, x, y, x2, y2)
        elif act == "type":
            _trusted_click(session, x, y)
            session.call("Input.insertText", {"text": str(step.get("text") or "")})
            time.sleep(0.15)
    return str(data.get("result") or "done"), bool(data.get("repick"))


_KEY_EVENTS = {
    "enter": ("Enter", "Enter", 13),
    "return": ("Enter", "Enter", 13),
    "tab": ("Tab", "Tab", 9),
    "escape": ("Escape", "Escape", 27),
    "esc": ("Escape", "Escape", 27),
    "space": (" ", "Space", 32),
    "backspace": ("Backspace", "Backspace", 8),
    "delete": ("Delete", "Delete", 46),
    "arrowup": ("ArrowUp", "ArrowUp", 38),
    "up": ("ArrowUp", "ArrowUp", 38),
    "arrowdown": ("ArrowDown", "ArrowDown", 40),
    "down": ("ArrowDown", "ArrowDown", 40),
    "arrowleft": ("ArrowLeft", "ArrowLeft", 37),
    "left": ("ArrowLeft", "ArrowLeft", 37),
    "arrowright": ("ArrowRight", "ArrowRight", 39),
    "right": ("ArrowRight", "ArrowRight", 39),
    "home": ("Home", "Home", 36),
    "end": ("End", "End", 35),
    "pageup": ("PageUp", "PageUp", 33),
    "pagedown": ("PageDown", "PageDown", 34),
}


def _key_event_params(key: str) -> dict:
    k = (key or "enter").strip()
    entry = _KEY_EVENTS.get(k.casefold())
    if entry:
        name, code, vk = entry
        return {
            "key": name,
            "code": code,
            "windowsVirtualKeyCode": vk,
            "nativeVirtualKeyCode": vk,
        }
    if len(k) == 1:
        upper = k.upper()
        code = "Key" + upper if upper.isalpha() else ("Digit" + upper if upper.isdigit() else k)
        return {
            "key": k,
            "code": code,
            "windowsVirtualKeyCode": ord(upper),
            "nativeVirtualKeyCode": ord(upper),
            "text": k,
        }
    return {"key": k, "code": k, "windowsVirtualKeyCode": 0, "nativeVirtualKeyCode": 0}


def _choose_expression(anchor: str, answer: str, call_id: str = "") -> str:
    """Script that selects one answer on the page."""
    script = """
(() => {
  const anchor = __ANCHOR__;
  const answer = __ANSWER__;
  const norm = (value) => (value || '').replace(/\\s+/g, ' ').trim().toLowerCase();
  if (!answer) return 'refused';

  const docs = [];
  const visit = (win) => {
    try { docs.push(win.document); } catch (e) { return; }
    for (let i = 0; i < win.frames.length; i++) {
      try { visit(win.frames[i]); } catch (e) {}
    }
  };
  visit(window);

  const nodes = (root, selector) => {
    const found = [...root.querySelectorAll(selector)];
    root.querySelectorAll('*').forEach((el) => {
      if (el.shadowRoot) found.push(...nodes(el.shadowRoot, selector));
    });
    return found;
  };

  const textOf = (el) => norm(el.innerText || el.textContent || el.getAttribute('aria-label') || '');

  const scopeLength = (el) => {
    if (!anchor) return 0;
    let node = el;
    let best = 1000000;
    for (let i = 0; i < 8 && node; i++) {
      const text = norm(node.innerText || node.textContent || '');
      if (text.includes(anchor)) best = Math.min(best, text.length);
      node = node.parentElement || node.host || null;
    }
    return best;
  };

  const inScope = (el) => !anchor || scopeLength(el) < anchor.length + 500;

  const planClick = (el) => {
    el.scrollIntoView({block: 'center', inline: 'center'});
    const r = el.getBoundingClientRect();
    return {act: 'click', x: r.left + r.width / 2, y: r.top + r.height / 2};
  };
  const planType = (el, value) => {
    el.scrollIntoView({block: 'center', inline: 'center'});
    const r = el.getBoundingClientRect();
    el.value = '';
    return {act: 'type', x: r.left + r.width / 2, y: r.top + r.height / 2, text: value};
  };
  const pack = (plan, result, repick) => {
    const data = {plan, result};
    if (repick) data.repick = true;
    return JSON.stringify(data);
  };

  const pickSelect = (doc) => {
    const selects = nodes(doc, 'select').filter((select) =>
      [...select.options].some((option) => norm(option.text) === answer)
    );
    const pool = anchor ? selects.filter(inScope) : selects;
    if (!anchor && pool.length > 1) return 'need a sentence';
    if (!pool.length) return '';
    pool.sort((a, b) => scopeLength(a) - scopeLength(b));
    const option = [...pool[0].options].find((item) => norm(item.text) === answer);
    if (!option) return 'refused';
    option.selected = true;
    pool[0].value = option.value;
    pool[0].dispatchEvent(new Event('input', {bubbles: true}));
    pool[0].dispatchEvent(new Event('change', {bubbles: true}));
    return 'chose ' + answer;
  };

  const pickRow = (doc) => {
    const rows = nodes(doc, '.choice-row, .form-check, [role="option"], [role="radio"], [role="checkbox"]');
    let matches = rows.filter((el) => textOf(el) === answer && inScope(el));
    if (!matches.length) {
      matches = rows.filter((el) => {
        const t = textOf(el);
        return t.length > answer.length && t.includes(answer) && inScope(el);
      });
      matches.sort((a, b) => textOf(a).length - textOf(b).length);
    }
    if (!matches.length) return '';
    const row = matches[0];
    const input = row.querySelector('input[type="radio"], input[type="checkbox"]');
    if (input && input.checked) return 'chose ' + answer;
    const clickable = row.querySelector('._clickable, label') || input || row;
    return pack([planClick(clickable)], 'chose ' + answer);
  };

  const pickText = (doc) => {
    const inputs = nodes(doc, 'input.fitb-input, textarea.fitb-input');
    if (!inputs.length) return '';
    if (!anchor && inputs.length > 1) return 'need a sentence';
    const holders = [];
    inputs.forEach((el) => {
      const holder = el.closest('p, li, td') || el.parentElement;
      if (holder && !holders.includes(holder)) holders.push(holder);
    });
    const pool = holders.filter(inScope);
    if (!pool.length) return '';
    pool.sort((a, b) => scopeLength(a) - scopeLength(b));
    const fields = inputs.filter((el) => pool[0].contains(el));
    const words = answer.split(' ').filter(Boolean);
    if (fields.length === 1 || words.length === 1 && fields.length > 1 && fields.filter((el) => !norm(el.value)).length === 1) {
      const field = fields.length === 1 ? fields[0] : fields.filter((el) => !norm(el.value))[0];
      return pack([planType(field, answer)], 'chose ' + answer);
    }
    if (words.length === fields.length) {
      return pack(fields.map((field, index) => planType(field, words[index])), 'chose ' + answer);
    }
    return '';
  };

  const openMenu = (doc) => {
    const openers = nodes(doc, 'button, [role="button"], [role="combobox"]').filter((el) => {
      const text = textOf(el);
      return (text === 'click to select' || text === '(click to select)') && inScope(el);
    });
    if (!openers.length) return '';
    if (!anchor && openers.length !== 1) return '';
    openers.sort((a, b) => scopeLength(a) - scopeLength(b));
    const opener = openers[0];
    if (opener.getAttribute('aria-expanded') === 'true') return '';
    if (window.__agentHoyMenuOpened === __CALL__) return '';
    window.__agentHoyMenuOpened = __CALL__;
    return pack([planClick(opener)], 'menu open', true);
  };

  for (const doc of docs) {
    const selected = pickSelect(doc);
    if (selected) return selected;
    const chosen = pickRow(doc);
    if (chosen) return chosen;
    const typed = pickText(doc);
    if (typed) return typed;
  }
  for (const doc of docs) {
    const menuPlan = openMenu(doc);
    if (menuPlan) return menuPlan;
  }
  return 'not found';
})()
"""
    return (
        script.replace("__ANCHOR__", json.dumps(" ".join(anchor.casefold().split())))
        .replace("__ANSWER__", json.dumps(" ".join(answer.casefold().split())))
        .replace("__CALL__", json.dumps(call_id))
    )


def _match_expression(source: str, target: str) -> str:
    """Script that drags a tile onto the drop zone of a labeled match row."""
    script = """
(() => {
  const src = __SRC__;
  const dst = __DST__;
  const norm = (value) => (value || '').replace(/\\s+/g, ' ').trim().toLowerCase();
  if (!src || !dst) return 'refused';

  const docs = [];
  const visit = (win) => {
    try { docs.push(win.document); } catch (e) { return; }
    for (let i = 0; i < win.frames.length; i++) {
      try { visit(win.frames[i]); } catch (e) {}
    }
  };
  visit(window);

  const nodes = (root, selector) => {
    const found = [...root.querySelectorAll(selector)];
    root.querySelectorAll('*').forEach((el) => {
      if (el.shadowRoot) found.push(...nodes(el.shadowRoot, selector));
    });
    return found;
  };

  const textOf = (el) => norm(el.innerText || el.textContent || '');
  const center = (el) => {
    el.scrollIntoView({block: 'nearest', inline: 'nearest'});
    const r = el.getBoundingClientRect();
    return {x: r.left + r.width / 2, y: r.top + r.height / 2};
  };

  for (const doc of docs) {
    const allTiles = nodes(doc, '[draggable=true]');
    const allRows = nodes(doc, '.match-row');
    const pickTiles = (key) => allTiles.filter((el) => {
      const t = textOf(el);
      return t === key || t.includes(key) || (key.includes(t) && t.length > 2);
    });
    const pickRows = (key) => allRows.filter((el) => textOf(el).includes(key));
    let tiles = pickTiles(src);
    let rows = pickRows(dst);
    if (!tiles.length || !rows.length) {
      // The model may swap source and target; accept either direction.
      const swappedTiles = pickTiles(dst);
      const swappedRows = pickRows(src);
      if (swappedTiles.length && swappedRows.length) {
        tiles = swappedTiles;
        rows = swappedRows;
      }
    }
    if (!tiles.length || !rows.length) continue;
    tiles.sort((a, b) => textOf(a).length - textOf(b).length);
    rows.sort((a, b) => textOf(a).length - textOf(b).length);
    const zone = rows[0].querySelector('.droppableContainer') || rows[0];
    tiles[0].scrollIntoView({block: 'nearest', inline: 'nearest'});
    zone.scrollIntoView({block: 'nearest', inline: 'nearest'});
    const s = center(tiles[0]);
    const d = center(zone);
    return JSON.stringify({
      plan: [{act: 'drag', x: s.x, y: s.y, x2: d.x, y2: d.y}],
      result: 'matched ' + src + ' -> ' + dst,
    });
  }
  return 'not found';
})()
"""
    return (
        script.replace("__SRC__", json.dumps(" ".join(source.casefold().split())))
        .replace("__DST__", json.dumps(" ".join(target.casefold().split())))
    )


def _move_expression(source: str, target: str) -> str:
    """Script that drags an ordering-list item onto the item holding its slot."""
    script = """
(() => {
  const src = __SRC__;
  const dst = __DST__;
  const norm = (value) => (value || '').replace(/\\s+/g, ' ').trim().toLowerCase();
  if (!src || !dst) return 'refused';

  const docs = [];
  const visit = (win) => {
    try { docs.push(win.document); } catch (e) { return; }
    for (let i = 0; i < win.frames.length; i++) {
      try { visit(win.frames[i]); } catch (e) {}
    }
  };
  visit(window);

  const nodes = (root, selector) => {
    const found = [...root.querySelectorAll(selector)];
    root.querySelectorAll('*').forEach((el) => {
      if (el.shadowRoot) found.push(...nodes(el.shadowRoot, selector));
    });
    return found;
  };

  const textOf = (el) => norm(el.innerText || el.textContent || '');
  const pick = (items, key) => {
    let m = items.filter((el) => textOf(el) === key);
    if (!m.length) {
      m = items.filter((el) => {
        const t = textOf(el);
        return t.includes(key) || (key.includes(t) && t.length > 2);
      });
      m.sort((a, b) => textOf(a).length - textOf(b).length);
    }
    return m[0] || null;
  };

  for (const doc of docs) {
    const items = nodes(doc, '.choice-item[draggable=true], .choice-item').filter(
      (el) => el.offsetParent !== null
    );
    if (items.length < 2) continue;
    const tile = pick(items, src);
    const slot = pick(items, dst);
    if (!tile || !slot || tile === slot) continue;
    tile.scrollIntoView({block: 'nearest', inline: 'nearest'});
    slot.scrollIntoView({block: 'nearest', inline: 'nearest'});
    const a = tile.getBoundingClientRect();
    const b = slot.getBoundingClientRect();
    return JSON.stringify({
      plan: [{
        act: 'drag',
        x: a.left + a.width / 2, y: a.top + a.height / 2,
        x2: b.left + b.width / 2, y2: b.top + b.height / 2,
      }],
      result: 'moved ' + src + ' -> ' + dst,
    });
  }
  return 'not found';
})()
"""
    return (
        script.replace("__SRC__", json.dumps(" ".join(source.casefold().split())))
        .replace("__DST__", json.dumps(" ".join(target.casefold().split())))
    )


def _match_verify_expression(source: str, target: str) -> str:
    """True when the labeled match row's drop zone holds the dragged item."""
    script = """
(() => {
  const src = __SRC__;
  const dst = __DST__;
  const norm = (value) => (value || '').replace(/\\s+/g, ' ').trim().toLowerCase();
  const rows = [...document.querySelectorAll('.match-row')];
  const placed = (rowKey, tileKey) => rows.some((row) => {
    const text = norm(row.innerText || '');
    if (!text.includes(rowKey)) return false;
    const zone = row.querySelector('.droppableContainer');
    return zone && norm(zone.innerText || '').includes(tileKey);
  });
  return placed(dst, src) || placed(src, dst);
})()
"""
    return (
        script.replace("__SRC__", json.dumps(" ".join(source.casefold().split())))
        .replace("__DST__", json.dumps(" ".join(target.casefold().split())))
    )


def _run_one(session: CdpSession, verb: str, argument: str) -> str:
    verb = verb.strip().lower()
    argument = argument.strip()
    if verb == "navigate":
        if not argument.startswith(("http://", "https://")) or len(argument) > 2000:
            return "navigate skipped"
        try:
            session.call("Page.navigate", {"url": argument})
        except CdpError:
            expression = f"location.assign({json.dumps(argument)})"
            session.call("Runtime.evaluate", {"expression": expression, "returnByValue": True})
        return "navigated"
    if verb == "scroll":
        direction = "up" if argument.lower().startswith("up") else "down"
        delta = -700 if direction == "up" else 700
        session.call(
            "Runtime.evaluate",
            {"expression": f"window.scrollBy(0, {delta})", "returnByValue": True},
        )
        return f"scrolled {direction}"
    if verb == "press":
        key = (argument or "enter").strip()[:40] or "enter"
        session.call(
            "Runtime.evaluate",
            {
                "expression": "(() => { const el = window.__agentHoyField; if (el && el.focus) el.focus(); })()",
                "returnByValue": True,
            },
        )
        params = _key_event_params(key)
        session.call("Input.dispatchKeyEvent", {"type": "keyDown", **params})
        session.call("Input.dispatchKeyEvent", {"type": "keyUp", **params})
        return f"pressed {key}"
    if verb == "type":
        text = argument[:500]
        expression = (
            "(() => {"
            " const el = (document.activeElement && ('value' in document.activeElement) && document.activeElement)"
            "   || window.__agentHoyField;"
            f" const text = {json.dumps(text)};"
            " if (!el || !('value' in el)) return 'no field';"
            " el.value = '';"
            " el.scrollIntoView({block: 'center', inline: 'center'});"
            " const r = el.getBoundingClientRect();"
            " return JSON.stringify({plan: [{act: 'type', x: r.left + r.width / 2, y: r.top + r.height / 2, text: text}], result: 'typed'});"
            "})()"
        )
        result = session.call("Runtime.evaluate", {"expression": expression, "returnByValue": True})
        raw = str((result.get("result") or {}).get("value") or "typed")
        planned = _run_plan(session, raw)
        return planned[0] if planned else raw
    if verb == "move":
        parts = re.split(r"\s*(?:->|=>|\|)\s*", argument, maxsplit=1)
        if len(parts) != 2 or not all(part.strip() for part in parts):
            return "refused"
        result = session.call(
            "Runtime.evaluate",
            {
                "expression": _move_expression(parts[0].strip(), parts[1].strip()),
                "returnByValue": True,
            },
        )
        raw = str((result.get("result") or {}).get("value") or "not found")
        planned = _run_plan(session, raw)
        return planned[0] if planned else raw
    if verb == "match":
        parts = re.split(r"\s*(?:->|=>|\|)\s*", argument, maxsplit=1)
        if len(parts) != 2 or not all(part.strip() for part in parts):
            return "refused"
        source, target = parts[0].strip(), parts[1].strip()
        for _ in range(3):
            result = session.call(
                "Runtime.evaluate",
                {
                    "expression": _match_expression(source, target),
                    "returnByValue": True,
                },
            )
            raw = str((result.get("result") or {}).get("value") or "not found")
            planned = _run_plan(session, raw)
            note = planned[0] if planned else raw
            if not note.startswith("matched"):
                return note
            time.sleep(0.4)
            check = session.call(
                "Runtime.evaluate",
                {"expression": _match_verify_expression(source, target), "returnByValue": True},
            )
            if (check.get("result") or {}).get("value"):
                return note
            note = "not placed"
        return note
    if verb == "choose":
        if not CHOOSE_ENABLED:
            return "choose is off"
        parsed = parse_choose(argument)
        if parsed is None:
            return "refused"
        anchor, answer = parsed
        call_id = str(int(time.time() * 1000))
        text_result = "not found"
        for _ in range(2):
            result = session.call(
                "Runtime.evaluate",
                {
                    "expression": _choose_expression(anchor, answer, call_id),
                    "returnByValue": True,
                },
            )
            if result.get("exceptionDetails"):
                detail = result.get("exceptionDetails") or {}
                return str(detail.get("text") or "choose failed")[:160]
            raw = str((result.get("result") or {}).get("value") or "not found")
            planned = _run_plan(session, raw)
            if planned is None:
                return raw
            text_result, repick = planned
            if not repick:
                return text_result
            time.sleep(0.6)
        return text_result
    if verb == "advance":
        time.sleep(0.8)
        expression = (
            "(() => {"
            " const cands = [...document.querySelectorAll('button, [role=\"button\"], input[type=\"submit\"]')]"
            "   .filter(b => !b.disabled && b.offsetParent !== null);"
            " const name = (b) => (b.innerText || b.value || b.getAttribute('aria-label') || '').trim();"
            " const pats = [/submit/i, /check answer|verify/i, /^high$/i, /^medium$/i, /^low$/i, /confidence/i, /next/i, /continue|proceed|complete|finish/i];"
            " for (const p of pats) {"
            "   const b = cands.find(el => p.test(name(el)));"
            "   if (b) {"
            "     b.scrollIntoView({block: 'center', inline: 'center'});"
            "     const r = b.getBoundingClientRect();"
            "     return JSON.stringify({plan: [{act: 'click', x: r.left + r.width / 2, y: r.top + r.height / 2}], result: 'advanced: ' + name(b)});"
            "   }"
            " }"
            " return 'nothing to advance';"
            "})()"
        )
        result = session.call("Runtime.evaluate", {"expression": expression, "returnByValue": True})
        raw = str((result.get("result") or {}).get("value") or "nothing to advance")
        planned = _run_plan(session, raw)
        return planned[0] if planned else raw
    if verb == "click":
        label = argument[:200]
        expression = (
            "(() => {"
            f" const needle = {json.dumps(label.casefold())};"
            " if (!needle) return 'missing label';"
            " const nodes = document.querySelectorAll('a,button,input,textarea,select,label,[role=\"button\"],[role=\"link\"]');"
            " for (const node of nodes) {"
            "   const name = (node.innerText || node.value || node.getAttribute('aria-label') || node.getAttribute('placeholder') || '').trim().toLowerCase();"
            "   if (name.includes(needle)) {"
            "     window.__agentHoyField = node;"
            "     node.scrollIntoView({block: 'center', inline: 'center'});"
            "     const r = node.getBoundingClientRect();"
            "     return JSON.stringify({plan: [{act: 'click', x: r.left + r.width / 2, y: r.top + r.height / 2}], result: 'clicked'});"
            "   }"
            " }"
            " return 'not found';"
            "})()"
        )
        result = session.call("Runtime.evaluate", {"expression": expression, "returnByValue": True})
        raw = str((result.get("result") or {}).get("value") or "clicked")
        planned = _run_plan(session, raw)
        return planned[0] if planned else raw
    return ""
