"""Path validation and secret redaction."""
from __future__ import annotations

import fnmatch
import posixpath
import re
from typing import Iterable, List

from .errors import BotError

MAX_PATH_LEN = 200
_ALLOWED = re.compile(r"^[\w.\-/ @+]+$", re.UNICODE)
BLOCKED = [".git", ".git/*", ".env", ".env.*", "*/.env", "*/.env.*"]
SAFE_ENV_TEMPLATES = {".env.example", ".env.sample", ".env.template"}


def validate_path(path: str) -> str:
    """Return a clean repo-relative POSIX path or raise BotError."""
    if not isinstance(path, str) or not path.strip():
        raise BotError("Empty file path.")
    p = path.strip()
    if "\x00" in p or "\\" in p:
        raise BotError(f"Invalid characters in path: {p!r}")
    if p.startswith("/") or re.match(r"^[A-Za-z]:", p) or p.startswith("~"):
        raise BotError("Absolute paths are not allowed; use a path inside the repository.")
    if any(seg == ".." for seg in p.split("/")):
        raise BotError("Path traversal ('..') is not allowed.")
    if len(p) > MAX_PATH_LEN or not _ALLOWED.match(p):
        raise BotError(f"Path contains unsupported characters or is too long: {p!r}")
    norm = posixpath.normpath(p)
    if norm in (".", "") or p.endswith("/"):
        raise BotError("Path must point to a file, not a folder.")
    if posixpath.basename(norm).lower() in SAFE_ENV_TEMPLATES:
        return norm
    if any(fnmatch.fnmatch(norm.lower(), pat) for pat in BLOCKED):
        raise BotError(f"'{norm}' is protected and can never be modified by the bot.")
    return norm


_PATTERNS = [
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}"),
    re.compile(r"(?i)(authorization|token|password)\s*[:=]\s*\S+"),
]
_known: List[str] = []


def register_secrets(values: Iterable[str]) -> None:
    for v in values:
        if v and len(v) >= 6 and v not in _known:
            _known.append(v)


def redact(text) -> str:
    text = str(text)
    for s in _known:
        text = text.replace(s, "***")
    for pat in _PATTERNS:
        text = pat.sub("***", text)
    return text
