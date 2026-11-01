"""Path validation, protected-file rules and secret redaction."""
from __future__ import annotations

import fnmatch
import logging
import posixpath
import re
from typing import Iterable, List

from .models import BotError

MAX_PATH_LEN = 200
MAX_FILE_BYTES = 1_000_000
_ALLOWED = re.compile(r"^[\w.\-/ @+]+$", re.UNICODE)

# Never touched by the bot, no matter what (even with --yes).
BLOCKED = [".git", ".git/*", ".env", ".env.*", "*/.env", "*/.env.*"]
SAFE_ENV_TEMPLATES = {".env.example", ".env.sample", ".env.template"}
# Allowed, but the user must confirm first.
IMPORTANT = [
    "README*", "LICENSE*", ".gitignore", ".github/*", "bot1/*",
    "requirements.txt", "package.json", "package-lock.json", "pom.xml",
    "build.gradle", "Dockerfile", "docker-compose.yml", "Makefile",
    "pyproject.toml", "setup.py",
    "main.py", "main.java", "main.js", "main.ts", "main.go", "main.c", "main.cpp",
    "index.html", "index.js", "index.ts", "app.py", "app.js", "app.ts", "app.java",
]


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
    if norm in (".", "") or norm.endswith("/") or p.endswith("/"):
        raise BotError("Path must point to a file, not a folder.")
    if posixpath.basename(norm).lower() in SAFE_ENV_TEMPLATES:
        return norm
    if any(fnmatch.fnmatch(norm, pat) or fnmatch.fnmatch(norm.lower(), pat.lower()) for pat in BLOCKED):
        raise BotError(f"'{norm}' is protected and can never be modified by the bot.")
    return norm


def is_important(path: str) -> bool:
    name = posixpath.basename(path)
    return any(
        fnmatch.fnmatch(path.lower(), p.lower()) or fnmatch.fnmatch(name.lower(), p.lower())
        for p in IMPORTANT
    )


def sanitize_commit_message(msg: str, fallback: str) -> str:
    msg = re.sub(r"[\x00-\x1f\x7f]", " ", (msg or "").splitlines()[0] if msg else "").strip()
    return (msg or fallback)[:72]


# ---------- secret redaction ----------
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


def redact(text: str) -> str:
    text = str(text)
    for s in _known:
        text = text.replace(s, "***")
    for pat in _PATTERNS:
        text = pat.sub("***", text)
    return text


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = ()
        return True
