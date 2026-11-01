"""Decide which folder a file belongs in (rules + what your repo already does)."""
from __future__ import annotations

import fnmatch
import posixpath
import re
from collections import Counter
from typing import Iterable, List, Tuple

from .errors import BotError
from .safety import validate_path

PROTECTED_TOP = {".git", ".github", "bot2", "inbox"}

ROOT_PATTERNS = ["readme*", "license*", "licence*", "contributing*", "changelog*", "code_of_conduct*",
                 ".gitignore", ".gitattributes", ".editorconfig", "requirements*.txt", "package.json",
                 "package-lock.json", "yarn.lock", "pom.xml", "build.gradle", "settings.gradle", "dockerfile",
                 "docker-compose.y*ml", "makefile", "pyproject.toml", "setup.py", "setup.cfg", "pipfile*",
                 "environment.yml", "tsconfig.json", "go.mod", "go.sum", "cargo.toml", "gemfile*", ".env.example"]

# category -> (extensions, existing-folder names to reuse, default new folder)
CATEGORIES = {
    "code": ({".py", ".java", ".js", ".jsx", ".ts", ".tsx", ".c", ".cpp", ".h", ".hpp", ".cs", ".go", ".rs",
              ".rb", ".php", ".kt", ".swift", ".sql", ".r", ".m", ".cu", ".ino", ".v", ".sv"},
             ["src", "source", "code", "app", "lib"], "src"),
    "web": ({".html", ".htm"}, ["web", "public", "pages", "templates", "static"], "web"),
    "style": ({".css", ".scss", ".sass", ".less"}, ["styles", "css", "static/css", "assets/css", "web/css"], "styles"),
    "notebook": ({".ipynb"}, ["notebooks", "notebook"], "notebooks"),
    "scripts": ({".sh", ".bat", ".ps1", ".cmd"}, ["scripts", "bin", "tools"], "scripts"),
    "docs": ({".pdf", ".doc", ".docx", ".ppt", ".pptx", ".odt", ".txt", ".md", ".rst", ".tex", ".rtf"},
             ["docs", "doc", "documentation", "documents", "reports"], "docs"),
    "data": ({".csv", ".tsv", ".json", ".xlsx", ".xls", ".xml", ".parquet", ".npy", ".npz", ".pkl", ".h5",
              ".db", ".sqlite", ".mat"}, ["data", "datasets", "dataset"], "data"),
    "images": ({".png", ".jpg", ".jpeg", ".gif", ".svg", ".bmp", ".webp", ".ico", ".tif", ".tiff"},
               ["assets/images", "images", "img", "static/images", "media/images", "assets"], "assets/images"),
    "media": ({".mp3", ".wav", ".mp4", ".mov", ".avi", ".mkv", ".webm", ".ogg", ".flac"},
              ["assets/media", "media", "assets/audio", "assets/video", "audio", "video"], "assets/media"),
    "fonts": ({".ttf", ".otf", ".woff", ".woff2"}, ["assets/fonts", "fonts"], "assets/fonts"),
    "config": ({".yml", ".yaml", ".toml", ".ini", ".cfg", ".conf", ".properties"},
               ["config", "configs", "conf", "settings"], "config"),
    "archives": ({".zip", ".tar", ".gz", ".tgz", ".7z", ".rar"}, ["archives", "downloads"], "archives"),
}
LEARN = {"code", "web", "style", "notebook", "scripts"}   # copy where same-type files already live

_TEST_RE = re.compile(r"^test_.*\.py$|.*_test\.(py|go)$|.*\.(test|spec)\.(js|jsx|ts|tsx)$|.*Test\.java$|^tests?\.\w+$")
SENSITIVE_NAMES = [".env", ".env.*", "id_rsa*", "id_ed25519*", "*.pem", "*.p12", "*.pfx", "*.key", "*.keystore"]
SECRET_PATTERNS = [
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{30,}"), re.compile(rb"github_pat_[A-Za-z0-9_]{30,}"),
    re.compile(rb"sk-ant-[A-Za-z0-9_\-]{20,}"), re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"xox[baprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY"),
]


def clean_name(name: str) -> str:
    """Make a browser-style file name safe: 'report (1).pdf' -> 'report_1.pdf'."""
    name = posixpath.basename(name.replace("\\", "/"))
    name = re.sub(r"\s*\((\d+)\)", r"_\1", name)
    name = re.sub(r"[^\w.\- @+]", "_", name, flags=re.UNICODE).rstrip(" .") or "file"
    stem, ext = posixpath.splitext(name)
    return stem[: 120 - len(ext)] + ext if len(name) > 120 else name


def is_sensitive_name(name: str) -> bool:
    low = name.lower()
    return low not in (".env.example", ".env.sample", ".env.template") and \
        any(fnmatch.fnmatch(low, p) for p in SENSITIVE_NAMES)


def looks_secret(data: bytes) -> bool:
    head = data[:1_000_000]
    return any(p.search(head) for p in SECRET_PATTERNS)


def blob_sha(data: bytes) -> str:
    import hashlib
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def is_test_file(name: str) -> bool:
    return bool(_TEST_RE.match(posixpath.basename(name)))


def validate_folder(folder: str) -> str:
    """Return a clean folder ('' = repo root) or raise BotError."""
    folder = folder.strip()
    if folder.startswith(("/", "~", "\\")):
        raise BotError("Use a folder inside the repository, not an absolute path.")
    folder = folder.rstrip("/")
    if not folder:
        return ""
    if len(folder.split("/")) > 6:
        raise BotError("Folder is nested too deeply.")
    clean = validate_path(folder + "/x")[:-2]
    if clean.split("/")[0].lower() in PROTECTED_TOP:
        raise BotError(f"'{clean}' is reserved for the bots and cannot be used.")
    return clean


def _dir(path: str) -> str:
    return posixpath.dirname(path)


def _top(counter: Counter) -> str:
    """Most common folder; ties go to the repo root, then alphabetical (deterministic)."""
    return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def choose_folder(name: str, existing: Iterable[str]) -> Tuple[str, str]:
    """Return (folder, reason). folder '' means the repository root."""
    low = name.lower()
    files = [f for f in existing if f.split("/")[0] not in PROTECTED_TOP]
    if any(fnmatch.fnmatch(low, p) for p in ROOT_PATTERNS):
        return "", "project files like this belong in the repository root"
    ext = posixpath.splitext(low)[1]

    if is_test_file(name):
        dirs = Counter(_dir(f) for f in files if is_test_file(f) and _dir(f))
        if dirs:
            d = _top(dirs)
            return d, f"your other test files live in {d}/"
        return "tests", "test files go in tests/"

    cat = next((c for c, (exts, _a, _d) in CATEGORIES.items() if ext in exts), None)
    if cat is None:
        return "misc", "unknown file type, so it goes in misc/"
    _exts, aliases, default = CATEGORIES[cat]

    if cat in LEARN:
        dirs = Counter(_dir(f) for f in files if posixpath.splitext(f.lower())[1] == ext and not is_test_file(f))
        if dirs:
            d = _top(dirs)
            return d, f"your other {ext} files live in {d + '/' if d else 'the repository root'}"
    for alias in aliases:
        if any(f.startswith(alias + "/") for f in files):
            return alias, f"your repo already has a {alias}/ folder for this kind of file"
    return default, f"no folder for {cat} files yet, so {default}/ is created"


def unique_path(path: str, taken: set) -> str:
    if path not in taken:
        return path
    stem, ext = posixpath.splitext(path)
    i = 2
    while f"{stem}_{i}{ext}" in taken:
        i += 1
    return f"{stem}_{i}{ext}"


def join(folder: str, name: str) -> str:
    return f"{folder}/{name}" if folder else name
