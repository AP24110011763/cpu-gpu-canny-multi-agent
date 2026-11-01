"""Offline command parser: understands common commands without any AI."""
from __future__ import annotations

import difflib
import posixpath
import re
from collections import Counter
from typing import List, Optional, Tuple

from .models import CREATE, DELETE, UPDATE, Action, BotError, NeedsAIError, Plan

DELETE_WORDS = r"(?:delete|remove|erase|drop|trash)"
UPDATE_WORDS = r"(?:update|change|modify|edit|fix|replace|append|rewrite|refactor|improve)"
CREATE_WORDS = r"(?:create|add|make|generate|write|new|build|scaffold)"

KINDS = [  # (regex, extension) - order matters (javascript before java)
    (r"\bjavascript\b|\bjs\b", ".js"), (r"\btypescript\b|\bts\b", ".ts"), (r"\bjava\b", ".java"),
    (r"\bhtml\b", ".html"), (r"\bcss\b|\bstylesheet\b", ".css"), (r"\bpython\b|\bpy\b", ".py"),
    (r"\bmarkdown\b|\breadme\b", ".md"), (r"\bjson\b", ".json"), (r"\bc\+\+|\bcpp\b", ".cpp"),
    (r"\bsql\b", ".sql"), (r"\byaml\b|\byml\b", ".yml"), (r"\btext\b|\btxt\b", ".txt"),
]
STOP = set("""a an the new create add make generate write build scaffold file files program script page module
project to in inside into under within folder directory dir called named for of my that with simple basic and
code application please can you i want need existing some one it this on at from delete remove erase drop trash
update change modify edit fix replace rewrite refactor improve""".split())
KIND_WORDS = set("javascript js typescript ts java html css stylesheet python py markdown json cpp sql yaml yml text txt c".split())
ROOT_ONLY = ("readme", "license", "changelog", "contributing")


def detect_op(low: str) -> Optional[str]:
    for pat, op in ((DELETE_WORDS, DELETE), (UPDATE_WORDS, UPDATE), (CREATE_WORDS, CREATE)):
        if re.search(rf"\b{pat}\b", low):
            return op
    return None


def _folder(cmd: str) -> Optional[str]:
    m = (re.search(r"(?:inside|in|into|under|within|to)\s+(?:the\s+)?[\"']?([\w\-./]+?)[\"']?\s+(?:folder|directory|dir)\b", cmd, re.I)
         or re.search(r"\b(?:folder|directory)\s+[\"']?([\w\-./]+)", cmd, re.I))
    return m.group(1).strip("/") if m else None


def _filename(cmd: str) -> Optional[str]:
    m = re.search(r"(?:called|named)\s+[\"']?([\w\-./]+)", cmd, re.I)
    if m and "." in m.group(1).strip("."):
        return m.group(1)
    m = re.search(r"(?<![\w/.\-])((?:[\w\-]+/)*[\w\-]+\.[A-Za-z0-9]{1,8})(?![\w])", cmd)
    return m.group(1) if m else None


def _suggest_folder(path: str, files: List[str]) -> str:
    name = posixpath.basename(path)
    ext = posixpath.splitext(name)[1].lower()
    if name.startswith(".") or name.lower().startswith(ROOT_ONLY) or not ext:
        return ""
    dirs = Counter(posixpath.dirname(f) for f in files if posixpath.splitext(f)[1].lower() == ext)
    return dirs.most_common(1)[0][0] if dirs else ""


def _unique(path: str, files: List[str]) -> str:
    if path not in files:
        return path
    stem, ext = posixpath.splitext(path)
    i = 2
    while f"{stem}_{i}{ext}" in files:
        i += 1
    return f"{stem}_{i}{ext}"


def resolve_target(phrase: str, files: List[str], explicit: Optional[str]) -> Tuple[str, bool]:
    """Find an existing file. Returns (path, fuzzy). Raises BotError if unclear."""
    if explicit:
        if explicit in files:
            return explicit, False
        same = [f for f in files if posixpath.basename(f) == explicit]
        if not same:
            same = [f for f in files if posixpath.basename(f).lower() == explicit.lower()]
        if len(same) > 1:
            root = [f for f in same if "/" not in f]
            if len(root) == 1:
                return root[0], False
            raise BotError(f"'{explicit}' matches several files: {', '.join(same)}. Please give the full path.")
        if same:
            return same[0], False
        close = difflib.get_close_matches(explicit, files, n=3, cutoff=0.5)
        raise BotError(f"File '{explicit}' does not exist." + (f" Did you mean: {', '.join(close)}?" if close else ""))
    words = [w for w in re.findall(r"[a-z0-9_]+", phrase.lower()) if w not in STOP and w not in KIND_WORDS]
    if not words:
        raise BotError("Please tell me which file you mean.")
    hits = [f for f in files if any(w in posixpath.basename(f).lower() for w in words)]
    if len(hits) > 1:
        root = [f for f in hits if "/" not in f]
        if len(root) == 1:
            hits = root
    if len(hits) == 1:
        return hits[0], True
    if not hits:
        raise BotError(f"I couldn't find a file matching '{' '.join(words)}'.")
    raise BotError(f"'{' '.join(words)}' matches several files: {', '.join(hits[:8])}. Please be specific.")


def _infer_new_name(cmd: str, files: List[str], folder: Optional[str] = None) -> str:
    low = cmd.lower()
    ext = next((e for pat, e in KINDS if re.search(pat, low)), None)
    if not ext:
        raise BotError("I need a file name or type (for example 'create notes.txt' or 'add a CSS file').")
    skip = set(re.findall(r"[a-z0-9_]+", (folder or "").lower()))
    words = [w for w in re.findall(r"[a-z0-9_]+", low) if w not in STOP and w not in KIND_WORDS and w not in skip]
    if words:
        return "_".join(words[:2]) + ext
    if ext == ".html":
        return "index.html" if "index.html" not in files else "page.html"
    return {".md": "NOTES", ".css": "style", ".js": "script", ".java": "Main"}.get(ext, "main") + ext


def parse(command: str, files: List[str]) -> Plan:
    cmd = command.strip()
    low = cmd.lower()
    if not cmd:
        raise BotError("Empty command.")
    op = detect_op(low)
    if op is None:
        raise NeedsAIError("I can't tell what to do. Start with create / update / delete, "
                           "or set ANTHROPIC_API_KEY for full natural-language understanding.")
    fname, folder = _filename(cmd), _folder(cmd)

    if op == CREATE:
        inferred = fname is None
        path = fname or _infer_new_name(cmd, files, folder)
        if folder and "/" not in path:
            path = f"{folder}/{path}"
        elif "/" not in path:
            sug = _suggest_folder(path, files)
            path = f"{sug}/{path}" if sug else path
        if inferred:
            path = _unique(path, files)
        return Plan([Action(CREATE, path, instruction=cmd)])

    if op == DELETE:
        target, fuzzy = resolve_target(cmd, files, fname)
        return Plan([Action(DELETE, target, fuzzy=fuzzy)])

    # UPDATE: only simple, unambiguous edits can be done offline
    m = re.search(r"\b(?:append|add)\s+[\"“'](.+?)[\"”']\s+(?:to|at the end of)\s+(?:the\s+)?(?:file\s+)?([\w\-./]+\.\w+)", cmd, re.I)
    r = re.search(r"\breplace\s+[\"“'](.+?)[\"”']\s+with\s+[\"“'](.+?)[\"”']\s+in\s+(?:the\s+)?(?:file\s+)?([\w\-./]+\.\w+)", cmd, re.I)
    if m or r:
        target, fuzzy = resolve_target(cmd, files, (m or r).group(m and 2 or 3))
        return Plan([Action(UPDATE, target, instruction=cmd, fuzzy=fuzzy)])
    raise NeedsAIError("Free-form edits need the AI engine. Set ANTHROPIC_API_KEY, or use the offline forms:\n"
                       "  append \"text\" to FILE\n  replace \"old\" with \"new\" in FILE")


def offline_edits(instruction: str):
    """Edits for the offline update forms. Returns list of (old, new); old == '' means append."""
    m = re.search(r"\b(?:append|add)\s+[\"“'](.+?)[\"”']\s+(?:to|at the end of)", instruction, re.I)
    if m:
        return [("", m.group(1))]
    r = re.search(r"\breplace\s+[\"“'](.+?)[\"”']\s+with\s+[\"“'](.+?)[\"”']", instruction, re.I)
    if r:
        return [(r.group(1), r.group(2))]
    raise NeedsAIError("Free-form edits need the AI engine (set ANTHROPIC_API_KEY).")
