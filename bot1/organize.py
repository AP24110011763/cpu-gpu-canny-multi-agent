"""Inbox organizer: moves every file dropped in inbox/ to the right folder and commits it."""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .errors import BotError
from .safety import redact, validate_path

from .placement import (blob_sha, choose_folder, clean_name, is_sensitive_name, join, looks_secret,
                        unique_path, validate_folder)

SKIP_NAMES = {".gitkeep", "_READ_ME.md"}
IDENTITY = ["-c", "user.name=github-actions[bot]",
            "-c", "user.email=41898282+github-actions[bot]@users.noreply.github.com"]


@dataclass
class Move:
    src: str
    dest: str
    status: str      # moved | renamed | duplicate | secret | refused
    reason: str = ""


def _git(root: Path, *args: str, check: bool = True) -> str:
    r = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise BotError(redact(f"git {args[0]} failed: {(r.stderr or r.stdout).strip().splitlines()[:1]}"))
    return r.stdout


def tracked(root: Path) -> Dict[str, str]:
    """path -> blob sha for every tracked file."""
    out: Dict[str, str] = {}
    for entry in _git(root, "ls-files", "-s", "-z").split("\0"):
        if entry:
            meta, path = entry.split("\t", 1)
            out[path] = meta.split()[1]
    return out


def _preview(data: bytes) -> str:
    try:
        return data[:300].decode("utf-8")
    except UnicodeDecodeError:
        return ""


def organize(root: Path, placer=None, inbox: str = "inbox", ask: bool = False) -> Tuple[List[Move], bool]:
    """Returns (moves, committed)."""
    root = Path(root)
    inbox_dir = root / inbox
    if not inbox_dir.is_dir():
        return [], False
    files = sorted(p for p in inbox_dir.rglob("*") if p.is_file() and p.name not in SKIP_NAMES)
    if not files:
        return [], False

    shas = tracked(root)
    taken = {f for f in shas if not f.startswith(inbox + "/")}
    by_sha = {s: f for f, s in shas.items() if f in taken}
    pool = set(taken)          # what the repo looked like before this run (used to learn conventions)
    ai: Dict[str, str] = {}
    if placer:
        items = [{"name": clean_name(p.name), "size": p.stat().st_size, "preview": _preview(p.read_bytes()[:400])}
                 for p in files if posix_dir(p, inbox_dir) == ""]
        try:
            ai = placer.place(items, sorted({posix_dir_of(f) for f in taken if posix_dir_of(f)}))
        except BotError as e:
            print(f"AI placement skipped: {e}")

    moves: List[Move] = []
    for p in files:
        rel = p.relative_to(inbox_dir).as_posix()
        src, hint, name, data = f"{inbox}/{rel}", posix_dir(p, inbox_dir), clean_name(rel), p.read_bytes()

        if is_sensitive_name(name) or looks_secret(data):
            _git(root, "rm", "-f", "--ignore-unmatch", "--", src)
            p.unlink(missing_ok=True)
            moves.append(Move(src, "", "secret", "looks like a password/key, so it was deleted and NOT filed"))
            continue
        try:
            if hint:
                folder, reason = validate_folder(hint), f"you dropped it in inbox/{hint}/"
            else:
                folder, reason = choose_folder(name, pool)
                if name in ai:
                    try:
                        folder, reason = validate_folder(ai[name]), "chosen by AI from your repo layout"
                    except BotError:
                        pass
                folder = validate_folder(folder)
                if ask and name not in ai and reason.startswith(("unknown file type", "no folder for")):
                    moves.append(Move(src, "", "asked", f"my best guess was `{folder or 'the root'}/` but I'm not sure"))
                    continue
            dest = join(folder, name)
            validate_path(dest)
        except BotError as e:
            moves.append(Move(src, "", "refused", str(e)))
            continue

        sha = blob_sha(data)
        if sha in by_sha:                       # same content already in the repo
            _git(root, "rm", "-f", "--ignore-unmatch", "--", src)
            p.unlink(missing_ok=True)
            moves.append(Move(src, by_sha[sha], "duplicate", "an identical file is already in the repo"))
            continue
        status = "moved"
        if dest in taken:                       # same name, different content: keep both
            dest, status = unique_path(dest, taken), "renamed"
            reason = "a different file with this name exists, so both are kept"
        (root / dest).parent.mkdir(parents=True, exist_ok=True)
        try:
            _git(root, "mv", "--", src, dest)
        except BotError:
            shutil.move(str(p), str(root / dest))
            _git(root, "add", "-A", "--", src, dest)
        taken.add(dest)
        by_sha[sha] = dest
        moves.append(Move(src, dest, status, reason))

    committed = False
    if _git(root, "status", "--porcelain").strip():
        n = sum(1 for m in moves if m.status in ("moved", "renamed"))
        body = "\n".join(f"- {m.src} -> {m.dest or '(removed)'} [{m.status}]" for m in moves)
        _git(root, "add", "-A")
        _git(root, *IDENTITY, "commit", "-m", f"Bot 1: organize {n} file(s) from inbox\n\n{body}")
        committed = True
    return moves, committed


def posix_dir(p: Path, base: Path) -> str:
    d = p.relative_to(base).parent.as_posix()
    return "" if d == "." else d


def posix_dir_of(path: str) -> str:
    d = os.path.dirname(path)
    return d


def push_branch(root: Path, branch: str, retries: int = 3) -> None:
    for _ in range(retries):
        r = subprocess.run(["git", "push", "origin", f"HEAD:{branch}"], cwd=root, capture_output=True, text=True)
        if r.returncode == 0:
            return
        subprocess.run(["git", *IDENTITY, "pull", "--rebase", "origin", branch], cwd=root, capture_output=True, text=True)
    raise BotError("Could not push to the branch (it may be protected, or it kept changing).")


ICON = {"asked": "❓", "moved": "✅", "renamed": "🔁", "duplicate": "♻️", "secret": "🚨", "refused": "⛔"}


def format_report(moves: List[Move]) -> str:
    done = sum(1 for m in moves if m.status in ("moved", "renamed"))
    lines = [f"### 📥 Bot 1 organized {done} file(s) from `inbox/`", ""]
    for m in moves:
        name = m.src.split("/", 1)[-1]
        if m.status in ("moved", "renamed"):
            lines.append(f"- {ICON[m.status]} `{name}` → `{m.dest}` — {m.reason}")
        elif m.status == "duplicate":
            lines.append(f"- {ICON[m.status]} `{name}` was not added: {m.reason} (`{m.dest}`)")
        elif m.status == "secret":
            lines.append(f"- {ICON[m.status]} `{name}` {m.reason}. **Revoke that key/password now**: it is still in the git history.")
        else:
            lines.append(f"- {ICON[m.status]} `{name}` was left in `inbox/`: {m.reason}")
    return "\n".join(lines)
