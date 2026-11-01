"""Local mode: give Bot 1 a file (or watch a folder) and it files + pushes it for you."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .errors import BotError
from .safety import validate_path

from .placement import (blob_sha, choose_folder, clean_name, is_sensitive_name, join, looks_secret,
                        unique_path, validate_folder)

MAX_BYTES = 50_000_000
IGNORED_SUFFIXES = (".crdownload", ".part", ".tmp", ".download")


@dataclass
class PushResult:
    local: str
    dest: str
    status: str      # pushed | duplicate | refused | failed | dry-run
    reason: str = ""
    url: str = ""


def push_files(repo, paths: List[Path], folder: Optional[str] = None, placer=None, dry_run: bool = False) -> List[PushResult]:
    shas: Dict[str, str] = repo.tree_shas()
    taken = set(shas)
    by_sha = {s: f for f, s in shas.items()}
    pool = set(taken)
    ai: Dict[str, str] = {}
    if placer and not folder:
        try:
            ai = placer.place([{"name": clean_name(p.name), "size": p.stat().st_size} for p in paths],
                              sorted({f.rsplit("/", 1)[0] for f in taken if "/" in f}))
        except BotError:
            pass
    out: List[PushResult] = []
    for p in paths:
        local = str(p)
        try:
            data = p.read_bytes()
        except OSError:
            out.append(PushResult(local, "", "failed", "cannot read the file"))
            continue
        name = clean_name(p.name)
        if is_sensitive_name(name) or looks_secret(data):
            out.append(PushResult(local, "", "refused", "looks like a password/key, so it was NOT pushed"))
            continue
        if len(data) > MAX_BYTES or not data:
            out.append(PushResult(local, "", "refused", "file is empty or larger than 50 MB"))
            continue
        try:
            if folder is not None:
                fld, reason = validate_folder(folder), "you chose this folder"
            else:
                fld, reason = choose_folder(name, pool)
                if name in ai:
                    try:
                        fld, reason = validate_folder(ai[name]), "chosen by AI from your repo layout"
                    except BotError:
                        pass
                fld = validate_folder(fld)
            dest = join(fld, name)
            validate_path(dest)
        except BotError as e:
            out.append(PushResult(local, "", "refused", str(e)))
            continue
        sha = blob_sha(data)
        if sha in by_sha:
            out.append(PushResult(local, by_sha[sha], "duplicate", "an identical file is already in the repo"))
            continue
        if dest in taken:
            dest, reason = unique_path(dest, taken), "a different file with this name exists, so both are kept"
        if dry_run:
            out.append(PushResult(local, dest, "dry-run", reason))
            continue
        try:
            c = repo.create_bytes(dest, data, f"Add {dest}")
        except BotError as e:
            out.append(PushResult(local, dest, "failed", str(e)))
            continue
        taken.add(dest)
        by_sha[sha] = dest
        out.append(PushResult(local, dest, "pushed", reason, c.url))
    return out


def ready_files(folder: Path, seen: Dict[Path, tuple]) -> List[Path]:
    """Files whose size/mtime did not change since the last poll (so they finished copying)."""
    ready, now = [], {}
    for p in sorted(folder.iterdir()):
        if not p.is_file() or p.name.startswith((".", "~$", "_")) or p.name.endswith(IGNORED_SUFFIXES):
            continue
        st = p.stat()
        now[p] = (st.st_size, st.st_mtime)
        if seen.get(p) == now[p]:
            ready.append(p)
    seen.clear()
    seen.update(now)
    return ready


def process_folder(repo, folder: Path, seen: Dict[Path, tuple], placer=None, dry_run=False) -> List[PushResult]:
    files = ready_files(folder, seen)
    if not files:
        return []
    results = push_files(repo, files, None, placer, dry_run)
    done = folder / "_done"
    for p, r in zip(files, results):
        if r.status in ("pushed", "duplicate") and not dry_run:
            done.mkdir(exist_ok=True)
            target = done / p.name
            n = 2
            while target.exists():
                target = done / f"{p.stem}_{n}{p.suffix}"
                n += 1
            p.rename(target)
            seen.pop(p, None)
    return results


def watch(repo, folder: Path, interval: float, log: Callable[[str], None], placer=None) -> None:
    seen: Dict[Path, tuple] = {}
    log(f"Watching {folder} - drop a file in and it is pushed automatically. Press Ctrl+C to stop.")
    while True:
        for r in process_folder(repo, folder, seen, placer):
            log(format_result(r))
        time.sleep(interval)


def format_result(r: PushResult) -> str:
    name = Path(r.local).name
    if r.status == "pushed":
        return f"[+] {name} -> {r.dest}  ({r.reason})" + (f"\n    {r.url}" if r.url else "")
    if r.status == "duplicate":
        return f"[=] {name}: {r.reason} ({r.dest})"
    if r.status == "dry-run":
        return f"[?] {name} would go to {r.dest}  ({r.reason})"
    return f"[!] {name}: {r.status} - {r.reason}"
