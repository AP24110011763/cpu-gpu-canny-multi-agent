"""Daily updates + file index, written from git history. Run inside a checked-out repo.

Writes two files:
  DAILY_UPDATES.md   - for every day: which files were added / changed / deleted, and the commit messages
  PROJECT_REPORT.md  - languages, libraries and the list of files that exist right now
The GitHub Actions workflow then commits them (only when something changed).
"""
from __future__ import annotations

import os
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .auto_report import REPORT, build_report

UPDATES = "DAILY_UPDATES.md"
OWN = {UPDATES, REPORT}
TZ = ZoneInfo(os.environ.get("BOT_TZ", "Asia/Kolkata"))
SEP = "\x1e"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


class LocalRepo:
    """Just enough of the repo interface for build_report(), reading the working folder."""
    def list_files(self):
        return [f for f in git("ls-files").splitlines() if f]

    def read(self, path):
        p = Path(path)
        return (p.read_text(encoding="utf-8", errors="replace"), "") if p.is_file() else None


def history():
    """-> OrderedDict day -> list of (time, message, {path: status}) newest first."""
    out = git("log", "--no-merges", "--name-status", "--date=iso-strict",
              f"--pretty=format:{SEP}%ad|%s")
    days: "OrderedDict[str, list]" = OrderedDict()
    for block in out.split(SEP)[1:]:
        lines = block.strip("\n").splitlines()
        when, _, msg = lines[0].partition("|")
        if msg.startswith("Bot 1:") or msg.startswith("Auto-update"):
            continue
        t = datetime.fromisoformat(when).astimezone(TZ)
        changes = {}
        for ln in lines[1:]:
            parts = ln.split("\t")
            if len(parts) >= 2 and parts[-1] not in OWN:
                changes[parts[-1]] = {"A": "added", "M": "changed", "D": "deleted"}.get(parts[0][0], "changed")
        days.setdefault(t.strftime("%Y-%m-%d"), []).append((t.strftime("%H:%M"), msg, changes))
    return days


def render(days) -> str:
    out = ["# Daily Updates", "", f"Written automatically by Bot 1. Times are in {TZ.key}.", ""]
    if not days:
        out.append("No updates yet.")
    for day, items in days.items():
        out += [f"## {day}", ""]
        merged = {}
        for _, _, ch in reversed(items):          # oldest first, later state wins
            merged.update(ch)
        for label in ("added", "changed", "deleted"):
            names = sorted(f for f, s in merged.items() if s == label)
            if names:
                out.append(f"- **{label.capitalize()}:** " + ", ".join(f"`{n}`" for n in names))
        out += ["", "Commits:"] + [f"- {t} - {m}" for t, m, _ in items] + [""]
    return "\n".join(out)


def main() -> int:
    Path(UPDATES).write_text(render(history()), encoding="utf-8")
    repo = LocalRepo()
    Path(REPORT).write_text(build_report(repo, repo.list_files()), encoding="utf-8")
    print(f"wrote {UPDATES} and {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
