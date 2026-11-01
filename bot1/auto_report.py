"""Automatic mode: write PROJECT_REPORT.md from the repository contents and commit it.

Runs from GitHub Actions on a schedule (and on pushes to main). It only commits when the
report content actually changed, so there are never empty or duplicate commits.
"""
from __future__ import annotations

import datetime as dt
import os
import sys
from typing import List

from .config import load_config
from .github_client import GitHubRepo
from .inspector import inspect_repo
from .models import BotError
from .security import redact, register_secrets

REPORT = "PROJECT_REPORT.md"
STAMP = "_Last updated:"


def build_report(repo, files: List[str]) -> str:
    own = [f for f in files if f != REPORT and not f.startswith((".github/", "bot1/"))]
    info = inspect_repo(repo, own)
    out = ["# Project Report", "",
           "Generated automatically by Bot 1 from the files in this repository.", "",
           "## Languages", ""]
    out += [f"- {k}: {v} file(s)" for k, v in info.languages.items()] or ["- none yet"]
    out += ["", "## Frameworks and libraries", "",
            ", ".join(info.frameworks + [d for d in info.dependencies if d not in info.frameworks][:30]) or "none detected",
            "", f"## Files ({len(own)})", ""]
    out += [f"- `{f}`" for f in own[:200]]
    if len(own) > 200:
        out.append(f"- ... and {len(own) - 200} more")
    return "\n".join(out) + "\n"


def stamp() -> str:
    return f"\n{STAMP} {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC_\n"


def _same(a: str, b: str) -> bool:
    strip = lambda t: "\n".join(l for l in t.splitlines() if not l.startswith(STAMP)).strip()
    return strip(a) == strip(b)


def run(repo) -> str:
    body = build_report(repo, repo.list_files())
    got = repo.read(REPORT)
    if got is None:
        repo.create(REPORT, body + stamp(), f"Bot 1: add {REPORT}")
        return "created"
    if _same(got[0], body):
        return "unchanged (no commit)"
    repo.update(REPORT, body + stamp(), got[1], f"Bot 1: update {REPORT}")
    return "updated"


def main() -> int:
    cfg = load_config(repo_override=os.environ.get("GITHUB_REPOSITORY"))
    register_secrets(cfg.secrets())
    try:
        print(f"{REPORT}: {run(GitHubRepo(cfg.github_token, cfg.repo, cfg.branch))}")
        return 0
    except BotError as e:
        print(redact(f"Bot 1 error: {e}"), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
