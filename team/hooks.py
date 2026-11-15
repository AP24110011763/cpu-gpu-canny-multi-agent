"""Called by Bot 1's organizer after it files uploads: asks you questions and hands code to Bot 2."""
from __future__ import annotations

import os
from typing import List, Optional
from urllib.parse import quote

from . import messages
from .bot1_agent import is_code
from .events import say
from .gh import GH


def gh_from_env() -> Optional[GH]:
    token, repo = os.environ.get("GITHUB_TOKEN", ""), os.environ.get("GITHUB_REPOSITORY", "")
    return GH(token, repo) if token and repo else None


def after_organize(moves: List, gh: Optional[GH] = None) -> None:
    gh = gh or gh_from_env()
    if gh is None:
        return
    log = None
    for m in moves:
        if m.status == "asked":
            log = log or gh.log_issue()
            name = m.src.split("/", 1)[-1]
            if gh.find_comment(log, f"file={quote(m.src, safe='/._-')}"):
                continue                                  # already asked: don't nag
            say(gh, log, "bot1", "user", "question",
                f"@{gh.owner} I couldn't tell where `{name}` belongs ({m.reason}). I left it in `inbox/`.\n\n"
                f"Reply `/bot1 put {name} in FOLDER`, for example `/bot1 put {name} in docs`.", file=m.src)
        elif m.status in ("moved", "renamed") and is_code(m.dest):
            log = log or gh.log_issue()
            say(gh, log, "bot1", "bot2", "review_request", f"I filed `{m.dest}`. Could you review it?", path=m.dest)
            gh.dispatch("bot2-review-request", {"path": m.dest, "thread": log})
