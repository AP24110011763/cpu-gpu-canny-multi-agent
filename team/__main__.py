"""python -m team chat | comment | handoff | result | digest | intro"""
from __future__ import annotations

import json
import os
import sys

from .bot1_agent import Bot1Agent, Ctx
from .bot2_agent import Bot2Agent
from .events import INTRO
from .gh import GH


def _event() -> dict:
    with open(os.environ["GITHUB_EVENT_PATH"], encoding="utf-8") as f:
        return json.load(f)


def _gh() -> GH:
    return GH(os.environ["GITHUB_TOKEN"], os.environ["GITHUB_REPOSITORY"])


def _repo_factory():
    from bot1.github_client import GitHubRepo
    return GitHubRepo(os.environ["GITHUB_TOKEN"], os.environ["GITHUB_REPOSITORY"])


def chat(inp=input, out=print, bots=None) -> int:
    """Terminal chat with both bots. '/bot2 review x.py' talks to one bot for one line; 'bot2' switches."""
    from .commands import find_mentions
    if bots is None:
        bots = build_terminal_bots(out)
    current = "bot1"
    out("Team chat. Type /bot1 or /bot2 before a message to pick who you talk to, or just 'bot1' / 'bot2' to switch. /quit to leave.\n")
    for b in ("bot1", "bot2"):
        out(f"{b}: {INTRO[b]}\n")
    ctx = Ctx(source="terminal")
    while True:
        try:
            line = inp(f"you → {current}> ").strip()
        except (EOFError, KeyboardInterrupt):
            out("")
            return 0
        if not line:
            continue
        if line.lower() in ("/quit", "/exit", "quit", "exit"):
            return 0
        if line.lower() in ("bot1", "bot2"):
            current = line.lower()
            continue
        target, text = current, line
        mentions = find_mentions(line)
        if mentions:
            target, text = mentions[0]
        out(f"{target}: {bots[target].handle(text, ctx)}\n")


def build_terminal_bots(out=print) -> dict:
    gh = None
    try:
        from bot1.config import load_config
        cfg = load_config()
        gh = GH(cfg.github_token, cfg.repo)
        os.environ.setdefault("GITHUB_TOKEN", cfg.github_token)
        os.environ.setdefault("GITHUB_REPOSITORY", cfg.repo)
        out(f"(connected to {cfg.repo})")
    except Exception as e:
        out(f"(not connected to GitHub: {e} — Bot 2 can still review local files.)")

    def repo_factory():
        from bot1.github_client import GitHubRepo
        from bot1.config import load_config
        c = load_config()
        return GitHubRepo(c.github_token, c.repo, c.branch)
    return {"bot1": Bot1Agent(repo_factory, gh), "bot2": Bot2Agent(gh)}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else "chat"
    if cmd == "chat":
        return chat()
    from . import events
    if cmd == "comment":
        events.handle_comment(_event(), _gh(), _repo_factory)
    elif cmd == "handoff":
        print("Bot 2 verdict:", events.handle_review_request(_event()["client_payload"], _gh()))
    elif cmd == "result":
        print("Bot 1 outcome:", events.handle_review_result(_event()["client_payload"], _gh()))
    elif cmd == "digest":
        events.weekly_digest(_gh())
    elif cmd == "intro":
        events.post_intros(_gh())
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
