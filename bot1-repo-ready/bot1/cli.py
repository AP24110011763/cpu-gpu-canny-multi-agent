"""Command-line interface:  python -m bot1 "Create a login.java file" """
from __future__ import annotations

import argparse
import logging
import sys
from typing import List

from .ai_engine import make_engine
from .bot import Bot
from .config import load_config
from .github_client import GitHubRepo
from .memory_repo import MemoryRepo
from .models import BotError, Result
from .security import RedactingFilter, redact, register_secrets

ICON = {"created": "[+]", "updated": "[~]", "deleted": "[-]", "skipped": "[=]",
        "cancelled": "[x]", "failed": "[!]", "dry-run": "[?]"}


def _confirm_factory(auto_yes: bool):
    def confirm(prompt: str) -> bool:
        print(prompt)
        if auto_yes:
            print("  --yes given: confirmed automatically.")
            return True
        if not sys.stdin.isatty():
            print("  No interactive terminal and no --yes flag: declined.")
            return False
        return input("Type 'yes' to confirm: ").strip().lower() in ("yes", "y")
    return confirm


def print_results(results: List[Result]) -> None:
    print("\nResult:")
    for r in results:
        line = f"{ICON.get(r.status, '[?]')} {r.status.upper():9} {r.message}"
        if r.commit_sha:
            line += f"\n            commit {r.commit_sha[:7]}" + (f"  {r.url}" if r.url else "")
        print(redact(line))
        if r.diff:
            print(redact("\n".join("            " + d for d in r.diff.splitlines())))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bot1", description="AI-powered GitHub file management bot")
    ap.add_argument("command", nargs="?", help="natural-language command (omit for interactive mode)")
    ap.add_argument("--repo", help="owner/name (default: GITHUB_REPO)")
    ap.add_argument("--branch", help="branch to use (default: repo default branch)")
    ap.add_argument("--dry-run", action="store_true", help="show what would happen, change nothing")
    ap.add_argument("--yes", action="store_true", help="auto-confirm risky operations")
    ap.add_argument("--no-ai", action="store_true", help="use the offline rule-based parser")
    ap.add_argument("--info", action="store_true", help="show repository analysis and exit")
    ap.add_argument("--install", action="store_true", help="upload the bot files into the repo (for GitHub automation)")
    ap.add_argument("--demo", action="store_true", help="practice on a fake in-memory repo (no token needed)")
    args = ap.parse_args(argv)

    logging.getLogger().addFilter(RedactingFilter())
    try:
        if args.demo:
            repo, engine = MemoryRepo(), make_engine(None, "", True)
            print("DEMO MODE: using a throw-away in-memory repository.")
        else:
            cfg = load_config(args.repo, args.branch)
            register_secrets(cfg.secrets())
            repo = GitHubRepo(cfg.github_token, cfg.repo, cfg.branch)
            engine = make_engine(cfg.anthropic_key, cfg.model, args.no_ai)
        print(f"Connected to {repo.name} (branch: {repo.branch}). Brain: {engine.name}.")
        bot = Bot(repo, engine, _confirm_factory(args.yes), print, args.dry_run)
        if args.install:
            from .install import install
            return 1 if install(repo, lambda m: print(redact(m))) else 0
        info = bot.refresh()
        if args.info:
            print("\n" + redact(info.summary))
            return 0
        if args.command:
            return _one(bot, args.command)
        print("Type a command (e.g. 'Create a Python calculator program'). Type 'exit' to quit.")
        while True:
            try:
                cmd = input("\nbot1> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if cmd.lower() in ("exit", "quit", "q"):
                return 0
            if cmd:
                _one(bot, cmd)
    except BotError as e:
        print(f"Error: {redact(str(e))}", file=sys.stderr)
        return 1


def _one(bot: Bot, command: str) -> int:
    try:
        results = bot.run(command)
    except BotError as e:
        print(f"Error: {redact(str(e))}", file=sys.stderr)
        return 1
    print_results(results)
    return 1 if any(r.status == "failed" for r in results) else 0
