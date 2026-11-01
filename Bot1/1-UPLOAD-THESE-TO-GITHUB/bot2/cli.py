"""Bot 2 command line.  Automatic (GitHub Actions): organize, activity.  Local: push, watch."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .config import load_config
from .github_client import GitHubRepo
from .errors import BotError
from .safety import redact, register_secrets

from .activity import GitHubLog, build, post
from .ai_placer import make_placer
from .organize import format_report, organize, push_branch
from .push import format_result, push_files, watch


def _model() -> str:
    return os.environ.get("BOT_MODEL", "").strip() or "claude-sonnet-5-5"


def cmd_organize() -> int:
    placer = make_placer(os.environ.get("ANTHROPIC_API_KEY", "").strip() or None, _model())
    moves, committed = organize(Path("."), placer)
    if not moves:
        print("Inbox is empty - nothing to do.")
        return 0
    if committed:
        branch = os.environ.get("GITHUB_REF_NAME") or subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True).stdout.strip()
        push_branch(Path("."), branch)
    report = format_report(moves)
    print(redact(report))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(report + "\n")
    token, repo = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPOSITORY")
    if token and repo:
        try:
            post(GitHubLog(token, repo, os.environ.get("GITHUB_API_URL")), report)
        except Exception as e:
            print(f"Could not post to the activity log: {type(e).__name__}")
    return 0


def cmd_activity() -> int:
    with open(os.environ["GITHUB_EVENT_PATH"], encoding="utf-8") as f:
        event = json.load(f)
    text = build(os.environ.get("GITHUB_EVENT_NAME", ""), event)
    if not text:
        print("Nothing worth reporting.")
        return 0
    n = post(GitHubLog(os.environ["GITHUB_TOKEN"], os.environ["GITHUB_REPOSITORY"], os.environ.get("GITHUB_API_URL")), redact(text))
    print(f"Posted to activity log issue #{n}")
    return 0


def _connect(args):
    cfg = load_config(args.repo, args.branch)
    register_secrets(cfg.secrets())
    return GitHubRepo(cfg.github_token, cfg.repo, cfg.branch), make_placer(cfg.anthropic_key, cfg.model)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bot2", description="Bot 2 - file organizer and repo reporter")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("organize", help="(GitHub Actions) file everything in inbox/")
    sub.add_parser("activity", help="(GitHub Actions) post the current event to the activity log")
    pu = sub.add_parser("push", help="push local file(s) into the right folder of your repo")
    pu.add_argument("files", nargs="+")
    pu.add_argument("--to", help="force a destination folder")
    pw = sub.add_parser("watch", help="auto-push every file you drop into a local folder")
    pw.add_argument("folder")
    pw.add_argument("--interval", type=float, default=5)
    for s in (pu, pw):
        s.add_argument("--repo"), s.add_argument("--branch"), s.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "organize":
            return cmd_organize()
        if args.cmd == "activity":
            return cmd_activity()
        repo, placer = _connect(args)
        print(f"Connected to {repo.name} (branch: {repo.branch}).")
        if args.cmd == "push":
            paths = [Path(f).expanduser() for f in args.files]
            missing = [str(p) for p in paths if not p.is_file()]
            if missing:
                raise BotError("Not a file: " + ", ".join(missing))
            results = push_files(repo, paths, args.to, placer, args.dry_run)
            for r in results:
                print(redact(format_result(r)))
            return 1 if any(r.status in ("failed", "refused") for r in results) else 0
        folder = Path(args.folder).expanduser()
        if not folder.is_dir():
            raise BotError(f"Folder not found: {folder}")
        try:
            watch(repo, folder, args.interval, lambda m: print(redact(m)), placer)
        except KeyboardInterrupt:
            print("\nStopped.")
        return 0
    except BotError as e:
        print(f"Error: {redact(str(e))}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
