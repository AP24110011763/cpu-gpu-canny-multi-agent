#!/usr/bin/env python3
"""Local testing and demonstration CLI for Bot 2.

Allows running Bot 2 locally against sample patch files, git diffs,
or unstaged changes without requiring an active GitHub Pull Request.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from bot2.config import BotConfig
from bot2.diff_parser import DiffParser
from bot2.reviewer import ReviewOrchestrator


def get_git_diff() -> str:
    """Retrieves unstaged and staged git diff from current repository."""
    try:
        # Check staged + unstaged changes vs HEAD
        diff_proc = subprocess.run(
            ["git", "diff", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        if diff_proc.stdout.strip():
            return diff_proc.stdout

        # Fallback to last commit diff if working directory is clean
        commit_proc = subprocess.run(
            ["git", "diff", "HEAD~1..HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return commit_proc.stdout
    except Exception as e:
        print(f"WARN: Could not fetch git diff: {e}")
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Bot 2 autonomous code review locally on a patch file or git diff."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--file", "-f",
        help="Path to a unified diff/patch file to review (e.g., sample_pr_diffs/buggy_change.patch).",
    )
    group.add_argument(
        "--git", "-g",
        action="store_true",
        help="Inspect current local git diff (unstaged/staged changes or latest commit).",
    )

    parser.add_argument(
        "--api-key",
        default=os.getenv("GEMINI_API_KEY", ""),
        help="Google Gemini API key for AI-powered semantic review (falls back to GEMINI_API_KEY env var).",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.0-flash",
        help="Gemini model to use (default: gemini-2.0-flash).",
    )
    parser.add_argument(
        "--output", "-o",
        help="Optional path to write generated markdown review report.",
    )

    args = parser.parse_args()

    # Load diff content
    if args.file:
        if not os.path.isfile(args.file):
            print(f"ERROR: Diff file '{args.file}' not found.", file=sys.stderr)
            return 1
        with open(args.file, "r", encoding="utf-8") as f:
            raw_diff = f.read()
    else:
        raw_diff = get_git_diff()
        if not raw_diff.strip():
            print("INFO: No git diff found in current repository. Make changes or pass a --file.")
            return 0

    # Parse diff hunks
    file_diffs = DiffParser.parse_full_unified_diff(raw_diff)
    print(f"INFO: Parsed {len(file_diffs)} changed file(s) from diff.")

    # Configure Bot
    config = BotConfig(
        github_token="",
        github_repository="local/test-repo",
        github_event_path="",
        github_sha="local-demo-sha",
        pr_number=999,
        gemini_api_key=args.api_key or None,
        gemini_model=args.model,
        require_ai_key=False,
        max_diff_bytes=150000,
    )

    orchestrator = ReviewOrchestrator(config)

    secret_notice = None
    if not config.gemini_api_key:
        secret_notice = (
            "No GEMINI_API_KEY provided. Bot 2 ran in Static Rules Only mode. "
            "To enable AI semantic review, provide --api-key or export GEMINI_API_KEY."
        )

    markdown_review, findings = orchestrator.run_review(
        file_diffs=file_diffs,
        raw_diff=raw_diff,
        secret_notice=secret_notice,
    )

    print("\n" + "=" * 60)
    print("BOT 2 REVIEW OUTPUT")
    print("=" * 60)
    print(markdown_review)
    print("=" * 60 + "\n")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(markdown_review)
        print(f"INFO: Review saved to '{args.output}'.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
