"""CLI Entrypoint for GitHub Actions execution of Bot 2."""

from __future__ import annotations

import json
import os
import sys

from bot2.config import BotConfig
from bot2.diff_parser import DiffParser
from bot2.reviewer import ReviewOrchestrator


def parse_event_payload(event_path: str) -> tuple[int, str]:
    """Extracts PR number and head SHA from GITHUB_EVENT_PATH JSON file."""
    if not event_path or not os.path.exists(event_path):
        return 0, ""

    try:
        with open(event_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        pr_info = payload.get("pull_request", {})
        pr_number = pr_info.get("number", 0)
        head_sha = pr_info.get("head", {}).get("sha", "")
        return pr_number, head_sha
    except Exception as e:
        print(f"WARN: Failed reading event payload at {event_path}: {e}")
        return 0, ""


def main() -> int:
    """Main execution flow for Bot 2 GitHub Action."""
    print("==================================================")
    print("🚀 Bot 2: Autonomous GitHub Code Review and Quality Bot")
    print("==================================================")

    config = BotConfig.from_env()

    # Determine PR Number & SHA
    pr_number = config.pr_number
    commit_sha = config.github_sha

    if not pr_number and config.github_event_path:
        event_pr, event_sha = parse_event_payload(config.github_event_path)
        pr_number = event_pr or pr_number
        commit_sha = event_sha or commit_sha

    print(f"Repository: {config.github_repository or 'Local/Unknown'}")
    print(f"Pull Request: #{pr_number or 'N/A'}")
    print(f"Commit SHA: {commit_sha[:7] if commit_sha else 'Unknown'}")

    # Validate secrets
    is_valid, notices = config.validate_secrets(is_github_action=bool(config.github_event_path))
    if not is_valid:
        print("ERROR: Critical validation failure in Bot 2 secrets. Aborting.")
        return 1

    secret_notice = None
    if not config.gemini_api_key:
        secret_notice = (
            "The `GEMINI_API_KEY` secret is not configured in this repository. "
            "Bot 2 performed static code analysis, security checks, and PR hygiene evaluation. "
            "To enable AI-powered semantic reviews, add `GEMINI_API_KEY` to GitHub Repository Secrets."
        )

    orchestrator = ReviewOrchestrator(config)

    # Fetch PR files and diff
    if not pr_number:
        print("ERROR: Could not identify pull request number. Ensure GITHUB_EVENT_PATH is set.")
        return 1

    print(f"INFO: Fetching changed files for Pull Request #{pr_number}...")
    file_diffs = orchestrator.github_client.get_pull_request_files(pr_number)

    # Fallback: if patch hunks were not included in files API (e.g., large files), fetch raw diff
    raw_diff = orchestrator.github_client.get_pull_request_raw_diff(pr_number)
    if not file_diffs and raw_diff:
        print("INFO: Parsing file diffs from raw unified diff...")
        file_diffs = DiffParser.parse_full_unified_diff(raw_diff)

    print(f"INFO: Detected {len(file_diffs)} changed file(s).")

    # Run review
    markdown_review, findings = orchestrator.run_review(
        file_diffs=file_diffs,
        raw_diff=raw_diff,
        secret_notice=secret_notice,
    )

    print("\n---------------- REVIEW PREVIEW ----------------")
    print(markdown_review)
    print("------------------------------------------------\n")

    # Post or update PR comment
    if config.github_token:
        print(f"INFO: Publishing review to Pull Request #{pr_number}...")
        success = orchestrator.publish_review(pr_number, markdown_review)
        if not success:
            print("ERROR: Failed to publish review comment to GitHub PR.")
            return 1
        print("✅ Review comment successfully published / updated.")
    else:
        print("WARN: No GITHUB_TOKEN provided; skipping PR comment publishing (dry-run mode).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
