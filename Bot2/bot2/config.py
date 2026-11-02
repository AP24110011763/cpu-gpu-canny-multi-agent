"""Configuration and environment management for Bot 2."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BotConfig:
    """Holds configuration parameters for Bot 2."""

    # GitHub credentials & context
    github_token: str
    github_repository: str
    github_event_path: str
    github_sha: str
    pr_number: Optional[int]

    # AI Model credentials & settings
    gemini_api_key: Optional[str]
    gemini_model: str
    require_ai_key: bool

    # Limits and execution toggles
    max_diff_bytes: int
    comment_marker: str = "<!-- bot-2-code-review -->"

    @classmethod
    def from_env(cls) -> BotConfig:
        """Constructs BotConfig by reading environment variables."""
        github_token = os.getenv("GITHUB_TOKEN", "").strip()
        github_repository = os.getenv("GITHUB_REPOSITORY", "").strip()
        github_event_path = os.getenv("GITHUB_EVENT_PATH", "").strip()
        github_sha = os.getenv("GITHUB_SHA", "HEAD").strip()

        # Optional PR number override (useful for local testing)
        pr_number_raw = os.getenv("PR_NUMBER", "").strip()
        pr_number = int(pr_number_raw) if pr_number_raw.isdigit() else None

        gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip() or None
        gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()
        require_ai_key = os.getenv("REQUIRE_AI_KEY", "false").strip().lower() in ("true", "1", "yes")

        max_diff_bytes_raw = os.getenv("MAX_DIFF_BYTES", "150000").strip()
        try:
            max_diff_bytes = int(max_diff_bytes_raw)
        except ValueError:
            max_diff_bytes = 150000

        return cls(
            github_token=github_token,
            github_repository=github_repository,
            github_event_path=github_event_path,
            github_sha=github_sha,
            pr_number=pr_number,
            gemini_api_key=gemini_api_key,
            gemini_model=gemini_model,
            require_ai_key=require_ai_key,
            max_diff_bytes=max_diff_bytes,
        )

    def validate_secrets(self, is_github_action: bool = True) -> tuple[bool, list[str]]:
        """
        Validates required secrets and outputs standard GitHub Actions annotations.
        Returns (is_valid, list_of_notices_or_errors).
        """
        messages: list[str] = []
        is_valid = True

        if is_github_action and not self.github_token:
            err = "CRITICAL: GITHUB_TOKEN is missing. Bot 2 cannot read PR data or post comments."
            messages.append(err)
            print(f"::error::{err}", file=sys.stderr)
            is_valid = False

        if not self.gemini_api_key:
            msg = (
                "NOTICE: GEMINI_API_KEY is not configured in repository secrets. "
                "Bot 2 will run static security and quality analysis, but AI semantic review is disabled. "
                "To enable AI reviews, add GEMINI_API_KEY under Settings -> Secrets and variables -> Actions."
            )
            messages.append(msg)
            if self.require_ai_key:
                print(f"::error::{msg}", file=sys.stderr)
                is_valid = False
            else:
                print(f"::warning::{msg}", file=sys.stderr)
        else:
            print("INFO: GEMINI_API_KEY detected. AI semantic review is enabled.")

        return is_valid, messages
