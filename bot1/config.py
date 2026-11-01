"""Configuration comes ONLY from environment variables (or a local .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from .models import BotError


@dataclass(frozen=True)
class Config:
    github_token: str
    repo: str
    anthropic_key: Optional[str] = None
    model: str = "claude-sonnet-5-5"
    branch: Optional[str] = None

    def secrets(self):
        return [s for s in (self.github_token, self.anthropic_key) if s]


def load_config(repo_override: Optional[str] = None, branch_override: Optional[str] = None) -> Config:
    try:  # .env is optional; ignore if python-dotenv is missing
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    repo = (repo_override or os.environ.get("GITHUB_REPO", "")).strip()
    if not token:
        raise BotError("GITHUB_TOKEN is not set. Put it in your environment or a local .env file.")
    if repo.count("/") != 1 or not all(repo.split("/")):
        raise BotError("GITHUB_REPO must look like 'owner/name'.")
    return Config(
        github_token=token,
        repo=repo,
        anthropic_key=os.environ.get("ANTHROPIC_API_KEY", "").strip() or None,
        model=os.environ.get("BOT_MODEL", "").strip() or "claude-sonnet-5-5",
        branch=(branch_override or os.environ.get("BOT_BRANCH", "")).strip() or None,
    )
