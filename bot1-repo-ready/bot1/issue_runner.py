"""Automatic mode: run Bot 1 from GitHub Issues (no command line needed).

Open an issue titled  "bot: create a login.java file"  and the bot does it,
replies on the issue with the result, and closes it.
Risky actions (deleting important/many files) wait for a comment: confirm
"""
from __future__ import annotations

import json
import os
import sys
from typing import Callable, List, Optional

from .ai_engine import make_engine
from .bot import Bot
from .config import load_config
from .github_client import GitHubRepo
from .models import BotError
from .security import redact, register_secrets

TRUSTED = {"OWNER", "MEMBER", "COLLABORATOR"}
PREFIX = "bot:"
PENDING = "<!-- bot1-pending -->"
DONE = "<!-- bot1-done -->"
CONFIRM_WORDS = {"confirm", "/confirm", "yes"}
ICON = {"created": "✅ created", "updated": "✏️ updated", "deleted": "🗑️ deleted", "skipped": "⏭️ skipped",
        "failed": "❌ failed", "cancelled": "⛔ cancelled", "dry-run": "🔎 dry-run"}


class GitHubIssues:
    def __init__(self, token: str, repo: str, base_url: Optional[str] = None):
        from github import Auth, Github
        kw = {"auth": Auth.Token(token)}
        if base_url:
            kw["base_url"] = base_url
        self._repo = Github(**kw).get_repo(repo)

    def comments(self, n: int) -> List[str]:
        return [c.body or "" for c in self._repo.get_issue(n).get_comments()]

    def comment(self, n: int, body: str) -> None:
        self._repo.get_issue(n).create_comment(body)

    def close(self, n: int) -> None:
        self._repo.get_issue(n).edit(state="closed", state_reason="completed")


def _format(command: str, results) -> str:
    lines = [f"**Bot 1** ran: `{command}`", ""]
    for r in results:
        line = f"- {ICON.get(r.status, r.status)} `{r.action.path}` - {r.message}"
        if r.commit_sha:
            line += f" ([commit {r.commit_sha[:7]}]({r.url}))" if r.url else f" (commit {r.commit_sha[:7]})"
        lines.append(line)
        if r.diff:
            lines += ["", "```diff", r.diff, "```"]
    return "\n".join(lines)


def handle(event_name: str, event: dict, issues, build_bot: Callable[[Callable[[str], bool]], Bot]) -> str:
    """Process one GitHub event. Returns a short status string (useful for tests/logs)."""
    issue = event.get("issue") or {}
    n = issue.get("number")
    title = (issue.get("title") or "").strip()
    if n is None or issue.get("pull_request") or not title.lower().startswith(PREFIX):
        return "ignored: not a bot issue"
    if (event.get("sender") or {}).get("type") == "Bot":
        return "ignored: sent by a bot"
    if issue.get("author_association") not in TRUSTED:
        return "ignored: issue author is not the owner or a collaborator"

    approved = False
    if event_name == "issues":
        if event.get("action") != "opened":
            return "ignored: only newly opened issues run"
        command = title[len(PREFIX):].strip() or (issue.get("body") or "").strip()
    elif event_name == "issue_comment":
        comment = event.get("comment") or {}
        if event.get("action") != "created" or (comment.get("body") or "").strip().lower() not in CONFIRM_WORDS:
            return "ignored: not a confirmation"
        if comment.get("author_association") not in TRUSTED:
            return "ignored: commenter is not the owner or a collaborator"
        marks = [m for c in issues.comments(n) for m in (PENDING, DONE) if m in c]
        if not marks or marks[-1] != PENDING:
            return "ignored: nothing is waiting for confirmation"
        command, approved = title[len(PREFIX):].strip() or (issue.get("body") or "").strip(), True
    else:
        return "ignored: unsupported event"
    if not command:
        issues.comment(n, "Please write the command in the issue title, e.g. `bot: create login.java`." + DONE)
        return "empty command"

    pending: List[str] = []

    def confirm(prompt: str) -> bool:
        if approved:
            return True
        pending.append(prompt)
        return False

    try:
        results = build_bot(confirm).run(command)
    except BotError as e:
        issues.comment(n, redact(f"❌ Bot 1 could not do that: {e}") + "\n" + DONE)
        return "error"
    if pending:
        issues.comment(n, redact(f"⚠️ **Bot 1 needs your OK first.**\n\n{pending[0]}\n\n"
                                 "Reply with the single word `confirm` to go ahead, or ignore this to cancel.") + "\n" + PENDING)
        return "waiting for confirmation"
    issues.comment(n, redact(_format(command, results)) + "\n" + DONE)
    if not any(r.status == "failed" for r in results):
        issues.close(n)
    return "done"


def main() -> int:
    cfg = load_config(repo_override=os.environ.get("GITHUB_REPOSITORY"))
    register_secrets(cfg.secrets())
    with open(os.environ["GITHUB_EVENT_PATH"], encoding="utf-8") as f:
        event = json.load(f)
    issues = GitHubIssues(cfg.github_token, cfg.repo)

    def build(confirm):
        return Bot(GitHubRepo(cfg.github_token, cfg.repo, cfg.branch),
                   make_engine(cfg.anthropic_key, cfg.model), confirm, log=print)
    try:
        print(handle(os.environ.get("GITHUB_EVENT_NAME", ""), event, issues, build))
        return 0
    except Exception as e:  # last resort: tell the user, never leak details
        n = (event.get("issue") or {}).get("number")
        msg = redact(f"❌ Bot 1 crashed: {type(e).__name__}. See the Actions log.")
        if n:
            try:
                issues.comment(n, msg + "\n" + DONE)
            except Exception:
                pass
        print(msg, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
