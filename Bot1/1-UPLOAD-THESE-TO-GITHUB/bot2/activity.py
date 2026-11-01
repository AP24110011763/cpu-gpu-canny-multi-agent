"""Repo activity updates: posts what happens in your repo to one 'Activity Log' issue."""
from __future__ import annotations

from typing import List, Optional

LOG_TITLE = "📋 Repo Activity Log (Bot 2)"
LABEL = "bot2-activity"


class GitHubLog:
    def __init__(self, token: str, repo: str, base_url: Optional[str] = None):
        from github import Auth, Github, GithubException
        self._exc = GithubException
        kw = {"auth": Auth.Token(token)}
        if base_url:
            kw["base_url"] = base_url
        self._repo = Github(**kw).get_repo(repo)

    def find(self) -> Optional[int]:
        for i in self._repo.get_issues(state="open", labels=[LABEL]):
            if i.title == LOG_TITLE:          # label + exact title identify the log issue
                return i.number
        return None

    def create(self) -> int:
        try:
            self._repo.create_label(LABEL, "0e8a16", "Used by Bot 2 for the activity log")
        except self._exc:
            pass                                    # label already exists
        body = ("This issue is the live activity feed for your repository, written by Bot 2.\n"
                "Every push, pull request and issue event appears below as a comment.\n"
                "Mute or unsubscribe from this issue any time.")
        kwargs = {"labels": [LABEL]}
        if getattr(self._repo.owner, "type", "") == "User":
            kwargs["assignees"] = [self._repo.owner.login]   # so you get a notification
        try:
            return self._repo.create_issue(LOG_TITLE, body, **kwargs).number
        except self._exc:
            kwargs.pop("assignees", None)
            return self._repo.create_issue(LOG_TITLE, body, **kwargs).number

    def comment(self, number: int, body: str) -> None:
        self._repo.get_issue(number).create_comment(body)


def post(log, text: str) -> int:
    n = log.find() or log.create()
    log.comment(n, text)
    return n


def _short(msg: str, n: int = 80) -> str:
    first = (msg or "").splitlines()[0] if msg else ""
    return first if len(first) <= n else first[: n - 1] + "…"


def format_push(ev: dict) -> Optional[str]:
    ref = ev.get("ref", "")
    who = (ev.get("pusher") or {}).get("name") or (ev.get("sender") or {}).get("login", "someone")
    if ref.startswith("refs/tags/"):
        return f"🏷️ Tag `{ref[10:]}` {'deleted' if ev.get('deleted') else 'pushed'} by **{who}**"
    branch = ref.replace("refs/heads/", "")
    if ev.get("deleted"):
        return f"🗑️ Branch `{branch}` was deleted by **{who}**"
    commits: List[dict] = ev.get("commits") or []
    if not commits:
        return None
    head = f"🔔 **{len(commits)} commit(s)** pushed to `{branch}` by **{who}**" + (" (force push)" if ev.get("forced") else "")
    lines = [head, ""]
    for c in commits[:10]:
        a, m, r = c.get("added") or [], c.get("modified") or [], c.get("removed") or []
        sha = (c.get("id") or "")[:7]
        lines.append(f"- [`{sha}`]({c.get('url', '')}) {_short(c.get('message', ''))}  (+{len(a)} ~{len(m)} -{len(r)})")
        for sign, group in (("added", a), ("changed", m), ("removed", r)):
            if group:
                shown = ", ".join(f"`{f}`" for f in group[:6]) + (f" and {len(group) - 6} more" if len(group) > 6 else "")
                lines.append(f"  - {sign}: {shown}")
    if len(commits) > 10:
        lines.append(f"- …and {len(commits) - 10} more commit(s)")
    if ev.get("compare"):
        lines += ["", f"[See all changes]({ev['compare']})"]
    return "\n".join(lines)


def format_pr(ev: dict) -> Optional[str]:
    pr, action = ev.get("pull_request") or {}, ev.get("action")
    if action not in ("opened", "closed", "reopened"):
        return None
    state = "merged" if action == "closed" and pr.get("merged") else action
    icon = {"opened": "🔀", "merged": "✅", "closed": "🚫", "reopened": "🔀"}[state]
    return (f"{icon} Pull request #{pr.get('number')} **{state}** by **{(pr.get('user') or {}).get('login', '?')}**: "
            f"[{_short(pr.get('title', ''), 100)}]({pr.get('html_url', '')}) "
            f"(`{(pr.get('head') or {}).get('ref', '?')}` → `{(pr.get('base') or {}).get('ref', '?')}`)")


def format_issue(ev: dict) -> Optional[str]:
    i, action = ev.get("issue") or {}, ev.get("action")
    title = i.get("title") or ""
    if action not in ("opened", "closed", "reopened") or title == LOG_TITLE:
        return None
    if (ev.get("sender") or {}).get("type") == "Bot":
        return None
    icon = {"opened": "🐞", "closed": "✅", "reopened": "🔁"}[action]
    return f"{icon} Issue #{i.get('number')} **{action}** by **{(ev.get('sender') or {}).get('login', '?')}**: [{_short(title, 100)}]({i.get('html_url', '')})"


def build(event_name: str, ev: dict) -> Optional[str]:
    if event_name == "push":
        return format_push(ev)
    if event_name == "pull_request":
        return format_pr(ev)
    if event_name == "issues":
        return format_issue(ev)
    return None
