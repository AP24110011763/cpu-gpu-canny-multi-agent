"""Tiny stdlib GitHub client for the team layer (comments, dispatch events, issues, files)."""
from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

LOG_TITLE = "📋 Repo Activity Log (Bot 1)"      # same issue Bot 1 already uses
LOG_LABEL = "bot1-activity"


class GH:
    def __init__(self, token: str, repo: str, api: Optional[str] = None):
        self.token, self.repo = token, repo
        self.api = (api or os.environ.get("GITHUB_API_URL") or "https://api.github.com").rstrip("/")

    @property
    def owner(self) -> str:
        return self.repo.split("/")[0]

    def _req(self, method: str, path: str, data: Any = None) -> Tuple[int, Any]:
        body = json.dumps(data).encode() if data is not None else None
        req = urllib.request.Request(self.api + path, data=body, method=method, headers={
            "Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "team-bots", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
                return r.status, (json.loads(raw) if raw else None)
        except urllib.error.HTTPError as e:
            return e.code, None
        except urllib.error.URLError:
            return 0, None

    # --- conversation ---
    def comment(self, number: int, body: str) -> None:
        st, _ = self._req("POST", f"/repos/{self.repo}/issues/{number}/comments", {"body": body})
        if st not in (200, 201):
            raise RuntimeError(f"could not post comment on #{number} (HTTP {st})")

    def find_comment(self, number: int, needle: str) -> bool:
        for page in (1, 2, 3):
            st, items = self._req("GET", f"/repos/{self.repo}/issues/{number}/comments?per_page=100&page={page}")
            if st != 200 or not items:
                return False
            if any(needle in (c.get("body") or "") for c in items):
                return True
        return False

    def log_issue(self) -> int:
        """The Activity Log issue Bot 1 owns; created if missing."""
        st, items = self._req("GET", f"/repos/{self.repo}/issues?state=open&labels={LOG_LABEL}&per_page=50")
        for i in items or []:
            if i.get("title") == LOG_TITLE and "pull_request" not in i:
                return i["number"]
        return self.create_issue(LOG_TITLE, "Live activity feed and team conversation between you, Bot 1 and Bot 2.", [LOG_LABEL])

    def create_issue(self, title: str, body: str, labels: Optional[List[str]] = None, assignees: Optional[List[str]] = None) -> int:
        payload: Dict[str, Any] = {"title": title, "body": body, "labels": labels or []}
        if assignees:
            payload["assignees"] = assignees
        st, res = self._req("POST", f"/repos/{self.repo}/issues", payload)
        if st == 422 and assignees:                   # assignee not allowed: retry without
            payload.pop("assignees")
            st, res = self._req("POST", f"/repos/{self.repo}/issues", payload)
        if st not in (200, 201):
            raise RuntimeError(f"could not create issue (HTTP {st})")
        return res["number"]

    # --- bot <-> bot ---
    def dispatch(self, event_type: str, payload: Dict[str, Any]) -> None:
        """repository_dispatch wakes the other bot's workflow (comments made with GITHUB_TOKEN can't)."""
        st, _ = self._req("POST", f"/repos/{self.repo}/dispatches", {"event_type": event_type, "client_payload": payload})
        if st not in (200, 201, 204):
            raise RuntimeError(f"could not send '{event_type}' (HTTP {st})")

    # --- reading the repo ---
    def get_file(self, path: str, ref: Optional[str] = None) -> Optional[str]:
        q = f"?ref={urllib.parse.quote(ref)}" if ref else ""
        st, res = self._req("GET", f"/repos/{self.repo}/contents/{urllib.parse.quote(path)}{q}")
        if st != 200 or not isinstance(res, dict) or res.get("encoding") != "base64":
            return None
        try:
            return base64.b64decode(res["content"]).decode("utf-8")
        except UnicodeDecodeError:
            return None

    def recent_commits(self, days: int = 7, limit: int = 30) -> List[Dict[str, Any]]:
        since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        st, items = self._req("GET", f"/repos/{self.repo}/commits?since={since}&per_page={limit}")
        out = []
        for c in items or []:
            st2, full = self._req("GET", f"/repos/{self.repo}/commits/{c['sha']}")
            files = [f["filename"] for f in (full or {}).get("files", [])] if st2 == 200 else []
            out.append({"sha": c["sha"][:7], "message": c["commit"]["message"].splitlines()[0],
                        "author": (c.get("author") or {}).get("login") or c["commit"]["author"]["name"], "files": files})
        return out
