"""Minimal GitHub client (PyGithub) used by the local push/watch mode."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .errors import BotError
from .safety import redact


@dataclass
class CommitInfo:
    sha: str
    url: str = ""


def translate_error(exc) -> BotError:
    status = getattr(exc, "status", None)
    data = getattr(exc, "data", None)
    msg = data.get("message", "") if isinstance(data, dict) else ""
    low = msg.lower()
    if status == 401:
        return BotError("GitHub authentication failed. Check that the token is valid and not expired.")
    if status == 403:
        if "rate limit" in low:
            return BotError("GitHub rate limit reached. Wait a few minutes and try again.")
        if "protected" in low:
            return BotError("The branch is protected, so the bot cannot push to it.")
        return BotError("Permission denied. The token needs 'Contents: Read and write' on this repository.")
    if status == 404:
        return BotError("Repository, branch or file not found (or the token has no access to it).")
    if status == 409:
        return BotError("The repository is empty or the branch changed. Add a first file (like a README) and retry.")
    if status == 422 and "sha" in low:
        return BotError("That file already exists in the repository; it was not overwritten.")
    return BotError(redact(f"GitHub API error {status}: {msg or 'request failed'}"))


class GitHubRepo:
    def __init__(self, token: str, full_name: str, branch: Optional[str] = None, base_url: Optional[str] = None):
        from github import Auth, Github, GithubException
        self._exc = GithubException
        kwargs = {"auth": Auth.Token(token)}
        if base_url:
            kwargs["base_url"] = base_url
        self.name = full_name
        try:
            self._repo = Github(**kwargs).get_repo(full_name)
            self.branch = branch or self._repo.default_branch
            self._repo.get_branch(self.branch)
        except GithubException as e:
            raise translate_error(e) from None

    def tree_shas(self) -> Dict[str, str]:
        """path -> git blob sha for every file (used to spot duplicates without downloading)."""
        try:
            tree = self._repo.get_git_tree(self.branch, recursive=True)
        except self._exc as e:
            if getattr(e, "status", None) == 409:
                return {}
            raise translate_error(e) from None
        return {t.path: t.sha for t in tree.tree if t.type == "blob"}

    def create_bytes(self, path: str, data: bytes, message: str) -> CommitInfo:
        """Create a file from raw bytes (PDF, image, zip...). Never overwrites."""
        try:
            res = self._repo.create_file(path, message, data, branch=self.branch)
        except self._exc as e:
            raise translate_error(e) from None
        c = res["commit"]
        return CommitInfo(sha=c.sha, url=getattr(c, "html_url", "") or "")

    def delete_path(self, path: str, message: str) -> CommitInfo:
        """Delete one file (it stays recoverable from the git history)."""
        shas = self.tree_shas()
        if path not in shas:
            raise BotError(f"'{path}' was not found in the repository.")
        try:
            res = self._repo.delete_file(path, message, shas[path], branch=self.branch)
        except self._exc as e:
            raise translate_error(e) from None
        c = res["commit"]
        return CommitInfo(sha=c.sha, url=getattr(c, "html_url", "") or "")

    def move_path(self, old: str, new: str, message: str) -> CommitInfo:
        """Move/rename a file: create it at the new path, then delete the old one. Never overwrites."""
        shas = self.tree_shas()
        if old not in shas:
            raise BotError(f"'{old}' was not found in the repository.")
        if new in shas:
            raise BotError(f"'{new}' already exists, so nothing was moved.")
        try:
            import base64
            blob = self._repo.get_git_blob(shas[old])
            data = base64.b64decode(blob.content) if blob.encoding == "base64" else blob.content.encode()
            self._repo.create_file(new, message, data, branch=self.branch)
            res = self._repo.delete_file(old, message, shas[old], branch=self.branch)
        except self._exc as e:
            raise translate_error(e) from None
        c = res["commit"]
        return CommitInfo(sha=c.sha, url=getattr(c, "html_url", "") or "")
