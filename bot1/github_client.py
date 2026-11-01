"""GitHub integration (PyGithub). Every commit is made through the GitHub
Contents API, so a successful call is already "committed AND pushed"."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple

from .models import BotError, ConflictError
from .security import MAX_FILE_BYTES, redact


@dataclass
class CommitInfo:
    sha: str
    url: str = ""


class RepoBackend(Protocol):
    name: str
    branch: str

    def list_files(self) -> List[str]: ...
    def get_sha(self, path: str) -> Optional[str]: ...
    def read(self, path: str) -> Optional[Tuple[str, str]]: ...      # (text, sha) | None
    def create(self, path: str, text: str, message: str) -> CommitInfo: ...
    def update(self, path: str, text: str, sha: str, message: str) -> CommitInfo: ...
    def delete(self, path: str, sha: str, message: str) -> CommitInfo: ...


def translate_error(exc) -> BotError:
    """Turn a GithubException into a safe, helpful message (never leaks secrets)."""
    status = getattr(exc, "status", None)
    data = getattr(exc, "data", None)
    msg = data.get("message", "") if isinstance(data, dict) else ""
    low = msg.lower()
    if status == 401:
        return BotError("GitHub authentication failed. Check that GITHUB_TOKEN is valid and not expired.")
    if status == 403:
        if "rate limit" in low:
            return BotError("GitHub rate limit reached. Wait a few minutes and try again.")
        if "protected" in low:
            return BotError("The branch is protected. Set BOT_BRANCH to a different branch.")
        return BotError("Permission denied. The token needs 'Contents: Read and write' on this repository "
                        "(modifying .github/workflows also needs the 'Workflows' permission).")
    if status == 404:
        return BotError("Repository, branch or file not found (or the token has no access to it).")
    if status == 409:
        return ConflictError("The branch changed while the bot was working (conflict).")
    if status == 422 and ("sha" in low or "does not match" in low):
        return ConflictError("The file changed on GitHub since it was read (conflict).")
    if status == 422 and "protected" in low:
        return BotError("The branch is protected. Set BOT_BRANCH to a different branch.")
    return BotError(redact(f"GitHub API error {status}: {msg or 'request failed'}"))


class GitHubRepo:
    def __init__(self, token: str, full_name: str, branch: Optional[str] = None,
                 base_url: Optional[str] = None):
        from github import Auth, Github, GithubException
        self._exc = GithubException
        kwargs = {"auth": Auth.Token(token)}
        if base_url:
            kwargs["base_url"] = base_url
        self._gh = Github(**kwargs)
        self.name = full_name
        try:
            self._repo = self._gh.get_repo(full_name)
            self.default_branch = self._repo.default_branch
            self.branch = branch or self.default_branch
            self._repo.get_branch(self.branch)  # verifies the branch exists
        except GithubException as e:
            raise translate_error(e) from None

    # ---- reading ----
    def list_files(self) -> List[str]:
        try:
            tree = self._repo.get_git_tree(self.branch, recursive=True)
        except self._exc as e:
            if getattr(e, "status", None) == 409:  # empty repository
                return []
            raise translate_error(e) from None
        return sorted(t.path for t in tree.tree if t.type == "blob")

    def _contents(self, path: str):
        try:
            return self._repo.get_contents(path, ref=self.branch)
        except self._exc as e:
            if getattr(e, "status", None) == 404:
                return None
            raise translate_error(e) from None

    def get_sha(self, path: str) -> Optional[str]:
        cf = self._contents(path)
        if cf is None:
            return None
        if isinstance(cf, list):
            raise BotError(f"'{path}' is a folder, not a file.")
        return cf.sha

    def read(self, path: str) -> Optional[Tuple[str, str]]:
        cf = self._contents(path)
        if cf is None:
            return None
        if isinstance(cf, list):
            raise BotError(f"'{path}' is a folder, not a file.")
        if cf.size > MAX_FILE_BYTES:
            raise BotError(f"'{path}' is larger than {MAX_FILE_BYTES // 1000} KB; the bot will not edit it.")
        try:
            return cf.decoded_content.decode("utf-8"), cf.sha
        except UnicodeDecodeError:
            raise BotError(f"'{path}' looks like a binary file; the bot only edits text files.") from None

    # ---- writing ----
    def _commit(self, res) -> CommitInfo:
        c = res["commit"]
        return CommitInfo(sha=c.sha, url=getattr(c, "html_url", "") or "")

    def create(self, path, text, message) -> CommitInfo:
        try:
            return self._commit(self._repo.create_file(path, message, text, branch=self.branch))
        except self._exc as e:
            if getattr(e, "status", None) == 422 and "sha" in str((e.data or {}).get("message", "")).lower():
                raise BotError(f"'{path}' already exists; it was not overwritten.") from None
            raise translate_error(e) from None

    def update(self, path, text, sha, message) -> CommitInfo:
        try:
            return self._commit(self._repo.update_file(path, message, text, sha, branch=self.branch))
        except self._exc as e:
            raise translate_error(e) from None

    def delete(self, path, sha, message) -> CommitInfo:
        try:
            return self._commit(self._repo.delete_file(path, message, sha, branch=self.branch))
        except self._exc as e:
            raise translate_error(e) from None
