"""GitHub REST API Client for fetching PR diffs and posting review comments."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Optional

from bot2.diff_parser import DiffParser, FileDiff


class GitHubClient:
    """Interacts with GitHub REST API using Python standard library."""

    def __init__(self, token: str, repository: str, api_base: str = "https://api.github.com"):
        self.token = token.strip()
        self.repository = repository.strip()  # "owner/repo"
        self.api_base = api_base.rstrip("/")

    @property
    def headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "Bot-2-Code-Review-Action",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _request(
        self,
        endpoint: str,
        method: str = "GET",
        data: Optional[dict[str, Any]] = None,
        custom_headers: Optional[dict[str, str]] = None,
    ) -> tuple[int, bytes, dict[str, str]]:
        """Sends an authenticated HTTP request to GitHub API."""
        url = f"{self.api_base}{endpoint}" if endpoint.startswith("/") else endpoint
        headers = dict(self.headers)
        if custom_headers:
            headers.update(custom_headers)

        body_bytes = None
        if data is not None:
            body_bytes = json.dumps(data).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=body_bytes, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_headers = {k.lower(): v for k, v in resp.headers.items()}
                return resp.status, resp.read(), resp_headers
        except urllib.error.HTTPError as e:
            error_body = b""
            try:
                error_body = e.read()
            except Exception:
                pass
            print(f"ERROR: GitHub API {method} {url} returned HTTP {e.code}: {e.reason}")
            return e.code, error_body, {}
        except urllib.error.URLError as e:
            print(f"ERROR: GitHub API connection error: {e.reason}")
            return 0, str(e.reason).encode("utf-8"), {}

    def get_pull_request_files(self, pr_number: int) -> list[FileDiff]:
        """
        Fetches the list of changed files for a pull request using:
        GET /repos/{owner}/{repo}/pulls/{pull_number}/files
        """
        endpoint = f"/repos/{self.repository}/pulls/{pr_number}/files?per_page=100"
        status, body, _ = self._request(endpoint)

        if status != 200:
            print(f"WARN: Could not fetch PR files from {endpoint} (HTTP {status})")
            return []

        try:
            files_data = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return []

        file_diffs: list[FileDiff] = []
        for item in files_data:
            filename = item.get("filename", "")
            patch = item.get("patch", "")
            status_str = item.get("status", "modified")
            additions = item.get("additions", 0)
            deletions = item.get("deletions", 0)

            changed_lines = DiffParser.parse_patch_hunks(filename, patch) if patch else []

            file_diffs.append(
                FileDiff(
                    filename=filename,
                    status=status_str,
                    additions=additions,
                    deletions=deletions,
                    patch=patch,
                    changed_lines=changed_lines,
                )
            )

        return file_diffs

    def get_pull_request_raw_diff(self, pr_number: int) -> str:
        """
        Fetches the raw unified diff for a pull request using:
        GET /repos/{owner}/{repo}/pulls/{pull_number} with diff media type
        """
        endpoint = f"/repos/{self.repository}/pulls/{pr_number}"
        custom_headers = {"Accept": "application/vnd.github.v3.diff"}
        status, body, _ = self._request(endpoint, custom_headers=custom_headers)

        if status == 200:
            return body.decode("utf-8", errors="replace")
        return ""

    def post_or_update_review_comment(self, pr_number: int, comment_body: str, marker: str) -> bool:
        """
        Finds any existing review comment containing `marker` and updates it via PATCH.
        If no such comment exists, creates a new one via POST.
        This guarantees zero duplicate comments across repeated commits/pushes.
        """
        existing_comment_id = self._find_comment_with_marker(pr_number, marker)

        if existing_comment_id is not None:
            # Update existing comment
            endpoint = f"/repos/{self.repository}/issues/comments/{existing_comment_id}"
            status, _, _ = self._request(endpoint, method="PATCH", data={"body": comment_body})
            if status in (200, 201):
                print(f"INFO: Successfully updated existing Bot 2 review comment (ID: {existing_comment_id}).")
                return True
            print(f"WARN: Failed to update comment {existing_comment_id} (HTTP {status}). Falling back to new comment.")

        # Post new comment
        endpoint = f"/repos/{self.repository}/issues/{pr_number}/comments"
        status, _, _ = self._request(endpoint, method="POST", data={"body": comment_body})
        if status in (200, 201):
            print("INFO: Successfully created new Bot 2 review comment.")
            return True

        print(f"ERROR: Failed to post review comment to PR #{pr_number} (HTTP {status}).")
        return False

    def _find_comment_with_marker(self, pr_number: int, marker: str) -> Optional[int]:
        """Finds ID of an existing comment containing the marker."""
        endpoint = f"/repos/{self.repository}/issues/{pr_number}/comments?per_page=100"
        status, body, _ = self._request(endpoint)
        if status != 200:
            return None

        try:
            comments = json.loads(body.decode("utf-8"))
            for c in comments:
                if marker in c.get("body", ""):
                    return c.get("id")
        except Exception as e:
            print(f"WARN: Error searching existing comments: {e}")

        return None
