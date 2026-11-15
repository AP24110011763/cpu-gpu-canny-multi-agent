"""AI-powered code review engine leveraging Google Gemini Flash models."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Optional

from bot2.diff_parser import FileDiff
from bot2.static_analyzer import ReviewFinding


SYSTEM_PROMPT = """You are Bot 2, an autonomous Senior Software Engineer and Code Reviewer.
Your task is to conduct an objective, high-precision code review on a GitHub Pull Request diff.

GUIDELINES:
1. Identify likely bugs, logic errors, race conditions, edge-case failures, unsafe patterns, and maintainability issues.
2. Check if relevant unit/integration tests or documentation appear to be missing for new features.
3. Focus strictly on changed lines (prefixed with + in diffs).
4. For every finding, provide the file path and specific line number whenever possible.
5. Give concrete, actionable suggestions rather than vague criticism. Include snippet fixes where helpful.
6. DO NOT report formatting preferences (spaces vs tabs, line length, trailing commas, quote style).
7. DO NOT report speculative issues as definite bugs. If code looks clean, say clearly that no important issues were found.
8. Set status to "Issues found" if bugs, security risks, or maintainability issues are discovered; otherwise set status to "No significant issues found".

OUTPUT FORMAT:
You MUST respond with ONLY valid JSON (no markdown fences, no surrounding commentary) adhering strictly to this schema:
{
  "status": "Issues found" | "No significant issues found",
  "summary": "Concise 2-3 sentence overview of the PR quality and findings.",
  "findings": [
    {
      "file_path": "relative/path/to/file.ext",
      "line_number": 42,
      "category": "Bug Risk" | "Security" | "Maintainability" | "Testing" | "Documentation",
      "severity": "CRITICAL" | "HIGH" | "MEDIUM" | "LOW" | "ADVISORY",
      "title": "Short descriptive title",
      "description": "Clear explanation of the finding and its impact.",
      "suggestion": "Actionable, concrete fix with example code if applicable."
    }
  ]
}
"""


class AIReviewer:
    """Manages prompting the Gemini API and parsing structured code review responses."""

    def __init__(self, api_key: Optional[str], model: str = "gemini-2.0-flash", max_diff_bytes: int = 150000):
        self.api_key = api_key
        self.model = model
        self.max_diff_bytes = max_diff_bytes

    def is_available(self) -> bool:
        """Returns True if the API key is present."""
        return bool(self.api_key and self.api_key.strip())

    def review_diff(self, full_diff: str, file_diffs: list[FileDiff]) -> tuple[Optional[str], list[ReviewFinding]]:
        """
        Submits the PR diff to the Gemini API and returns (summary_override, list_of_findings).
        """
        if not self.is_available():
            return None, []

        prepared_diff = self._prepare_diff(full_diff, file_diffs)
        if not prepared_diff.strip():
            return "No code changes detected in this pull request to review.", []

        user_content = f"Please review this Pull Request diff:\n\n```diff\n{prepared_diff}\n```"

        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": SYSTEM_PROMPT + "\n\n" + user_content}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json",
            },
        }

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return self._parse_gemini_response(data)
        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            print(f"WARN: Gemini API returned HTTP {e.code}: {e.reason}. Response: {error_body[:200]}")
            return None, []
        except urllib.error.URLError as e:
            print(f"WARN: Network error calling Gemini API: {e.reason}")
            return None, []
        except Exception as e:
            print(f"WARN: Unexpected error in AI reviewer: {e}")
            return None, []

    def _prepare_diff(self, full_diff: str, file_diffs: list[FileDiff]) -> str:
        """Filters out binary and noise files, and enforces size limit."""
        meaningful_patches: list[str] = []
        ignored_extensions = {".lock", ".png", ".jpg", ".jpeg", ".svg", ".ico", ".pdf", ".map"}

        for fd in file_diffs:
            if fd.is_binary or any(fd.filename.endswith(ext) for ext in ignored_extensions):
                continue
            if fd.patch:
                meaningful_patches.append(f"--- a/{fd.filename}\n+++ b/{fd.filename}\n{fd.patch}")

        merged = "\n\n".join(meaningful_patches) if meaningful_patches else full_diff

        # Enforce max bytes
        if len(merged.encode("utf-8")) > self.max_diff_bytes:
            print(f"INFO: Truncating diff to {self.max_diff_bytes} bytes for AI review.")
            return merged[: self.max_diff_bytes] + "\n\n... [diff truncated for length] ..."

        return merged

    def _parse_gemini_response(self, response_data: dict) -> tuple[Optional[str], list[ReviewFinding]]:
        """Parses the Gemini JSON response payload into ReviewFinding instances."""
        findings: list[ReviewFinding] = []
        summary: Optional[str] = None

        try:
            candidates = response_data.get("candidates", [])
            if not candidates:
                return None, []

            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if not parts:
                return None, []

            raw_text = parts[0].get("text", "").strip()

            # Clean JSON markdown fences if model included them
            clean_text = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.MULTILINE)
            clean_text = re.sub(r"\s*```$", "", clean_text, flags=re.MULTILINE).strip()

            parsed = json.loads(clean_text)
            summary = parsed.get("summary")

            for item in parsed.get("findings", []):
                findings.append(
                    ReviewFinding(
                        file_path=item.get("file_path", "Unknown"),
                        line_number=item.get("line_number"),
                        category=item.get("category", "Maintainability"),
                        severity=item.get("severity", "MEDIUM"),
                        title=item.get("title", "Review Finding"),
                        description=item.get("description", ""),
                        suggestion=item.get("suggestion", ""),
                        source="ai",
                    )
                )

        except json.JSONDecodeError as e:
            print(f"WARN: Failed to parse Gemini response as JSON: {e}")
        except Exception as e:
            print(f"WARN: Error while processing Gemini response: {e}")

        return summary, findings
