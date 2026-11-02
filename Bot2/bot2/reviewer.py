"""Main reviewer coordinator: combines static and AI analysis, deduplicates, and formats."""

from __future__ import annotations

import os
import sys
from typing import Optional

from bot2.ai_reviewer import AIReviewer
from bot2.config import BotConfig
from bot2.diff_parser import FileDiff
from bot2.formatter import ReviewFormatter
from bot2.github_client import GitHubClient
from bot2.static_analyzer import ReviewFinding, StaticAnalyzer


class ReviewOrchestrator:
    """Coordinates the review process across static analysis and AI model."""

    def __init__(self, config: BotConfig):
        self.config = config
        self.static_analyzer = StaticAnalyzer()
        self.ai_reviewer = AIReviewer(
            api_key=config.gemini_api_key,
            model=config.gemini_model,
            max_diff_bytes=config.max_diff_bytes,
        )
        self.github_client = GitHubClient(
            token=config.github_token,
            repository=config.github_repository,
        )

    def merge_findings(
        self, static_findings: list[ReviewFinding], ai_findings: list[ReviewFinding]
    ) -> list[ReviewFinding]:
        """
        Merges static and AI findings, prioritizing higher severity and eliminating duplicates.
        """
        merged: list[ReviewFinding] = list(static_findings)
        seen_keys = {
            (f.file_path, f.line_number, f.category.lower())
            for f in static_findings
            if f.line_number is not None
        }

        for finding in ai_findings:
            key = (finding.file_path, finding.line_number, finding.category.lower())
            if finding.line_number is not None and key in seen_keys:
                # Issue already flagged at same line and category by static analyzer
                continue
            merged.append(finding)

        # Sort: CRITICAL -> HIGH -> MEDIUM -> LOW -> ADVISORY
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "ADVISORY": 4}
        merged.sort(key=lambda x: severity_order.get(x.severity.upper(), 5))
        return merged

    def run_review(
        self,
        file_diffs: list[FileDiff],
        raw_diff: str,
        secret_notice: Optional[str] = None,
    ) -> tuple[str, list[ReviewFinding]]:
        """Executes static + AI analysis, formats comment, and returns (markdown, findings)."""
        # 1. Run Static Analysis
        static_findings = self.static_analyzer.run_all(file_diffs)

        # 2. Run AI Analysis if API key is configured
        ai_summary = None
        ai_findings: list[ReviewFinding] = []

        if self.ai_reviewer.is_available():
            print("INFO: Executing AI semantic review with Gemini...")
            ai_summary, ai_findings = self.ai_reviewer.review_diff(raw_diff, file_diffs)
            print(f"INFO: AI review completed with {len(ai_findings)} findings.")
        else:
            print("INFO: AI review skipped (no API key configured). Running in static mode.")

        # 3. Merge & Deduplicate
        all_findings = self.merge_findings(static_findings, ai_findings)

        # 4. Format Markdown Review
        markdown_review = ReviewFormatter.format_review(
            findings=all_findings,
            file_diffs=file_diffs,
            commit_sha=self.config.github_sha,
            ai_enabled=self.ai_reviewer.is_available(),
            ai_summary=ai_summary,
            secret_notice=secret_notice,
        )

        return markdown_review, all_findings

    def publish_review(self, pr_number: int, markdown_review: str) -> bool:
        """Publishes or updates the review comment on GitHub PR and writes step summary."""
        # Write to GitHub Actions step summary if available
        step_summary_path = os.getenv("GITHUB_STEP_SUMMARY")
        if step_summary_path:
            try:
                with open(step_summary_path, "a", encoding="utf-8") as f:
                    f.write(markdown_review + "\n")
            except Exception as e:
                print(f"WARN: Could not write to GITHUB_STEP_SUMMARY: {e}")

        # Post / Update PR comment on GitHub
        return self.github_client.post_or_update_review_comment(
            pr_number=pr_number,
            comment_body=markdown_review,
            marker=self.config.comment_marker,
        )
