"""Unit tests for Review Formatter."""

import unittest
from bot2.diff_parser import FileDiff
from bot2.formatter import ReviewFormatter
from bot2.static_analyzer import ReviewFinding


class TestReviewFormatter(unittest.TestCase):
    def test_format_clean_review(self):
        file_diffs = [FileDiff(filename="app.py", additions=2, deletions=1)]
        markdown = ReviewFormatter.format_review(
            findings=[],
            file_diffs=file_diffs,
            commit_sha="abcdef123456",
            ai_enabled=True,
            ai_summary="Clean changes, all good.",
        )

        self.assertIn("<!-- bot-2-code-review -->", markdown)
        self.assertIn("🟢 **No significant issues found**", markdown)
        self.assertIn("Hybrid (Static + AI)", markdown)
        self.assertIn("`abcdef1`", markdown)
        self.assertIn("Clean changes, all good.", markdown)

    def test_format_critical_issues_review(self):
        finding = ReviewFinding(
            file_path="src/login.py",
            line_number=33,
            category="Security",
            severity="CRITICAL",
            title="SQL Injection Detected",
            description="User input directly concatenated into query.",
            suggestion="Use parameter binding.",
            source="static",
        )
        file_diffs = [FileDiff(filename="src/login.py")]
        markdown = ReviewFormatter.format_review(
            findings=[finding],
            file_diffs=file_diffs,
            commit_sha="998877665544",
            ai_enabled=False,
            secret_notice="GEMINI_API_KEY is not set.",
        )

        self.assertIn("<!-- bot-2-code-review -->", markdown)
        self.assertIn("🔴 **Issues found**", markdown)
        self.assertIn("Static Rules Only", markdown)
        self.assertIn("Configuration Notice:", markdown)
        self.assertIn("SQL Injection Detected", markdown)
        self.assertIn("src/login.py (Line 33)", markdown)
        self.assertIn("Use parameter binding.", markdown)


if __name__ == "__main__":
    unittest.main()
