"""Unit tests for the Review Orchestrator."""

import unittest
from bot2.config import BotConfig
from bot2.diff_parser import FileDiff
from bot2.reviewer import ReviewOrchestrator
from bot2.static_analyzer import ReviewFinding


class TestReviewOrchestrator(unittest.TestCase):
    def setUp(self):
        self.config = BotConfig(
            github_token="",
            github_repository="owner/repo",
            github_event_path="",
            github_sha="testsha123",
            pr_number=1,
            gemini_api_key=None,
            gemini_model="gemini-2.0-flash",
            require_ai_key=False,
            max_diff_bytes=100000,
        )
        self.orchestrator = ReviewOrchestrator(self.config)

    def test_merge_findings_deduplication(self):
        static_f = ReviewFinding(
            file_path="app.py",
            line_number=20,
            category="Security",
            severity="CRITICAL",
            title="Static Finding",
            description="Static desc",
            suggestion="Static sug",
            source="static",
        )
        ai_duplicate = ReviewFinding(
            file_path="app.py",
            line_number=20,
            category="Security",
            severity="HIGH",
            title="AI Duplicate Finding",
            description="AI desc",
            suggestion="AI sug",
            source="ai",
        )
        ai_new = ReviewFinding(
            file_path="app.py",
            line_number=45,
            category="Bug Risk",
            severity="HIGH",
            title="AI New Finding",
            description="AI desc",
            suggestion="AI sug",
            source="ai",
        )

        merged = self.orchestrator.merge_findings([static_f], [ai_duplicate, ai_new])
        self.assertEqual(len(merged), 2)
        # Verify deduplication kept the static finding on line 20
        self.assertEqual(merged[0].line_number, 20)
        self.assertEqual(merged[0].source, "static")
        # Verify new AI finding was included
        self.assertEqual(merged[1].line_number, 45)
        self.assertEqual(merged[1].source, "ai")

    def test_merge_findings_severity_sorting(self):
        f_medium = ReviewFinding("a.py", 1, "Maintainability", "MEDIUM", "T1", "D1", "S1")
        f_critical = ReviewFinding("b.py", 2, "Security", "CRITICAL", "T2", "D2", "S2")
        f_high = ReviewFinding("c.py", 3, "Bug Risk", "HIGH", "T3", "D3", "S3")

        merged = self.orchestrator.merge_findings([f_medium, f_critical], [f_high])
        severities = [f.severity for f in merged]
        self.assertEqual(severities, ["CRITICAL", "HIGH", "MEDIUM"])


if __name__ == "__main__":
    unittest.main()
