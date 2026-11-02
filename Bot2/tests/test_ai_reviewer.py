"""Unit tests for the AI Reviewer module."""

import json
import unittest
from bot2.ai_reviewer import AIReviewer
from bot2.diff_parser import FileDiff


class TestAIReviewer(unittest.TestCase):
    def test_availability(self):
        reviewer_none = AIReviewer(api_key=None)
        self.assertFalse(reviewer_none.is_available())

        reviewer_empty = AIReviewer(api_key="   ")
        self.assertFalse(reviewer_empty.is_available())

        reviewer_active = AIReviewer(api_key="mock-key-12345")
        self.assertTrue(reviewer_active.is_available())

    def test_parse_gemini_json_response(self):
        reviewer = AIReviewer(api_key="mock")
        mock_response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps({
                                    "status": "Issues found",
                                    "summary": "Detected a logic error in loop bounds.",
                                    "findings": [
                                        {
                                            "file_path": "src/parser.py",
                                            "line_number": 88,
                                            "category": "Bug Risk",
                                            "severity": "HIGH",
                                            "title": "Index Out of Bounds",
                                            "description": "Loop index exceeds array length.",
                                            "suggestion": "Use len(arr) - 1."
                                        }
                                    ]
                                })
                            }
                        ]
                    }
                }
            ]
        }

        summary, findings = reviewer._parse_gemini_response(mock_response)
        self.assertEqual(summary, "Detected a logic error in loop bounds.")
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].file_path, "src/parser.py")
        self.assertEqual(findings[0].line_number, 88)
        self.assertEqual(findings[0].severity, "HIGH")
        self.assertEqual(findings[0].source, "ai")

    def test_parse_gemini_response_with_markdown_fences(self):
        reviewer = AIReviewer(api_key="mock")
        raw_json_str = json.dumps({
            "status": "No significant issues found",
            "summary": "Clean code.",
            "findings": []
        })
        fenced_text = f"```json\n{raw_json_str}\n```"
        mock_response = {
            "candidates": [
                {"content": {"parts": [{"text": fenced_text}]}}
            ]
        }

        summary, findings = reviewer._parse_gemini_response(mock_response)
        self.assertEqual(summary, "Clean code.")
        self.assertEqual(len(findings), 0)

    def test_prepare_diff_truncation(self):
        reviewer = AIReviewer(api_key="mock", max_diff_bytes=50)
        fd = FileDiff(filename="big.py", patch="+" + ("a" * 100))
        truncated = reviewer._prepare_diff("raw", [fd])
        self.assertIn("... [diff truncated for length] ...", truncated)


if __name__ == "__main__":
    unittest.main()
