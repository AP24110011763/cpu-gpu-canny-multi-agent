"""Unit tests for GitHubClient comment deduplication logic."""

import json
import unittest
from unittest.mock import MagicMock, patch
from bot2.github_client import GitHubClient


class TestGitHubClient(unittest.TestCase):
    def setUp(self):
        self.client = GitHubClient(token="mock-token", repository="test-owner/test-repo")

    @patch.object(GitHubClient, "_request")
    def test_find_comment_with_marker_found(self, mock_request):
        mock_comments = [
            {"id": 101, "body": "Standard user review"},
            {"id": 202, "body": "<!-- bot-2-code-review -->\n## Bot 2 Review"},
        ]
        mock_request.return_value = (200, json.dumps(mock_comments).encode("utf-8"), {})

        comment_id = self.client._find_comment_with_marker(42, "<!-- bot-2-code-review -->")
        self.assertEqual(comment_id, 202)

    @patch.object(GitHubClient, "_request")
    def test_find_comment_with_marker_not_found(self, mock_request):
        mock_comments = [
            {"id": 101, "body": "Just an author comment."},
        ]
        mock_request.return_value = (200, json.dumps(mock_comments).encode("utf-8"), {})

        comment_id = self.client._find_comment_with_marker(42, "<!-- bot-2-code-review -->")
        self.assertIsNone(comment_id)

    @patch.object(GitHubClient, "_find_comment_with_marker")
    @patch.object(GitHubClient, "_request")
    def test_post_or_update_updates_when_existing(self, mock_request, mock_find):
        mock_find.return_value = 555
        mock_request.return_value = (200, b"{}", {})

        success = self.client.post_or_update_review_comment(42, "New body", "<!-- bot-2-code-review -->")
        self.assertTrue(success)

        # Ensure PATCH was invoked on the specific comment ID endpoint
        endpoint_called = mock_request.call_args[0][0]
        method_called = mock_request.call_args[1].get("method")
        self.assertIn("/repos/test-owner/test-repo/issues/comments/555", endpoint_called)
        self.assertEqual(method_called, "PATCH")

    @patch.object(GitHubClient, "_find_comment_with_marker")
    @patch.object(GitHubClient, "_request")
    def test_post_or_update_creates_new_when_not_found(self, mock_request, mock_find):
        mock_find.return_value = None
        mock_request.return_value = (201, b"{}", {})

        success = self.client.post_or_update_review_comment(42, "New body", "<!-- bot-2-code-review -->")
        self.assertTrue(success)

        # Ensure POST was invoked on PR comments endpoint
        endpoint_called = mock_request.call_args[0][0]
        method_called = mock_request.call_args[1].get("method")
        self.assertIn("/repos/test-owner/test-repo/issues/42/comments", endpoint_called)
        self.assertEqual(method_called, "POST")


if __name__ == "__main__":
    unittest.main()
