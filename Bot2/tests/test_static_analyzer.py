"""Unit tests for the Static Analyzer."""

import unittest
from bot2.diff_parser import ChangedLine, FileDiff
from bot2.static_analyzer import StaticAnalyzer


class TestStaticAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = StaticAnalyzer()

    def test_detect_github_token(self):
        line = ChangedLine("config.py", 10, 'TOKEN = "ghp_1234567890abcdef1234567890abcdef1234"')
        fd = FileDiff(filename="config.py", changed_lines=[line])
        findings = self.analyzer.analyze_file_diff(fd)

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].category, "Security")
        self.assertEqual(findings[0].severity, "CRITICAL")
        self.assertIn("GitHub Personal Access Token", findings[0].title)
        self.assertEqual(findings[0].line_number, 10)

    def test_detect_eval_and_os_system(self):
        lines = [
            ChangedLine("utils.py", 14, "res = eval(user_input)"),
            ChangedLine("utils.py", 22, "os.system('ls ' + folder)"),
        ]
        fd = FileDiff(filename="utils.py", changed_lines=lines)
        findings = self.analyzer.analyze_file_diff(fd)

        titles = [f.title for f in findings]
        self.assertTrue(any("Dynamic Code Execution" in t for t in titles))
        self.assertTrue(any("Unsafe Command Execution" in t for t in titles))

    def test_detect_sql_injection(self):
        line = ChangedLine("db.py", 30, "sql = f\"SELECT * FROM users WHERE id = {user_id}\"")
        fd = FileDiff(filename="db.py", changed_lines=[line])
        findings = self.analyzer.analyze_file_diff(fd)

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].category, "Security")
        self.assertEqual(findings[0].severity, "CRITICAL")
        self.assertIn("SQL Injection", findings[0].title)

    def test_detect_bare_except(self):
        line = ChangedLine("worker.py", 45, "except:")
        fd = FileDiff(filename="worker.py", changed_lines=[line])
        findings = self.analyzer.analyze_file_diff(fd)

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].category, "Bug Risk")
        self.assertIn("Bare `except:`", findings[0].title)

    def test_clean_lines_no_findings(self):
        lines = [
            ChangedLine("math_ops.py", 1, "def add(a: int, b: int) -> int:"),
            ChangedLine("math_ops.py", 2, "    return a + b"),
        ]
        fd = FileDiff(filename="math_ops.py", changed_lines=lines)
        findings = self.analyzer.analyze_file_diff(fd)
        self.assertEqual(len(findings), 0)

    def test_missing_tests_check(self):
        # 20 lines added in a source file, 0 test files touched
        fd = FileDiff(
            filename="src/feature.py",
            additions=25,
            changed_lines=[ChangedLine("src/feature.py", i, f"code_{i}") for i in range(25)],
        )
        hygiene = self.analyzer.check_pr_hygiene([fd])
        self.assertEqual(len(hygiene), 1)
        self.assertEqual(hygiene[0].category, "Testing")
        self.assertIn("Missing Test Coverage", hygiene[0].title)

    def test_tests_present_satisfies_hygiene(self):
        src_fd = FileDiff(filename="src/feature.py", additions=25)
        test_fd = FileDiff(filename="tests/test_feature.py", additions=15)
        hygiene = self.analyzer.check_pr_hygiene([src_fd, test_fd])
        self.assertEqual(len(hygiene), 0)


if __name__ == "__main__":
    unittest.main()
