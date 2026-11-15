"""Rule-based static code analyzer for security, bug risks, testing, and documentation."""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass
from typing import Optional

from bot2.diff_parser import FileDiff, ChangedLine


@dataclass
class ReviewFinding:
    """Represents a code review finding with actionable suggestion."""
    file_path: str
    line_number: Optional[int]
    category: str  # Security, Bug Risk, Maintainability, Testing, Documentation
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW, ADVISORY
    title: str
    description: str
    suggestion: str
    source: str = "static"  # static | ai

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "line_number": self.line_number,
            "category": self.category,
            "severity": self.severity,
            "title": self.title,
            "description": self.description,
            "suggestion": self.suggestion,
            "source": self.source,
        }


class StaticAnalyzer:
    """Performs AST parsing, regex pattern matching, and PR-level checks."""

    # Patterns for detecting hardcoded credentials & secrets
    SECRET_PATTERNS = [
        (
            re.compile(r"""(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{36}"""),
            "GitHub Personal Access Token Detected",
            "CRITICAL",
            "Security",
            "A personal GitHub token was found directly in source code.",
            "Revoke the token in GitHub Settings and inject it via GitHub Actions Secrets.",
        ),
        (
            re.compile(r"""AKIA[0-9A-Z]{16}"""),
            "AWS Access Key ID Detected",
            "CRITICAL",
            "Security",
            "An AWS access key ID pattern was detected.",
            "Use IAM Roles or GitHub Secrets to inject AWS credentials at runtime.",
        ),
        (
            re.compile(r"""-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----"""),
            "Private Key Exposed",
            "CRITICAL",
            "Security",
            "Private encryption or SSH key material was committed to the repository.",
            "Revoke this key immediately and manage keys via secure secrets storage.",
        ),
        (
            re.compile(r"""(?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|password|passwd|client[_-]?secret)\s*[:=]\s*['"][A-Za-z0-9_\-.~!@#$%^&*+=/]{8,}['"]""", re.IGNORECASE),
            "Hardcoded Credential / Secret Detected",
            "CRITICAL",
            "Security",
            "A sensitive token or credential appears to be hardcoded in the source code.",
            "Remove hardcoded secrets immediately. Store credentials in GitHub Actions Secrets or environment variables.",
        ),
    ]

    # Patterns for dangerous execution and injection patterns
    CODE_SMELL_PATTERNS = [
        (
            re.compile(r"""\b(?:eval|exec)\s*\("""),
            "Dynamic Code Execution (`eval`/`exec`)",
            "HIGH",
            "Security",
            "Using `eval()` or `exec()` executes arbitrary string input, which can lead to remote code execution (RCE).",
            "Avoid dynamic code execution. Use `ast.literal_eval()` for safe literals or refactor to dedicated parsers.",
        ),
        (
            re.compile(r"""\bsubprocess\.(?:call|check_call|check_output|Popen|run)\s*\([^)]*shell\s*=\s*True"""),
            "Subprocess Execution with `shell=True`",
            "HIGH",
            "Security",
            "Executing subprocesses with `shell=True` allows shell injection if any arguments include untrusted input.",
            "Pass command arguments as a list and set `shell=False` (e.g. `subprocess.run(['cmd', 'arg'])`).",
        ),
        (
            re.compile(r"""\bos\.system\s*\("""),
            "Unsafe Command Execution with `os.system`",
            "HIGH",
            "Security",
            "`os.system` runs commands through the system shell without proper escaping.",
            "Replace `os.system()` with `subprocess.run([...], check=True)` passing argument lists.",
        ),
        (
            re.compile(r"""(?i)\b(?:cursor|db|conn)\.execute\s*\(.*?%\s*s|f['"].*?\b(?:SELECT|INSERT|UPDATE|DELETE)\b.*?\{"""),
            "Potential SQL Injection Risk",
            "CRITICAL",
            "Security",
            "SQL query appears to be constructed using string formatting or interpolation instead of parameterized queries.",
            "Use parameterized SQL queries (e.g., `cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))`).",
        ),
        (
            re.compile(r"""\bpickle\.loads?\s*\("""),
            "Unsafe Object Deserialization (`pickle`)",
            "HIGH",
            "Security",
            "`pickle` is insecure against erroneous or maliciously constructed data.",
            "Use safer serialization formats such as JSON or Protocol Buffers.",
        ),
        (
            re.compile(r"""\bhashlib\.(?:md5|sha1)\s*\("""),
            "Weak Cryptographic Hash (`md5`/`sha1`)",
            "MEDIUM",
            "Security",
            "MD5 and SHA-1 have known collision vulnerabilities and should not be used for security purposes.",
            "Use `hashlib.sha256()` or `hashlib.sha3_256()` for hashing, or `bcrypt`/`argon2` for password hashing.",
        ),
        (
            re.compile(r"""^\s*except\s*:\s*(?:#.*)?$|^\s*except\s*:\s*(?:pass|\.\.\.)"""),
            "Bare `except:` Swallowing Exceptions",
            "MEDIUM",
            "Bug Risk",
            "A bare `except:` catches all exceptions (including SystemExit, KeyboardInterrupt) and silently ignores them.",
            "Catch specific exception classes (e.g., `except (ValueError, KeyError):`) and log or handle the error appropriately.",
        ),
        (
            re.compile(r"""\binnerHTML\s*=\s*[^;]+"""),
            "Raw `innerHTML` Assignment (XSS Risk)",
            "HIGH",
            "Security",
            "Assigning unsanitized input to `innerHTML` exposes the application to Cross-Site Scripting (XSS).",
            "Use `textContent` or sanitize data with a trusted sanitizer before assignment.",
        ),
    ]

    SOURCE_EXTENSIONS = {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".h", ".go", ".rs", ".rb", ".php"
    }

    TEST_PATTERNS = [
        re.compile(r"^tests?/"),
        re.compile(r"^__tests__/"),
        re.compile(r"test_[^/]+\.py$"),
        re.compile(r"[^/]+_test\.py$"),
        re.compile(r"[^/]+\.test\.[jt]sx?$"),
        re.compile(r"[^/]+\.spec\.[jt]sx?$"),
        re.compile(r"Test[^/]+\.java$"),
    ]

    DOC_PATTERNS = [
        re.compile(r"^README(\.[a-zA-Z]+)?$", re.IGNORECASE),
        re.compile(r"^docs?/", re.IGNORECASE),
        re.compile(r"\.md$", re.IGNORECASE),
    ]

    @classmethod
    def is_source_file(cls, filename: str) -> bool:
        ext = os.path.splitext(filename)[1].lower()
        return ext in cls.SOURCE_EXTENSIONS

    @classmethod
    def is_test_file(cls, filename: str) -> bool:
        norm = filename.replace("\\", "/")
        return any(p.search(norm) for p in cls.TEST_PATTERNS)

    @classmethod
    def is_doc_file(cls, filename: str) -> bool:
        norm = filename.replace("\\", "/")
        return any(p.search(norm) for p in cls.DOC_PATTERNS)

    def analyze_file_diff(self, file_diff: FileDiff) -> list[ReviewFinding]:
        """Analyzes a single file diff for syntax errors, secrets, and code smells on changed lines."""
        findings: list[ReviewFinding] = []

        if file_diff.is_deleted or file_diff.is_binary:
            return findings

        # 1. Python AST Syntax Check (if whole file or snippet can be parsed)
        if file_diff.filename.endswith(".py"):
            self._check_python_syntax(file_diff, findings)

        # 2. Check each newly changed line against security & code smell patterns
        for changed_line in file_diff.changed_lines:
            line_text = changed_line.content
            line_no = changed_line.line_number

            # Check Secret Patterns (take highest-priority match per line)
            matched_secret = False
            for pattern, title, severity, category, desc, suggestion in self.SECRET_PATTERNS:
                if pattern.search(line_text):
                    findings.append(
                        ReviewFinding(
                            file_path=file_diff.filename,
                            line_number=line_no,
                            category=category,
                            severity=severity,
                            title=title,
                            description=desc,
                            suggestion=suggestion,
                            source="static",
                        )
                    )
                    matched_secret = True
                    break  # Avoid double reporting same secret with generic rule

            # Check Code Smell Patterns
            for pattern, title, severity, category, desc, suggestion in self.CODE_SMELL_PATTERNS:
                if pattern.search(line_text):
                    findings.append(
                        ReviewFinding(
                            file_path=file_diff.filename,
                            line_number=line_no,
                            category=category,
                            severity=severity,
                            title=title,
                            description=desc,
                            suggestion=suggestion,
                            source="static",
                        )
                    )

        return findings

    def _check_python_syntax(self, file_diff: FileDiff, findings: list[ReviewFinding]) -> None:
        """Attempts to parse on-disk file if it exists, or warns if invalid syntax was introduced."""
        content = None
        if os.path.isfile(file_diff.filename):
            try:
                with open(file_diff.filename, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception:
                pass
        elif file_diff.status == "added" and file_diff.changed_lines:
            content = "\n".join(cl.content for cl in file_diff.changed_lines)

        if content is not None:
            try:
                tree = ast.parse(content, filename=file_diff.filename)
                self._check_python_ast(tree, file_diff, findings)
            except SyntaxError as e:
                findings.append(
                    ReviewFinding(
                        file_path=file_diff.filename,
                        line_number=e.lineno or 1,
                        category="Bug Risk",
                        severity="CRITICAL",
                        title="Python Syntax Error",
                        description=f"SyntaxError encountered while parsing file: {e.msg}",
                        suggestion="Fix the syntax error so the file can compile and run cleanly.",
                        source="static",
                    )
                )

    def _check_python_ast(self, tree: ast.AST, file_diff: FileDiff, findings: list[ReviewFinding]) -> None:
        """AST inspection for bug risks on modified lines."""
        changed_line_nums = {cl.line_number for cl in file_diff.changed_lines}

        for node in ast.walk(tree):
            lineno = getattr(node, "lineno", None)
            if lineno and changed_line_nums and lineno not in changed_line_nums:
                continue

            # Check for mutable default arguments in functions
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for default in node.args.defaults + node.args.kw_defaults:
                    if default is not None and isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        findings.append(
                            ReviewFinding(
                                file_path=file_diff.filename,
                                line_number=lineno,
                                category="Bug Risk",
                                severity="MEDIUM",
                                title=f"Mutable Default Argument in `{node.name}()`",
                                description=(
                                    "Default argument values are evaluated once when the function is defined. "
                                    "Mutable default arguments (lists, dicts, sets) persist state across calls, causing subtle bugs."
                                ),
                                suggestion="Use `None` as the default value and initialize the mutable object inside the function body.",
                                source="static",
                            )
                        )

    def check_pr_hygiene(self, file_diffs: list[FileDiff]) -> list[ReviewFinding]:
        """
        Inspects the overall pull request changes to determine if corresponding
        tests or documentation were omitted.
        """
        findings: list[ReviewFinding] = []

        source_files_changed = [
            f.filename for f in file_diffs
            if self.is_source_file(f.filename) and not self.is_test_file(f.filename) and not f.is_deleted
        ]
        test_files_changed = [
            f.filename for f in file_diffs
            if self.is_test_file(f.filename)
        ]
        doc_files_changed = [
            f.filename for f in file_diffs
            if self.is_doc_file(f.filename)
        ]

        # 1. Missing test check: if code files changed with non-trivial additions, but zero tests changed
        total_code_additions = sum(
            f.additions for f in file_diffs
            if self.is_source_file(f.filename) and not self.is_test_file(f.filename)
        )

        if source_files_changed and not test_files_changed and total_code_additions >= 10:
            findings.append(
                ReviewFinding(
                    file_path=source_files_changed[0],
                    line_number=None,
                    category="Testing",
                    severity="ADVISORY",
                    title="Missing Test Coverage",
                    description=(
                        f"This PR modifies/adds {len(source_files_changed)} source file(s) "
                        f"({', '.join(source_files_changed[:3])}{'...' if len(source_files_changed) > 3 else ''}), "
                        f"but no unit or integration tests were added or updated."
                    ),
                    suggestion=(
                        "Add corresponding test cases in the test suite to verify the new behavior "
                        "and guard against regressions."
                    ),
                    source="static",
                )
            )

        # 2. Missing documentation check: substantial PR with zero documentation updates
        if total_code_additions >= 80 and not doc_files_changed:
            findings.append(
                ReviewFinding(
                    file_path=source_files_changed[0] if source_files_changed else "README.md",
                    line_number=None,
                    category="Documentation",
                    severity="ADVISORY",
                    title="Missing Documentation Update",
                    description=(
                        f"Significant functionality was introduced ({total_code_additions} additions), "
                        "but no README or documentation files were updated."
                    ),
                    suggestion=(
                        "Update `README.md` or relevant architecture documentation to describe new features, "
                        "configuration flags, or API changes."
                    ),
                    source="static",
                )
            )

        return findings

    def run_all(self, file_diffs: list[FileDiff]) -> list[ReviewFinding]:
        """Runs file-level analysis and PR-level hygiene checks."""
        all_findings: list[ReviewFinding] = []

        for fd in file_diffs:
            all_findings.extend(self.analyze_file_diff(fd))

        all_findings.extend(self.check_pr_hygiene(file_diffs))
        return all_findings
