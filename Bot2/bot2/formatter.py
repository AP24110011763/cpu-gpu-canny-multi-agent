"""Markdown formatting engine for Bot 2 pull request review comments."""

from __future__ import annotations

import datetime
from typing import Optional

from bot2.diff_parser import FileDiff
from bot2.static_analyzer import ReviewFinding


SEVERITY_BADGES = {
    "CRITICAL": "🔴 `CRITICAL`",
    "HIGH": "🟠 `HIGH`",
    "MEDIUM": "🟡 `MEDIUM`",
    "LOW": "🔵 `LOW`",
    "ADVISORY": "⚪ `ADVISORY`",
}


class ReviewFormatter:
    """Formats review findings into a GitHub Markdown comment."""

    COMMENT_MARKER = "<!-- bot-2-code-review -->"

    @classmethod
    def format_review(
        cls,
        findings: list[ReviewFinding],
        file_diffs: list[FileDiff],
        commit_sha: str,
        ai_enabled: bool,
        ai_summary: Optional[str] = None,
        secret_notice: Optional[str] = None,
    ) -> str:
        """Constructs the complete GitHub markdown review comment."""
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        short_sha = commit_sha[:7] if commit_sha else "HEAD"

        # Separate hygiene findings (tests & docs) from code-level findings
        code_findings = [f for f in findings if f.category not in ("Testing", "Documentation")]
        hygiene_findings = [f for f in findings if f.category in ("Testing", "Documentation")]

        # Determine overall status
        has_critical_or_high = any(f.severity in ("CRITICAL", "HIGH") for f in findings)
        has_issues = len(code_findings) > 0 or has_critical_or_high

        if has_critical_or_high:
            status_badge = "🔴 **Issues found**"
            status_text = "Issues found (Action required)"
        elif has_issues:
            status_badge = "🟡 **Issues found**"
            status_text = "Issues found (Improvements recommended)"
        elif hygiene_findings:
            status_badge = "🟡 **Advisories noted**"
            status_text = "Advisories noted (Tests/Docs recommended)"
        else:
            status_badge = "🟢 **No significant issues found**"
            status_text = "No significant issues found"

        mode_desc = "Hybrid (Static + AI)" if ai_enabled else "Static Rules Only"

        # Markdown Header and Metadata Table
        lines = [
            cls.COMMENT_MARKER,
            "## 🤖 Bot 2: Autonomous Code Review & Quality Report",
            "",
            "| Review Status | Files Changed | Code Findings | Advisories | Engine Mode |",
            "| :--- | :---: | :---: | :---: | :---: |",
            f"| {status_badge} | {len(file_diffs)} | {len(code_findings)} | {len(hygiene_findings)} | {mode_desc} |",
            "",
        ]

        # Missing Secret Warning Callout (ensures zero silent failures)
        if secret_notice:
            lines.extend([
                "> [!WARNING]",
                f"> **Configuration Notice:** {secret_notice}",
                "",
            ])

        # Executive Summary Section
        lines.append("### 📋 Executive Summary")
        if ai_summary and ai_summary.strip():
            lines.append(ai_summary.strip())
        elif not findings:
            lines.append(
                "All changed files were inspected for potential bugs, logic errors, security vulnerabilities, "
                "and testing hygiene. **No significant issues were found.** Changes look clean and safe to merge."
            )
        elif has_critical_or_high:
            lines.append(
                f"Review completed with status **{status_text}**. Critical or high-severity problems "
                f"were detected in the pull request changes. Please review the actionable suggestions below."
            )
        else:
            lines.append(
                f"Review completed with status **{status_text}**. Several maintainability or hygiene improvements "
                f"were identified to help improve code quality before merging."
            )
        lines.append("")

        # Detailed Findings Section
        if code_findings:
            lines.append("### 🔍 Detailed Findings")
            for idx, finding in enumerate(code_findings, 1):
                badge = SEVERITY_BADGES.get(finding.severity.upper(), f"`{finding.severity}`")
                loc_str = finding.file_path
                if finding.line_number is not None:
                    loc_str += f" (Line {finding.line_number})"

                lines.extend([
                    f"#### {idx}. {badge} {finding.title}",
                    f"- **Location:** `{loc_str}`",
                    f"- **Category:** {finding.category} | **Detector:** `{finding.source.upper()}`",
                    f"- **Problem:** {finding.description}",
                    f"- **Actionable Suggestion:** {finding.suggestion}",
                    "",
                ])
        else:
            lines.extend([
                "### 🔍 Detailed Findings",
                "✅ **No code defects, security smells, or logic errors were identified in the changed lines.**",
                "",
            ])

        # Testing & Documentation Section
        lines.append("### 🧪 Testing & Documentation Hygiene")
        if hygiene_findings:
            for h in hygiene_findings:
                lines.extend([
                    f"- ⚠️ **{h.title}:** {h.description}",
                    f"  - *Action:* {h.suggestion}",
                ])
            lines.append("")
        else:
            lines.extend([
                "- ✅ Test coverage and documentation requirements appear satisfied for this change.",
                "",
            ])

        # Footer with Commit SHA & Deduplication Information
        lines.extend([
            "---",
            f"> ℹ️ *Bot 2 reviewed commit [`{short_sha}`] at {timestamp}. "
            "To prevent duplicate comments, this review updates automatically when new commits are pushed.*",
        ])

        return "\n".join(lines)
