"""Unified diff and patch parser for extracting changed files and line numbers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChangedLine:
    """Represents a single added/modified line in a file."""
    file_path: str
    line_number: int
    content: str


@dataclass
class FileDiff:
    """Represents the diff changes for a single file."""
    filename: str
    status: str = "modified"  # added, modified, deleted, renamed
    old_filename: Optional[str] = None
    additions: int = 0
    deletions: int = 0
    patch: str = ""
    changed_lines: list[ChangedLine] = field(default_factory=list)

    @property
    def is_deleted(self) -> bool:
        return self.status == "deleted"

    @property
    def is_binary(self) -> bool:
        return "Binary files differ" in self.patch or "GIT binary patch" in self.patch


class DiffParser:
    """Parses standard unified diffs or GitHub patch hunks."""

    HUNK_HEADER_RE = re.compile(r"^@@\s+-\d+(?:,\d+)?\s+\+(\d+)(?:,(\d+))?\s+@@")

    @classmethod
    def parse_patch_hunks(cls, filename: str, patch_text: str) -> list[ChangedLine]:
        """
        Parses a git diff patch hunk string (such as from GitHub API 'patch' field)
        and extracts the added/modified lines with their corresponding line numbers in the new file.
        """
        changed_lines: list[ChangedLine] = []
        if not patch_text:
            return changed_lines

        current_line_num = 0

        for raw_line in patch_text.splitlines():
            # Check for hunk header @@ -a,b +c,d @@
            hunk_match = cls.HUNK_HEADER_RE.match(raw_line)
            if hunk_match:
                current_line_num = int(hunk_match.group(1))
                continue

            if raw_line.startswith("+") and not raw_line.startswith("+++"):
                # Newly added or modified line in new file
                content = raw_line[1:]
                changed_lines.append(
                    ChangedLine(
                        file_path=filename,
                        line_number=current_line_num,
                        content=content,
                    )
                )
                current_line_num += 1
            elif raw_line.startswith("-") and not raw_line.startswith("---"):
                # Removed line - does not advance new file line counter
                continue
            elif raw_line.startswith(" "):
                # Unmodified context line - advances new file line counter
                current_line_num += 1
            elif raw_line.startswith(r"\ "):
                # e.g. "\ No newline at end of file"
                continue

        return changed_lines

    @classmethod
    def parse_full_unified_diff(cls, diff_text: str) -> list[FileDiff]:
        """
        Parses a full multi-file unified git diff (e.g., git diff output or .diff file).
        """
        file_diffs: list[FileDiff] = []
        if not diff_text or not diff_text.strip():
            return file_diffs

        # Split diff by file header: "diff --git"
        chunks = re.split(r"(?=^diff --git )", diff_text, flags=re.MULTILINE)

        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk or not chunk.startswith("diff --git"):
                continue

            lines = chunk.splitlines()
            header_line = lines[0]

            # Header format: diff --git a/path/to/file b/path/to/file
            header_match = re.search(r"diff --git a/(.*?) b/(.*)", header_line)
            if not header_match:
                continue

            old_file = header_match.group(1)
            new_file = header_match.group(2)

            status = "modified"
            if "new file mode" in chunk:
                status = "added"
            elif "deleted file mode" in chunk:
                status = "deleted"
            elif "rename from" in chunk:
                status = "renamed"

            # Locate where patch hunks start (first line starting with '@@')
            patch_start_idx = -1
            for idx, line in enumerate(lines):
                if line.startswith("@@"):
                    patch_start_idx = idx
                    break

            patch_text = "\n".join(lines[patch_start_idx:]) if patch_start_idx != -1 else ""

            changed_lines: list[ChangedLine] = []
            if status != "deleted":
                changed_lines = cls.parse_patch_hunks(new_file, patch_text)

            additions = sum(1 for line in lines if line.startswith("+") and not line.startswith("+++"))
            deletions = sum(1 for line in lines if line.startswith("-") and not line.startswith("---"))

            file_diffs.append(
                FileDiff(
                    filename=new_file,
                    status=status,
                    old_filename=old_file if old_file != new_file else None,
                    additions=additions,
                    deletions=deletions,
                    patch=patch_text,
                    changed_lines=changed_lines,
                )
            )

        return file_diffs
