"""Bot 2 as a teammate: the code reviewer. Same commands from GitHub comments and from the terminal."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from . import messages
from .bot1_agent import Ctx
from .commands import split_command
from .paths import ensure_paths

HELP = """**What I do:** I review code. I check every pull request, and I review files Bot 1 files for you.

| Say this | I will |
|---|---|
| `/bot2 review PR 3` | review pull request #3 and post the review on it |
| `/bot2 review PATH` | review one file in the repo (terminal: a local file) |
| `/bot2 explain N [PATH]` | explain finding N from the last review of that PR or file |
| `/bot2 status` | say how I'm set up (AI review on or off) |

A file is **approved** when it has no CRITICAL or HIGH findings. Talk to Bot 1 with `/bot1 help`."""

ICON = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵", "ADVISORY": "⚪"}
ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "ADVISORY"]


def whole_file_diff(path: str, text: str):
    """Treat a whole file as 'added', so the PR analyzer can review it."""
    ensure_paths()
    from bot2.diff_parser import ChangedLine, FileDiff
    lines = text.splitlines()
    patch = f"@@ -0,0 +1,{len(lines)} @@\n" + "\n".join("+" + ln for ln in lines)
    return FileDiff(filename=path, status="added", additions=len(lines), patch=patch,
                    changed_lines=[ChangedLine(path, i, ln) for i, ln in enumerate(lines, 1)]), patch


def review_text(path: str, text: str) -> Tuple[str, list]:
    """Returns (verdict, findings). Static analysis always; Gemini too when GEMINI_API_KEY is set."""
    ensure_paths()
    from bot2.ai_reviewer import AIReviewer
    from bot2.static_analyzer import StaticAnalyzer
    fd, patch = whole_file_diff(path, text)
    findings = list(StaticAnalyzer().run_all([fd]))
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        ai = AIReviewer(api_key=key, model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"), max_diff_bytes=150000)
        _summary, ai_findings = ai.review_diff(f"--- /dev/null\n+++ b/{path}\n{patch}", [fd])
        findings += ai_findings
    findings.sort(key=lambda f: ORDER.index(f.severity.upper()) if f.severity.upper() in ORDER else 9)
    return verdict_of(findings), findings


def verdict_of(findings: list) -> str:
    return "needs-fix" if any(f.severity.upper() in ("CRITICAL", "HIGH") for f in findings) else "approved"


def counts(findings: list) -> str:
    n = {s: sum(1 for f in findings if f.severity.upper() == s) for s in ORDER}
    return ", ".join(f"{n[s]} {s.lower()}" for s in ORDER if n[s]) or "no findings"


def summarize(findings: list, limit: int = 6) -> str:
    lines = []
    for i, f in enumerate(findings[:limit], 1):
        where = f"{f.file_path}:{f.line_number}" if f.line_number else f.file_path
        lines.append(f"{i}. {ICON.get(f.severity.upper(), '•')} **{f.severity}** `{where}`: {f.title}")
    if len(findings) > limit:
        lines.append(f"…and {len(findings) - limit} more.")
    return "\n".join(lines)


def explain(findings: list, n: int) -> str:
    if not 1 <= n <= len(findings):
        return f"There are {len(findings)} finding(s); pick a number from 1 to {len(findings)}."
    f = findings[n - 1]
    where = f"{f.file_path}:{f.line_number}" if f.line_number else f.file_path
    return (f"**Finding {n}: {f.title}** ({f.severity}, {f.category}) at `{where}`\n\n"
            f"**Why it matters:** {f.description}\n\n**How to fix it:** {f.suggestion}")


class Bot2Agent:
    name = "bot2"

    def __init__(self, gh=None, read_file: Optional[Callable[[str], Optional[str]]] = None,
                 pr_reviewer: Optional[Callable[[int], Tuple[str, list]]] = None):
        self.gh = gh
        self.read_file = read_file or self._default_read
        self.pr_reviewer = pr_reviewer or self._review_pr

    def handle(self, text: str, ctx: Ctx) -> str:
        cmd, args = split_command(text)
        if cmd in ("help", "hi", "hello", "?"):
            return HELP
        if cmd == "status":
            ai = bool(os.environ.get("GEMINI_API_KEY", "").strip())
            return ("Static analysis: on. AI review (Gemini): " + ("on." if ai else "off (add the `GEMINI_API_KEY` secret to turn it on)."))
        if cmd == "review":
            return self._review(args, ctx)
        if cmd == "explain":
            return self._explain(args, ctx)
        return f"I don't know `{cmd}`. Try `/bot2 help`."

    # --- review ---
    def _review(self, args: str, ctx: Ctx) -> str:
        m = re.fullmatch(r"(?:pr|pull request|#)?\s*#?(\d+)", args.strip(), re.I)
        if not args.strip() and ctx.pr_number:
            m = re.fullmatch(r"(\d+)", str(ctx.pr_number))
        if m:
            n = int(m.group(1))
            if self.gh is None and self.pr_reviewer is self._review_pr:
                return "I need a GitHub token to review a PR. Set `GITHUB_TOKEN` and `GITHUB_REPOSITORY`."
            verdict, findings = self.pr_reviewer(n)
            return f"Reviewed PR #{n} ({counts(findings)}). The full review is on the PR.\n\n{summarize(findings)}"
        path = args.strip()
        if not path:
            return "Usage: `/bot2 review PR 3` or `/bot2 review src/file.py`."
        text = self.read_file(path)
        if text is None:
            return f"I couldn't read `{path}` (missing, binary, or not on the default branch)."
        verdict, findings = review_text(path, text)
        head = "✅ **Approved**" if verdict == "approved" else "⚠️ **Needs fixes**"
        body = f"{head}: `{path}` has {counts(findings)}."
        return body + ("\n\n" + summarize(findings) + "\n\nAsk `/bot2 explain N " + path + "` for details." if findings else "")

    def _explain(self, args: str, ctx: Ctx) -> str:
        parts = args.split(None, 1)
        if not parts or not parts[0].isdigit():
            return "Usage: `/bot2 explain N [PATH]`, for example `/bot2 explain 1 src/auth.py`."
        n, path = int(parts[0]), (parts[1].strip() if len(parts) > 1 else "")
        if path:
            text = self.read_file(path)
            if text is None:
                return f"I couldn't read `{path}`."
            _v, findings = review_text(path, text)
        elif ctx.pr_number:
            _v, findings = self.pr_reviewer(ctx.pr_number)
        else:
            return "Tell me which file: `/bot2 explain N PATH` (in a PR thread I use that PR)."
        return explain(findings, n)

    def _default_read(self, path: str) -> Optional[str]:
        from bot1.safety import validate_path
        from bot1.errors import BotError
        try:
            path = validate_path(path)
        except BotError:
            return None
        local = Path(path)
        if self.gh is None or local.is_file():
            try:
                return local.read_text(encoding="utf-8") if local.is_file() else None
            except (UnicodeDecodeError, OSError):
                return None
        return self.gh.get_file(path)

    def _review_pr(self, number: int) -> Tuple[str, list]:
        """Full PR review through Bot 2's normal pipeline (posts/updates the single review comment)."""
        ensure_paths()
        from bot2.config import BotConfig
        from bot2.diff_parser import DiffParser
        from bot2.reviewer import ReviewOrchestrator
        cfg = BotConfig.from_env()
        client_repo = cfg.github_repository or (self.gh.repo if self.gh else "")
        from dataclasses import replace
        cfg = replace(cfg, github_repository=client_repo, github_token=cfg.github_token or (self.gh.token if self.gh else ""))
        orch = ReviewOrchestrator(cfg)
        diffs = orch.github_client.get_pull_request_files(number)
        raw = orch.github_client.get_pull_request_raw_diff(number)
        if not diffs and raw:
            diffs = DiffParser.parse_full_unified_diff(raw)
        md, findings = orch.run_review(diffs, raw, None)
        orch.publish_review(number, md)
        return verdict_of(findings), findings
