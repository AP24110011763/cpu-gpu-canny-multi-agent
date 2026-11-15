"""What each bot does when GitHub wakes it up: a comment, a message from the other bot, or the weekly schedule."""
from __future__ import annotations

import posixpath
from collections import Counter
from typing import Callable, Dict, Optional

from . import messages
from .bot1_agent import Bot1Agent, Ctx, is_code
from .bot2_agent import Bot2Agent, counts, review_text, summarize
from .commands import find_mentions, is_authorized

COA_DIRS = ("agents/", "tools/", "llm/", "orchestrator.py", "schemas.py", "main.py", "app.py")
INTRO = {
    "bot1": "Hi, I'm **Bot 1**, your file manager. Upload files to `inbox/` and I'll file them in the right folder. "
            "If I'm unsure where something belongs I'll ask you. When I file code I'll ask Bot 2 to review it. "
            "I also post a weekly digest. Say `/bot1 help` to see what I can do.",
    "bot2": "Hi, I'm **Bot 2**, your code reviewer. I review every pull request and every code file Bot 1 files. "
            "I tell you what's wrong and why, and Bot 1 gets my verdict. I also warn you when project code changes "
            "without tests. Say `/bot2 help` to see what I can do.",
}


def say(gh, number: int, sender: str, to: str, mtype: str, text: str, **meta) -> None:
    gh.comment(number, messages.make(sender, to, mtype, text, **meta))


# ---------------- you -> bot (GitHub comment) ----------------
def handle_comment(event: dict, gh, repo_factory: Callable[[], object]) -> int:
    """Answer '/bot1 ...' and '/bot2 ...' comments. Returns how many commands were handled."""
    comment, issue = event.get("comment") or {}, event.get("issue") or {}
    body = comment.get("body") or ""
    if messages.is_bot_message(body) or (comment.get("user") or {}).get("type") == "Bot":
        return 0
    number = issue.get("number")
    mentions = find_mentions(body)
    if not mentions or not number:
        return 0
    agents: Dict[str, object] = {"bot1": Bot1Agent(repo_factory, gh), "bot2": Bot2Agent(gh)}
    ctx = Ctx(thread=number, pr_number=number if issue.get("pull_request") else None,
              author=(comment.get("user") or {}).get("login", "you"))
    for bot, rest in mentions:
        if not is_authorized(comment.get("author_association", "")):
            say(gh, number, bot, "user", "refused",
                "Sorry, only the repository owner and collaborators can give me commands.")
            continue
        try:
            reply = agents[bot].handle(rest, ctx)
        except Exception as e:                      # never go silent on the user
            reply = f"⚠️ Something went wrong ({type(e).__name__}). Check the workflow run for details."
        say(gh, number, bot, "user", "reply", reply)
    return len(mentions)


# ---------------- bot1 -> bot2 -> bot1 ----------------
def handle_review_request(payload: dict, gh) -> str:
    """Bot 2 wakes up because Bot 1 asked for a review. Replies in the thread and tells Bot 1 the verdict."""
    from bot1.errors import BotError
    from bot1.safety import validate_path
    try:
        path = validate_path(payload.get("path", ""))
    except BotError as e:
        return f"refused: {e}"
    thread = int(payload.get("thread") or gh.log_issue())
    text = gh.get_file(path)
    if text is None:
        say(gh, thread, "bot2", "bot1", "review_result", f"I couldn't read `{path}` (binary or missing), so I skipped it.", path=path, verdict="skipped")
        return "skipped"
    verdict, findings = review_text(path, text)
    head = "✅ Approved" if verdict == "approved" else "⚠️ Needs fixes"
    detail = ("\n\n" + summarize(findings) + f"\n\nAsk me `/bot2 explain N {path}` for details.") if findings else ""
    say(gh, thread, "bot2", "bot1", "review_result", f"{head}: `{path}` has {counts(findings)}.{detail}", path=path, verdict=verdict)
    gh.dispatch("bot1-review-result", {"path": path, "thread": thread, "verdict": verdict,
                                       "counts": counts(findings), "summary": summarize(findings)})
    return verdict


def handle_review_result(payload: dict, gh) -> str:
    """Bot 1 wakes up with Bot 2's verdict and tells you what happens next."""
    path, verdict, thread = payload.get("path", "?"), payload.get("verdict", ""), int(payload.get("thread") or gh.log_issue())
    owner = gh.owner
    if verdict == "approved":
        say(gh, thread, "bot1", "user", "update", f"@{owner} Bot 2 approved `{path}` ({payload.get('counts')}). Nothing for you to do.", path=path)
        return "approved"
    if verdict == "needs-fix":
        n = gh.create_issue(f"Needs fix: {path}", f"Bot 2 reviewed `{path}` and found problems ({payload.get('counts')}):\n\n"
                            f"{payload.get('summary', '')}\n\nReview thread: #{thread}", ["needs-fix"], [owner])
        say(gh, thread, "bot1", "user", "alert", f"@{owner} Bot 2 found problems in `{path}` ({payload.get('counts')}). I opened #{n} labelled `needs-fix` and assigned it to you.", path=path)
        return "needs-fix"
    say(gh, thread, "bot1", "user", "update", f"Bot 2 skipped `{path}` (couldn't read it).", path=path)
    return "skipped"


# ---------------- proactive ----------------
def post_intros(gh) -> None:
    log = gh.log_issue()
    for bot in ("bot1", "bot2"):
        say(gh, log, bot, "user", "intro", INTRO[bot])


def weekly_digest(gh, days: int = 7) -> None:
    log = gh.log_issue()
    commits = gh.recent_commits(days)
    if not commits:
        say(gh, log, "bot1", "user", "digest", f"Quiet week: no commits in the last {days} days.")
        return
    files = [f for c in commits for f in c["files"]]
    tops = Counter(f.split("/")[0] if "/" in f else "(root)" for f in files)
    authors = Counter(c["author"] for c in commits)
    lines = [f"**Weekly digest** (last {days} days): {len(commits)} commit(s) by " + ", ".join(f"{a} ({n})" for a, n in authors.most_common(5)) + ".",
             "Most active folders: " + ", ".join(f"`{d}` ({n})" for d, n in tops.most_common(5)) + "."]
    coa = sorted({f for f in files if f.startswith(COA_DIRS)})
    if coa:
        lines.append("Project (COA) files changed: " + ", ".join(f"`{f}`" for f in coa[:8]) + (" …" if len(coa) > 8 else ""))
    say(gh, log, "bot1", "user", "digest", "\n".join(lines))
    if coa and not any(f.startswith("tests/") for f in files):
        say(gh, log, "bot2", "user", "warning", "Heads-up: project code changed this week (" + ", ".join(f"`{f}`" for f in coa[:5]) +
            ") but nothing under `tests/` changed. Does it need new tests?")
