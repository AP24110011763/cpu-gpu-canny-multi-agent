"""Tests for the team layer: messages, commands, permissions, both agents and the bot<->bot handoff."""
import subprocess
from types import SimpleNamespace

import pytest

from team import events, hooks, messages
from team.__main__ import chat
from team.bot1_agent import Bot1Agent, Ctx
from team.bot2_agent import Bot2Agent
from team.commands import find_mentions, is_authorized, split_command


class FakeGH:
    repo, owner = "me/proj", "me"

    def __init__(self, files=None):
        self.comments, self.dispatches, self.issues, self.files = [], [], [], files or {}

    def comment(self, n, body): self.comments.append((n, body))
    def find_comment(self, n, needle): return any(n == c[0] and needle in c[1] for c in self.comments)
    def log_issue(self): return 1
    def dispatch(self, t, p): self.dispatches.append((t, p))
    def get_file(self, path, ref=None): return self.files.get(path)
    def create_issue(self, title, body, labels=None, assignees=None):
        self.issues.append((title, body, labels, assignees)); return 50 + len(self.issues)


class FakeRepo:
    def __init__(self, files): self.files, self.log = dict.fromkeys(files, "sha"), []
    def tree_shas(self): return dict(self.files)
    def delete_path(self, p, m): self.log.append(("delete", p)); self.files.pop(p); return SimpleNamespace(url="")
    def move_path(self, a, b, m): self.log.append(("move", a, b)); self.files[b] = self.files.pop(a); return SimpleNamespace(url="")


BAD = 'import os\nPASSWORD = "hunter2hunter2"\nos.system("rm -rf " + x)\nresult = eval(user_input)\n'
GOOD = '"""Add numbers."""\n\n\ndef add(a: int, b: int) -> int:\n    return a + b\n'


# ---- messages / commands ----
def test_message_roundtrip():
    body = messages.make("bot1", "bot2", "review_request", "hi", path="docs/my file.py")
    meta = messages.parse(body)
    assert meta["from"] == "bot1" and meta["to"] == "bot2" and meta["path"] == "docs/my file.py"
    assert messages.parse("just a human comment") is None


def test_mentions():
    assert find_mentions("/bot1 list docs") == [("bot1", "list docs")]
    assert find_mentions("@Bot2: review PR 3") == [("bot2", "review PR 3")]
    assert find_mentions("hello\n/bot1 status\n/bot2 help") == [("bot1", "status"), ("bot2", "help")]
    assert find_mentions("mail me at x@bot1.com") == []
    assert find_mentions("```\n/bot1 delete everything\n```") == []


def test_permissions_and_split():
    assert is_authorized("OWNER") and is_authorized("collaborator")
    assert not is_authorized("NONE") and not is_authorized("CONTRIBUTOR") and not is_authorized("")
    assert split_command("Delete a.txt confirm") == ("delete", "a.txt confirm")
    assert split_command("") == ("help", "")


# ---- Bot 1 ----
def agent(files=("docs/a.pdf", "inbox/notes.txt"), gh=None):
    repo = FakeRepo(files)
    return Bot1Agent(lambda: repo, gh), repo


def test_bot1_list_and_help():
    a, _ = agent()
    assert "docs/a.pdf" in a.handle("list", Ctx())
    assert "/bot1 move" in a.handle("help", Ctx())


def test_bot1_delete_needs_confirmation():
    a, repo = agent()
    out = a.handle("delete docs/a.pdf", Ctx())
    assert "confirm" in out and repo.log == []
    a.handle("delete docs/a.pdf confirm", Ctx())
    assert repo.log == [("delete", "docs/a.pdf")]


def test_bot1_refuses_protected_paths():
    a, repo = agent(files=(".github/workflows/x.yml",))
    assert "part of Bot 1 itself" in a.handle("delete .github/workflows/x.yml confirm", Ctx()) or repo.log == []
    assert repo.log == []


def test_bot1_move_code_asks_bot2():
    gh = FakeGH()
    a, repo = agent(files=("old/a.py",), gh=gh)
    out = a.handle("move old/a.py src/a.py", Ctx(thread=7))
    assert repo.log == [("move", "old/a.py", "src/a.py")] and "asked Bot 2" in out
    assert gh.dispatches == [("bot2-review-request", {"path": "src/a.py", "thread": 7})]
    meta = messages.parse(gh.comments[0][1])
    assert (meta["from"], meta["to"], meta["type"]) == ("bot1", "bot2", "review_request")


def test_bot1_put_answers_its_question():
    a, repo = agent(files=("inbox/notes.txt",))
    a.handle("put notes.txt in docs", Ctx())
    assert repo.log == [("move", "inbox/notes.txt", "docs/notes.txt")]
    assert "can't find" in a.handle("put ghost.txt in docs", Ctx())


def test_bot1_status_lists_inbox():
    a, _ = agent()
    assert "notes.txt" in a.handle("status", Ctx())


# ---- Bot 2 ----
def test_bot2_flags_bad_file_and_approves_good_one():
    files = {"bad.py": BAD, "good.py": GOOD}
    b = Bot2Agent(gh=None, read_file=files.get)
    bad = b.handle("review bad.py", Ctx())
    assert "Needs fixes" in bad and "CRITICAL" in bad
    assert "Approved" in b.handle("review good.py", Ctx())
    assert "couldn't read" in b.handle("review nope.py", Ctx())


def test_bot2_explain():
    b = Bot2Agent(gh=None, read_file={"bad.py": BAD}.get)
    out = b.handle("explain 1 bad.py", Ctx())
    assert "Why it matters" in out and "How to fix it" in out
    assert "pick a number" in b.handle("explain 99 bad.py", Ctx())


def test_bot2_pr_review_uses_pipeline():
    b = Bot2Agent(gh=FakeGH(), pr_reviewer=lambda n: ("approved", []))
    assert "Reviewed PR #3" in b.handle("review PR 3", Ctx())
    assert "Reviewed PR #9" in b.handle("review", Ctx(pr_number=9))


# ---- GitHub comment flow ----
def comment_event(body, assoc="OWNER", user_type="User"):
    return {"comment": {"body": body, "author_association": assoc, "user": {"login": "me", "type": user_type}},
            "issue": {"number": 4}}


def test_comment_gets_reply_in_thread():
    gh = FakeGH()
    assert events.handle_comment(comment_event("/bot2 help"), gh, lambda: None) == 1
    n, body = gh.comments[0]
    assert n == 4 and "What I do" in body and messages.parse(body)["from"] == "bot2"


def test_stranger_cannot_command_bots():
    gh = FakeGH()
    repo = FakeRepo(["docs/a.pdf"])
    events.handle_comment(comment_event("/bot1 delete docs/a.pdf confirm", assoc="NONE"), gh, lambda: repo)
    assert repo.log == [] and "only the repository owner" in gh.comments[0][1]


def test_bots_ignore_bot_comments():
    gh = FakeGH()
    body = messages.make("bot1", "user", "reply", "try /bot2 help")
    assert events.handle_comment(comment_event(body, user_type="Bot"), gh, lambda: None) == 0
    assert gh.comments == []


# ---- bot <-> bot ----
def test_full_handoff_needs_fix_opens_issue():
    gh = FakeGH(files={"src/bad.py": BAD})
    assert events.handle_review_request({"path": "src/bad.py", "thread": 1}, gh) == "needs-fix"
    meta = messages.parse(gh.comments[0][1])
    assert (meta["from"], meta["to"], meta["verdict"]) == ("bot2", "bot1", "needs-fix")
    event, payload = gh.dispatches[0]
    assert event == "bot1-review-result"
    events.handle_review_result(payload, gh)
    assert gh.issues[0][0] == "Needs fix: src/bad.py" and gh.issues[0][2:] == (["needs-fix"], ["me"])


def test_full_handoff_approved_just_reports():
    gh = FakeGH(files={"src/good.py": GOOD})
    assert events.handle_review_request({"path": "src/good.py", "thread": 1}, gh) == "approved"
    events.handle_review_result(gh.dispatches[0][1], gh)
    assert gh.issues == [] and "approved" in gh.comments[-1][1]


def test_handoff_rejects_unsafe_path():
    gh = FakeGH()
    assert events.handle_review_request({"path": "../../etc/passwd"}, gh).startswith("refused")
    assert gh.comments == [] and gh.dispatches == []


# ---- hooks / organizer ----
def test_after_organize_asks_once_and_hands_off_code():
    gh = FakeGH()
    moves = [SimpleNamespace(src="inbox/x.weird", dest="", status="asked", reason="unsure"),
             SimpleNamespace(src="inbox/a.py", dest="src/a.py", status="moved", reason="")]
    hooks.after_organize(moves, gh)
    hooks.after_organize(moves[:1], gh)               # same question again: must not repeat
    questions = [c for c in gh.comments if messages.parse(c[1])["type"] == "question"]
    assert len(questions) == 1 and "/bot1 put x.weird in" in questions[0][1]
    assert gh.dispatches == [("bot2-review-request", {"path": "src/a.py", "thread": 1})]


def test_organizer_leaves_unsure_file_in_inbox(tmp_path):
    from bot1.organize import organize
    run = lambda *a: subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)
    run("init", "-q"); (tmp_path / "inbox").mkdir()
    (tmp_path / "inbox" / "mystery.zzz").write_text("hello")
    (tmp_path / "inbox" / "notes.md").write_text("# hi")
    run("add", "-A")
    moves, _ = organize(tmp_path, ask=True)
    status = {m.src: m.status for m in moves}
    assert status["inbox/mystery.zzz"] == "asked" and (tmp_path / "inbox" / "mystery.zzz").exists()
    assert status["inbox/notes.md"] in ("moved", "asked")
    moves2, _ = organize(tmp_path, ask=False)         # without asking it files it as before
    assert {m.status for m in moves2 if m.src.endswith(".zzz")} <= {"moved"}


# ---- terminal chat ----
def test_terminal_chat_routes_to_both_bots():
    a, _ = agent()
    bots = {"bot1": a, "bot2": Bot2Agent(gh=None, read_file={"g.py": GOOD}.get)}
    lines = iter(["help", "/bot2 review g.py", "bot2", "status", "/quit"])
    out = []
    assert chat(inp=lambda _p: next(lines), out=out.append, bots=bots) == 0
    text = "\n".join(out)
    assert "bot1: **What I do:** I keep your repo tidy" in text
    assert "Approved" in text and "Static analysis: on" in text
