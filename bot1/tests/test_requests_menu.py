from pathlib import Path

import pytest

from bot1.errors import BotError
from bot1.requests_menu import run_menu, run_request


class C:
    url = "http://c"


class FakeRepo:
    name = "me/repo"
    branch = "main"

    def __init__(self, files):
        self.files = dict(files)

    def tree_shas(self):
        return {k: "sha" for k in self.files}

    def create_bytes(self, path, data, message):
        assert path not in self.files
        self.files[path] = data
        return C()

    def delete_path(self, path, message):
        if path not in self.files:
            raise BotError("not found")
        del self.files[path]
        return C()

    def move_path(self, old, new, message):
        if new in self.files:
            raise BotError("exists")
        self.files[new] = self.files.pop(old)
        return C()


def scripted(*answers):
    it = iter(answers)
    return lambda prompt="": next(it)


def test_delete_needs_confirmation():
    r = FakeRepo({"docs/a.txt": b"x"})
    with pytest.raises(BotError):
        run_request(r, "delete-file", "docs/a.txt")
    assert "docs/a.txt" in r.files
    assert "Deleted" in run_request(r, "delete-file", "docs/a.txt", confirm=True)
    assert "docs/a.txt" not in r.files


def test_protected_and_bad_paths_refused():
    r = FakeRepo({".github/workflows/x.yml": b"x", "bot1/cli.py": b"x", ".env": b"x"})
    for p in (".github/workflows/x.yml", "bot1/cli.py", ".env", "../etc/passwd", "/abs"):
        with pytest.raises(BotError):
            run_request(r, "delete-file", p, confirm=True)
    assert len(r.files) == 3


def test_move_never_overwrites():
    r = FakeRepo({"a.txt": b"1", "docs/a.txt": b"2"})
    with pytest.raises(BotError):
        run_request(r, "move-file", "a.txt", "docs/a.txt")
    run_request(r, "move-file", "a.txt", "docs/b.txt")
    assert "docs/b.txt" in r.files and "a.txt" not in r.files


def test_list_and_unknown_action():
    r = FakeRepo({"docs/a.txt": b"1", "src/b.py": b"2"})
    assert "docs/a.txt" in run_request(r, "list-files", "docs") and "src/b.py" not in run_request(r, "list-files", "docs")
    with pytest.raises(BotError):
        run_request(r, "nuke", "")


def test_menu_delete_cancel_and_confirm():
    r = FakeRepo({"docs/a.txt": b"1"})
    out = []
    run_menu(r, None, scripted("2", "docs/a.txt", "no", "5"), out.append)
    assert "docs/a.txt" in r.files and any("Cancelled" in o for o in out)
    run_menu(r, None, scripted("2", "l", "1", "yes", "5"), out.append)
    assert "docs/a.txt" not in r.files


def test_menu_push_and_move(tmp_path: Path):
    f = tmp_path / "notes.txt"
    f.write_text("hello")
    r = FakeRepo({"seed.md": b"s"})
    out = []
    run_menu(r, None, scripted("1", str(f), "docs", "3", "docs/notes.txt", "archive/notes.txt", "5"), out.append)
    assert "archive/notes.txt" in r.files and "docs/notes.txt" not in r.files
