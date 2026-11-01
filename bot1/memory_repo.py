"""In-memory fake repository: used by --demo and by the unit tests."""
from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Tuple

from .github_client import CommitInfo
from .models import BotError, ConflictError

SAMPLE = {
    "README.md": "# Demo Project\n\nA small sample repository.\n",
    "src/app.py": "def greet(name):\n    return 'Hello ' + name\n",
    "src/db.py": "DB_HOST = 'localhost'\nDB_PORT = 5432\n\ndef connect():\n    return (DB_HOST, DB_PORT)\n",
    "styles/main.css": "body { margin: 0; }\n",
    "tests/test_app.py": "from src.app import greet\n\ndef test_greet():\n    assert greet('A') == 'Hello A'\n",
    "requirements.txt": "flask\npytest\n",
}


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


class MemoryRepo:
    def __init__(self, files: Optional[Dict[str, str]] = None, name: str = "demo/repo", branch: str = "main"):
        self.files: Dict[str, str] = dict(files if files is not None else SAMPLE)
        self.name, self.branch = name, branch
        self.commits: List[Tuple[str, str, str]] = []   # (op, path, message)
        self.fail_next_writes = 0                        # simulate conflicts in tests

    def list_files(self): return sorted(self.files)

    def get_sha(self, path): return _sha(self.files[path]) if path in self.files else None

    def read(self, path):
        if path not in self.files:
            return None
        return self.files[path], _sha(self.files[path])

    def _commit(self, op, path, message):
        self.commits.append((op, path, message))
        return CommitInfo(sha=_sha(f"{len(self.commits)}{path}")[:40], url=f"memory://commit/{len(self.commits)}")

    def _maybe_conflict(self):
        if self.fail_next_writes > 0:
            self.fail_next_writes -= 1
            raise ConflictError("simulated conflict")

    def create(self, path, text, message):
        if path in self.files:
            raise BotError(f"'{path}' already exists; it was not overwritten.")
        self.files[path] = text
        return self._commit("create", path, message)

    def update(self, path, text, sha, message):
        self._maybe_conflict()
        if path not in self.files or _sha(self.files[path]) != sha:
            raise ConflictError("stale sha")
        self.files[path] = text
        return self._commit("update", path, message)

    def delete(self, path, sha, message):
        self._maybe_conflict()
        if path not in self.files or _sha(self.files[path]) != sha:
            raise ConflictError("stale sha")
        del self.files[path]
        return self._commit("delete", path, message)
