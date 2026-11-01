"""Plain data classes shared by all modules."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

CREATE, UPDATE, DELETE = "create", "update", "delete"


@dataclass
class Action:
    operation: str                  # create | update | delete
    path: str                       # repo-relative path, e.g. "styles/main.css"
    instruction: str = ""           # what to write / change
    content: Optional[str] = None   # pre-made content (rule-based creates)
    fuzzy: bool = False             # True if the path was guessed from a vague name


@dataclass
class Plan:
    actions: List[Action] = field(default_factory=list)
    commit_message: Optional[str] = None
    notes: str = ""


@dataclass
class Result:
    action: Action
    status: str                     # created|updated|deleted|skipped|cancelled|failed
    message: str
    commit_sha: Optional[str] = None
    url: Optional[str] = None
    diff: str = ""


class BotError(Exception):
    """Expected, user-facing error (safe to print)."""


class ConflictError(BotError):
    """The file changed on GitHub between read and write."""


class NeedsAIError(BotError):
    """The rule-based parser cannot handle this command."""
