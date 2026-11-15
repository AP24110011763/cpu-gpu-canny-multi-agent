"""Finding who is being addressed, and checking they are allowed to give orders."""
from __future__ import annotations

import re
from typing import List, Tuple

# "/bot1 list docs" is the recommended form. "@bot1" also works, but GitHub would also
# notify a real GitHub user named "bot1", so the help text teaches the slash form.
_MENTION = re.compile(r"(?<![\w/@.])[/@](bot1|bot2)\b[:,]?[ \t]*(.*)", re.I)
ALLOWED = {"OWNER", "MEMBER", "COLLABORATOR"}


def find_mentions(text: str) -> List[Tuple[str, str]]:
    """[(bot, command text)] for each line that addresses a bot. Ignores fenced code blocks."""
    out, in_code = [], False
    for line in (text or "").splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        m = _MENTION.search(line)
        if m:
            out.append((m.group(1).lower(), m.group(2).strip()))
    return out


def is_authorized(author_association: str) -> bool:
    return (author_association or "").upper() in ALLOWED


def split_command(rest: str) -> Tuple[str, str]:
    parts = (rest or "").strip().split(None, 1)
    return (parts[0].lower().rstrip(":"), parts[1].strip() if len(parts) > 1 else "") if parts else ("help", "")
