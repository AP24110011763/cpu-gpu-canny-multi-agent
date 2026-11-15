"""Hidden-marker messages. Every bot comment carries one, so bots (and humans) can tell who said what.

    <!-- bot-msg from=bot1 to=bot2 type=review_request path=src/a.py -->
"""
from __future__ import annotations

import re
from typing import Dict, Optional
from urllib.parse import quote, unquote

BOTS = {"bot1": "Bot 1", "bot2": "Bot 2"}
ICONS = {"bot1": "📂", "bot2": "🔍"}
_MARK = re.compile(r"<!--\s*bot-msg\s+(.*?)\s*-->", re.S)


def make(sender: str, to: str, mtype: str, text: str, **meta: str) -> str:
    fields = {"from": sender, "to": to, "type": mtype, **{k: str(v) for k, v in meta.items()}}
    marker = "<!-- bot-msg " + " ".join(f"{k}={quote(v, safe='/._-')}" for k, v in fields.items()) + " -->"
    title = f"{ICONS.get(sender, '🤖')} **{BOTS.get(sender, sender)}**"
    if to in BOTS:
        title += f" → {BOTS[to]}"
    return f"{marker}\n{title}\n\n{text}"


def parse(body: str) -> Optional[Dict[str, str]]:
    m = _MARK.search(body or "")
    if not m:
        return None
    out: Dict[str, str] = {}
    for part in m.group(1).split():
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = unquote(v)
    return out if "from" in out else None


def is_bot_message(body: str) -> bool:
    return parse(body) is not None
