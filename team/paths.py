"""Makes the Bot2/ package importable from the repo root (Bot 2 lives in Bot2/bot2/)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def ensure_paths() -> None:
    for p in (ROOT, ROOT / "Bot2"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
