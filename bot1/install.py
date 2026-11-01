"""Upload the bot's own files into the repo (so GitHub Actions can run it)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict

from .models import BotError

SKIP_PARTS = {"__pycache__", ".pytest_cache", "tests"}


def collect_files() -> Dict[str, str]:
    root = Path(__file__).resolve().parent
    files: Dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if p.is_file() and not (SKIP_PARTS & set(rel.parts)) and p.suffix != ".pyc" and p.name != ".env":
            files[f"bot1/{rel.as_posix()}"] = p.read_text(encoding="utf-8")
    wf = root.parent / ".github" / "workflows"
    if wf.is_dir():
        for p in sorted(wf.glob("*.yml")):
            files[f".github/workflows/{p.name}"] = p.read_text(encoding="utf-8")
    return files


def install(repo, log: Callable[[str], None] = print) -> int:
    """Create/update the bot files in the repo. Returns the number of failures."""
    failed = 0
    for path, text in collect_files().items():
        try:
            got = repo.read(path)
            if got is None:
                repo.create(path, text, f"Install Bot 1: add {path}")
                log(f"[+] added    {path}")
            elif got[0] == text:
                log(f"[=] same     {path}")
            else:
                repo.update(path, text, got[1], f"Install Bot 1: update {path}")
                log(f"[~] updated  {path}")
        except BotError as e:
            failed += 1
            hint = ""
            if path.startswith(".github/workflows/"):
                hint = ("  -> your token needs the 'Workflows: Read and write' permission, "
                        "or create this file by hand on github.com")
            log(f"[!] failed   {path}: {e}{hint}")
    return failed
