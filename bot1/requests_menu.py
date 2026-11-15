"""Ask-me-what-to-do mode: a menu (push / delete / move / list) for your terminal, plus the same
requests as a form on GitHub (Actions tab -> "Bot 1 - Request" -> Run workflow)."""
from __future__ import annotations

import os
import getpass
from pathlib import Path
from typing import Callable, List, Optional

from .errors import BotError
from .safety import validate_path, redact

PROTECTED_PREFIXES = (".github/", "bot1/")     # the bot never edits itself or its workflows
ACTIONS = ("list-files", "delete-file", "move-file")
MAX_LIST = 200


def check_modifiable(path: str) -> str:
    norm = validate_path(path)
    if norm.lower().startswith(PROTECTED_PREFIXES):
        raise BotError(f"'{norm}' is part of Bot 1 itself, so it can't be deleted or moved by the bot.")
    return norm


def list_files(repo, prefix: str = "") -> List[str]:
    prefix = prefix.strip().strip("/")
    files = sorted(repo.tree_shas())
    return [f for f in files if not prefix or f == prefix or f.startswith(prefix + "/")]


def format_list(files: List[str]) -> str:
    if not files:
        return "No files found."
    shown = "\n".join(f"  {i}) {f}" for i, f in enumerate(files[:MAX_LIST], 1))
    more = f"\n  ... and {len(files) - MAX_LIST} more (give a folder to narrow it down)" if len(files) > MAX_LIST else ""
    return f"{len(files)} file(s):\n{shown}{more}"


def run_request(repo, action: str, path: str = "", new_path: str = "", confirm: bool = False) -> str:
    """Do one request and return a human-readable result. Raises BotError for expected problems."""
    action = (action or "").strip()
    if action == "list-files":
        return format_list(list_files(repo, path))
    if action == "delete-file":
        p = check_modifiable(path)
        if not confirm:
            raise BotError("Delete not confirmed. Tick 'confirm' (or type yes) to delete.")
        c = repo.delete_path(p, f"Delete {p}")
        return f"[-] Deleted {p}" + (f"\n    {c.url}" if c.url else "") + "\n    (It can be restored from the git history.)"
    if action == "move-file":
        old, new = check_modifiable(path), check_modifiable(new_path)
        if old == new:
            raise BotError("The new path is the same as the old one.")
        c = repo.move_path(old, new, f"Move {old} to {new}")
        return f"[>] Moved {old} -> {new}" + (f"\n    {c.url}" if c.url else "")
    raise BotError(f"Unknown action {action!r}. Choose one of: {', '.join(ACTIONS)}.")


# ---------- interactive menu ----------

def _local_path(text: str) -> Path:
    t = text.strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "'\"":
        t = t[1:-1]
    return Path(t.replace("\\ ", " ")).expanduser()      # Terminal adds "\ " when you drag a file in


def _pick_repo_file(repo, ask: Callable[[str], str], out: Callable[[str], None], what: str) -> Optional[str]:
    files = [f for f in list_files(repo) if not f.lower().startswith(PROTECTED_PREFIXES)]
    while True:
        t = ask(f"{what} (type the path, 'l' to list files, Enter to cancel): ").strip()
        if not t:
            return None
        if t.lower() == "l":
            out(format_list(files))
            continue
        if t.isdigit() and 1 <= int(t) <= len(files):
            return files[int(t) - 1]
        return t


def run_menu(repo, placer=None, ask: Callable[[str], str] = input, out: Callable[[str], None] = print) -> int:
    from .push import format_result, push_files
    while True:
        out("\nWhat would you like to do?\n"
            "  1) Push a file to the repo\n"
            "  2) Delete a file from the repo\n"
            "  3) Move / rename a file in the repo\n"
            "  4) List the files in the repo\n"
            "  5) Quit")
        choice = ask("Choose 1-5: ").strip()
        try:
            if choice == "1":
                raw = ask("File on your computer (drag it into this window, Enter to cancel): ")
                if not raw.strip():
                    continue
                f = _local_path(raw)
                if not f.is_file():
                    raise BotError(f"Not a file: {f}")
                folder = ask("Folder in the repo (Enter = let Bot 1 choose): ").strip().strip("/") or None
                for r in push_files(repo, [f], folder, placer):
                    out(redact(format_result(r)))
            elif choice == "2":
                p = _pick_repo_file(repo, ask, out, "File to delete")
                if p is None:
                    continue
                p = check_modifiable(p)
                if p not in repo.tree_shas():
                    raise BotError(f"'{p}' was not found in the repository.")
                ok = ask(f"Delete '{p}' from {repo.name}? Type yes to confirm: ").strip().lower() == "yes"
                out(run_request(repo, "delete-file", p, confirm=True) if ok else "Cancelled - nothing was deleted.")
            elif choice == "3":
                old = _pick_repo_file(repo, ask, out, "File to move")
                if old is None:
                    continue
                new = ask("New path, e.g. docs/report.pdf (Enter to cancel): ").strip()
                if not new:
                    continue
                out(run_request(repo, "move-file", old, new))
            elif choice == "4":
                out(format_list(list_files(repo, ask("Folder to list (Enter = everything): "))))
            elif choice in ("5", "q", "quit", "exit"):
                out("Bye!")
                return 0
            else:
                out("Please type a number from 1 to 5.")
        except BotError as e:
            out(f"Error: {redact(str(e))}")


def setup_config(ask: Callable[[str], str] = input, secret: Callable[[str], str] = getpass.getpass,
                 out: Callable[[str], None] = print, repo_override: Optional[str] = None,
                 branch_override: Optional[str] = None):
    """load_config(), but ask for the repo/token on first use and offer to save them privately."""
    from .config import load_config
    try:
        return load_config(repo_override, branch_override)
    except BotError:
        pass
    out("First-time setup (asked once).")
    repo = repo_override or os.environ.get("GITHUB_REPO", "").strip() or ask("Your repo (looks like username/repo): ").strip()
    token = os.environ.get("GITHUB_TOKEN", "").strip() or secret("Paste your GitHub token (hidden while typing): ").strip()
    os.environ["GITHUB_REPO"], os.environ["GITHUB_TOKEN"] = repo, token
    cfg = load_config(repo_override, branch_override)
    if ask("Save these on this computer so you aren't asked again? (y/N): ").strip().lower() == "y":
        envfile = Path.home() / ".bot1.env"
        old = os.umask(0o077)
        try:
            envfile.write_text(f"GITHUB_REPO={repo}\nGITHUB_TOKEN={token}\n", encoding="utf-8")
        finally:
            os.umask(old)
        out(f"Saved privately in {envfile}.")
    return cfg
