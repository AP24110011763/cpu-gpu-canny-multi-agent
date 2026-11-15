"""Bot 1 as a teammate: the file manager. Same commands from GitHub comments and from the terminal."""
from __future__ import annotations

import posixpath
from dataclasses import dataclass
from typing import Callable, Optional

from . import messages
from .commands import split_command

CODE_EXTS = {".py", ".js", ".ts", ".java", ".c", ".cpp", ".cu", ".h", ".go", ".rs", ".sh", ".rb", ".php"}

HELP = """**What I do:** I keep your repo tidy. I file uploads from `inbox/`, move or delete files, and tell Bot 2 when new code needs a review.

| Say this | I will |
|---|---|
| `/bot1 list [folder]` | list files |
| `/bot1 move OLD NEW` | move a file (never overwrites) |
| `/bot1 delete PATH` | ask you to confirm, then delete it |
| `/bot1 delete PATH confirm` | delete it (recoverable from git history) |
| `/bot1 put FILE in FOLDER` | answer my question about where an inbox file belongs |
| `/bot1 status` | show what is waiting in `inbox/` |
| `/bot1 review PATH` | ask Bot 2 to review a file |

Talk to Bot 2 with `/bot2 help`."""


@dataclass
class Ctx:
    thread: Optional[int] = None       # issue/PR number to reply in
    pr_number: Optional[int] = None
    author: str = "you"
    source: str = "github"             # github | terminal


def is_code(path: str) -> bool:
    return posixpath.splitext(path.lower())[1] in CODE_EXTS


class Bot1Agent:
    name = "bot1"

    def __init__(self, repo_factory: Callable[[], object], gh=None):
        self._repo_factory, self._repo, self.gh = repo_factory, None, gh

    @property
    def repo(self):
        if self._repo is None:
            self._repo = self._repo_factory()
        return self._repo

    def handle(self, text: str, ctx: Ctx) -> str:
        from bot1.errors import BotError
        cmd, args = split_command(text)
        try:
            if cmd in ("help", "hi", "hello", "?"):
                return HELP
            if cmd in ("list", "ls"):
                return self._run("list-files", args)
            if cmd == "move":
                parts = args.replace(" to ", " ").split()
                if len(parts) != 2:
                    return "Usage: `/bot1 move OLD NEW`, for example `/bot1 move docs/a.pdf archive/a.pdf`."
                return self._after_move(parts[1], self._run("move-file", parts[0], parts[1]), ctx)
            if cmd in ("delete", "rm"):
                words = args.split()
                confirmed = bool(words) and words[-1].lower() in ("confirm", "yes")
                path = " ".join(words[:-1] if confirmed else words)
                if not path:
                    return "Usage: `/bot1 delete PATH`."
                if not confirmed:
                    return (f"You asked me to delete `{path}`. Nothing is deleted yet. "
                            f"Reply `/bot1 delete {path} confirm` to go ahead (it stays in git history).")
                return self._run("delete-file", path, confirm=True)
            if cmd == "put":
                return self._put(args, ctx)
            if cmd == "status":
                return self._status()
            if cmd == "review":
                return self._ask_review(args.strip(), ctx)
            return f"I don't know `{cmd}`. Try `/bot1 help`."
        except BotError as e:
            from bot1.safety import redact
            return f"⚠️ {redact(str(e))}"

    # --- commands ---
    def _run(self, action: str, path: str = "", new: str = "", confirm: bool = False) -> str:
        from bot1.requests_menu import run_request
        return "```\n" + run_request(self.repo, action, path, new, confirm) + "\n```"

    def _put(self, args: str, ctx: Ctx) -> str:
        from bot1.errors import BotError
        from bot1.placement import clean_name, join, validate_folder
        words = args.replace(" in ", " ").replace(" to ", " ").split()
        if len(words) != 2:
            return "Usage: `/bot1 put FILE in FOLDER`, for example `/bot1 put notes.txt in docs`."
        name, folder = words
        src = name if name.startswith("inbox/") else f"inbox/{name}"
        dest = join(validate_folder(folder), clean_name(posixpath.basename(src)))
        if src not in self.repo.tree_shas():
            raise BotError(f"I can't find `{src}`. Is it still in inbox/?")
        return self._after_move(dest, self._run("move-file", src, dest), ctx)

    def _status(self) -> str:
        files = sorted(self.repo.tree_shas())
        waiting = [f for f in files if f.startswith("inbox/") and not f.endswith((".gitkeep", "_READ_ME.md"))]
        lines = [f"I can see {len(files)} file(s) in the repo."]
        lines.append("Waiting in `inbox/`: " + (", ".join(f"`{w[6:]}`" for w in waiting) if waiting else "nothing."))
        if waiting:
            lines.append("Tell me where each belongs with `/bot1 put FILE in FOLDER`.")
        return "\n".join(lines)

    def _after_move(self, dest: str, result: str, ctx: Ctx) -> str:
        if is_code(dest) and self.gh is not None:
            result += "\n" + self._ask_review(dest, ctx)
        return result

    def _ask_review(self, path: str, ctx: Ctx) -> str:
        if not path:
            return "Usage: `/bot1 review PATH`."
        if self.gh is None:
            return "(I can only ask Bot 2 for a review when I'm connected to GitHub.)"
        thread = ctx.thread or self.gh.log_issue()
        self.gh.comment(thread, messages.make("bot1", "bot2", "review_request", f"I filed `{path}`. Could you review it?", path=path))
        self.gh.dispatch("bot2-review-request", {"path": path, "thread": thread})
        return f"📨 I asked Bot 2 to review `{path}`. They'll reply in issue #{thread}."
