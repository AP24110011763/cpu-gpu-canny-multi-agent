"""Orchestrator: command -> plan -> safety checks -> confirmation -> GitHub commits."""
from __future__ import annotations

import difflib
import posixpath
from typing import Callable, List

from .ai_engine import AIEngine, Edit
from .inspector import inspect_repo
from .models import CREATE, DELETE, UPDATE, Action, BotError, ConflictError, NeedsAIError, Plan, Result
from .security import (MAX_FILE_BYTES, is_important, redact, sanitize_commit_message, validate_path)

MAX_ACTIONS = 10
RETRIES = 3


def apply_edits(text: str, edits: List[Edit]) -> str:
    """Apply search/replace edits. Each 'old' must match exactly once (never guesses)."""
    for old, new in edits:
        if old == "":
            text = text + ("" if not text or text.endswith("\n") else "\n") + new + ("" if new.endswith("\n") else "\n")
            continue
        n = text.count(old)
        if n == 0:
            raise BotError("The requested section was not found in the file, so nothing was changed.")
        if n > 1:
            raise BotError("The requested section appears more than once; please be more specific.")
        text = text.replace(old, new, 1)
    return text


def _diff(path: str, old: str, new: str, limit: int = 30) -> str:
    d = list(difflib.unified_diff(old.splitlines(), new.splitlines(), f"a/{path}", f"b/{path}", lineterm="", n=1))
    return "\n".join(d[:limit] + (["..."] if len(d) > limit else []))


class Bot:
    def __init__(self, repo, ai: AIEngine, confirm: Callable[[str], bool],
                 log: Callable[[str], None] = print, dry_run: bool = False):
        self.repo, self.ai, self.confirm, self.dry_run = repo, ai, confirm, dry_run
        self._log = log
        self._summary = None
        self._files: List[str] = []

    def log(self, msg: str) -> None:
        self._log(redact(msg))

    # ---------- repository understanding ----------
    def refresh(self):
        self._files = self.repo.list_files()
        info = inspect_repo(self.repo, self._files)
        self._summary = info.summary
        return info

    # ---------- main entry ----------
    def run(self, command: str) -> List[Result]:
        info = self.refresh()
        try:
            plan = self.ai.plan(command, self._summary, self._files)
            self._validate_plan(plan)
        except NeedsAIError as e:
            raise BotError(str(e)) from None
        self.log(f"Plan ({len(plan.actions)} action(s)) on {self.repo.name}@{self.repo.branch}:")
        for a in plan.actions:
            self.log(f"  - {a.operation.upper():6} {a.path}" + ("  (guessed from your description)" if a.fuzzy else ""))

        results: List[Result] = []
        runnable: List[Action] = []
        for a in plan.actions:                      # pre-flight checks
            exists = a.path in self._files
            if a.operation == CREATE and exists:
                results.append(Result(a, "skipped", f"'{a.path}' already exists - not overwritten. "
                                                    "Ask me to update it instead."))
            elif a.operation in (UPDATE, DELETE) and not exists:
                results.append(Result(a, "failed", f"'{a.path}' does not exist in the repository."))
            else:
                runnable.append(a)

        if runnable and not self.dry_run:
            reasons = self._confirmation_reasons(runnable)
            if reasons:
                prompt = ("Confirmation required:\n  - " + "\n  - ".join(reasons) +
                          "\nProceed with ALL of these actions?")
                if not self.confirm(prompt):
                    results += [Result(a, "cancelled", "Cancelled - nothing was changed.") for a in runnable]
                    return results
        for a in runnable:
            msg = (sanitize_commit_message(plan.commit_message, "") if len(plan.actions) == 1 else "") \
                or self._default_message(a)
            if self.dry_run:
                results.append(Result(a, "dry-run", f"Would {a.operation} '{a.path}' with message: {msg}"))
                continue
            try:
                results.append({CREATE: self._create, UPDATE: self._update, DELETE: self._delete}[a.operation](a, msg))
            except BotError as e:
                results.append(Result(a, "failed", redact(str(e))))
            except Exception as e:  # never leak internals/secrets
                results.append(Result(a, "failed", redact(f"Unexpected error: {type(e).__name__}")))
        return results

    # ---------- safety ----------
    def _validate_plan(self, plan: Plan) -> None:
        if not plan.actions:
            raise BotError("Nothing to do.")
        if len(plan.actions) > MAX_ACTIONS:
            raise BotError(f"Too many files in one command ({len(plan.actions)}); the limit is {MAX_ACTIONS}.")
        seen = set()
        for a in plan.actions:
            a.path = validate_path(a.path)
            key = (a.operation, a.path)
            if key in seen:
                raise BotError(f"Duplicate action for '{a.path}'.")
            seen.add(key)

    def _confirmation_reasons(self, actions: List[Action]) -> List[str]:
        out: List[str] = []
        deletes = [a for a in actions if a.operation == DELETE]
        if len(deletes) > 1:
            out.append(f"Deleting {len(deletes)} files: " + ", ".join(a.path for a in deletes))
        for a in deletes:
            if is_important(a.path):
                out.append(f"'{a.path}' looks like an important file and will be DELETED")
            elif a.fuzzy:
                out.append(f"'{a.path}' was guessed from your description and will be DELETED")
        for a in actions:
            if a.path.startswith(".github/workflows/"):
                out.append(f"'{a.path}' is a CI workflow file ({a.operation})")
        return list(dict.fromkeys(out))

    @staticmethod
    def _default_message(a: Action) -> str:
        name = posixpath.basename(a.path)
        return {CREATE: f"Add {a.path}", DELETE: f"Delete {a.path}",
                UPDATE: sanitize_commit_message(f"Update {name}: {a.instruction}", f"Update {a.path}")}[a.operation]

    # ---------- operations ----------
    def _create(self, a: Action, msg: str) -> Result:
        text = a.content if a.content is not None else self.ai.generate_file(a.path, a.instruction, self._summary)
        if not text.strip():
            raise BotError("The generated file was empty; nothing was created.")
        if len(text.encode()) > MAX_FILE_BYTES:
            raise BotError("The generated file is too large.")
        if not text.endswith("\n"):
            text += "\n"
        c = self.repo.create(a.path, text, msg)
        self._files.append(a.path)
        return Result(a, "created", f"Created '{a.path}' ({len(text.splitlines())} lines).", c.sha, c.url)

    def _update(self, a: Action, msg: str) -> Result:
        edits = None
        for _ in range(RETRIES):
            got = self.repo.read(a.path)
            if got is None:
                raise BotError(f"'{a.path}' no longer exists.")
            old, sha = got
            if edits is None:
                edits = self.ai.edit_file(a.path, old, a.instruction)
                if not edits:
                    return Result(a, "skipped", "No changes were needed (no commit made).")
            try:
                new = apply_edits(old, edits)
            except BotError as e:
                raise BotError(f"{e} (the file may have changed while editing)") from None
            if new == old:
                return Result(a, "skipped", f"'{a.path}' already up to date (no commit made).")
            if old.strip() and not new.strip():
                raise BotError("That edit would empty the file; use a delete command instead.")
            if len(new.encode()) > MAX_FILE_BYTES:
                raise BotError("The updated file would be too large.")
            try:
                c = self.repo.update(a.path, new, sha, msg)
            except ConflictError:
                self.log(f"  conflict on '{a.path}', re-reading and retrying...")
                continue
            return Result(a, "updated", f"Updated '{a.path}'.", c.sha, c.url, _diff(a.path, old, new))
        raise BotError(f"Could not update '{a.path}' after {RETRIES} attempts because it kept changing.")

    def _delete(self, a: Action, msg: str) -> Result:
        for _ in range(RETRIES):
            sha = self.repo.get_sha(a.path)
            if sha is None:
                return Result(a, "skipped", f"'{a.path}' is already gone (no commit made).")
            try:
                c = self.repo.delete(a.path, sha, msg)
            except ConflictError:
                self.log(f"  conflict on '{a.path}', retrying...")
                continue
            if a.path in self._files:
                self._files.remove(a.path)
            return Result(a, "deleted", f"Deleted '{a.path}'.", c.sha, c.url)
        raise BotError(f"Could not delete '{a.path}' after {RETRIES} attempts.")
