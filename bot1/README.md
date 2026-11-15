# Bot 1 - File Organizer + Repo Reporter (lives in your repo)

**Give it a file:** on github.com open the `inbox/` folder -> *Add file* -> *Upload files* -> *Commit changes*.
Within about a minute Bot 1 moves the file to the right folder (`report.pdf` -> `docs/`, `logo.png` -> `assets/images/`),
commits it, and posts an update to the issue **"Repo Activity Log (Bot 1)"**.

**How it picks the folder:** (1) a sub-folder you used (`inbox/docs/x.pdf` -> `docs/x.pdf`), (2) where files of the same
type already live in your repo, (3) a folder name your repo already uses (`docs/`, `data/`, `images/`...), (4) defaults
(`src/`, `docs/`, `data/`, `assets/images/`, `tests/`, `scripts/`, `config/`, `misc/`). With an `ANTHROPIC_API_KEY` repo
secret, Claude picks from your real folder layout instead.

**Updates:** the Activity Log issue gets a comment for every push, pull request and issue (who, what changed, links)
and for every file Bot 1 files.

**Safety:** nothing is overwritten (same name + different content -> `name_2.ext`); identical files are not added twice;
files that look like passwords/keys are deleted and reported (revoke them: they stay in git history); `.git/`, `.github/`,
`bot1/` are off limits; `..` and absolute paths are refused.

## Ask-me mode: Bot 1 asks what you want to do
Run `python3 -m bot1` (or double-click `Start-Bot1-Menu.command` on a Mac). Bot 1 asks:
1) push a file, 2) delete a file, 3) move/rename a file, 4) list files, 5) quit.
Deleting always asks you to type `yes`; nothing is overwritten; `.github/` and `bot1/` can't be deleted or moved.
First run asks for your repo (`username/repo`) and a GitHub token (Contents: Read and write) and can save them privately.

**On github.com:** Actions tab -> **Bot 1 - Request** -> *Run workflow* -> pick `list-files`, `delete-file` or `move-file`,
fill in the path (tick *confirm* for a delete) and run it. To *push* a file there, upload it into `inbox/`.

## Optional: push straight from your Mac (no github.com)
Double-click `Start-Bot1-Drop.command` (from the zip). Everything you drop into `~/BotDrop` is pushed and filed.
Or: `python3 -m bot1 push report.pdf photo.png`, `python3 -m bot1 push x.txt --to docs`, `python3 -m bot1 watch ~/BotDrop`
(add `--dry-run` to preview). Needs `GITHUB_TOKEN` (Contents: Read and write) and `GITHUB_REPO`.

## Tests
`python3 -m pytest bot1/tests -q`
