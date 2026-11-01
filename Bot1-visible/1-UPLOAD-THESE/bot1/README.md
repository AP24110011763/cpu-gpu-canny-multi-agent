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

## Optional: push straight from your Mac (no github.com)
Double-click `Start-Bot1-Drop.command` (from the zip). Everything you drop into `~/BotDrop` is pushed and filed.
Or: `python3 -m bot1 push report.pdf photo.png`, `python3 -m bot1 push x.txt --to docs`, `python3 -m bot1 watch ~/BotDrop`
(add `--dry-run` to preview). Needs `GITHUB_TOKEN` (Contents: Read and write) and `GITHUB_REPO`.

## Tests
`python3 -m pytest bot1/tests -q`
