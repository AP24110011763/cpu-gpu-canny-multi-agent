# Bot 1 - AI-Powered GitHub File Management Bot

Type a plain-English command; the bot creates, updates or deletes files in **this repository** and commits
them to GitHub automatically (each API commit is already "pushed").

```
python -m bot1 "Create a Python calculator program"
```

Everything lives in the `bot1/` folder plus `.github/workflows/bot1.yml`. No existing file is touched.

## How it works

```
command -> understand (Claude, or offline rules) -> plan -> safety checks -> confirm if risky -> commit via GitHub API
```

| File | Job |
|---|---|
| `cli.py` | command line + interactive mode |
| `bot.py` | the main logic: plan, confirm, create / update / delete, retry on conflicts |
| `ai_engine.py`, `rule_parser.py`, `templates.py` | understanding commands and writing content |
| `github_client.py` | GitHub API (PyGithub) |
| `inspector.py` | detects languages, frameworks, dependencies, folders |
| `security.py` | path validation, protected files, secret redaction |
| `memory_repo.py` | fake repo for `--demo` and tests |

## Setup (step by step)

1. **Python 3.10+**, then from the repo root: `pip install -r bot1/requirements.txt`
2. **Create a GitHub token (least privilege)**: GitHub -> Settings -> Developer settings -> Personal access tokens ->
   *Fine-grained tokens* -> Generate. Choose **Only select repositories** -> pick this one.
   Permissions: **Contents: Read and write** (Metadata: Read is added automatically). Nothing else.
3. *(Optional but recommended)* get an Anthropic API key for full natural-language understanding.
4. **Configure environment variables**
   - Local: `cp bot1/.env.example .env`, fill in the values. `.env` is git-ignored and the bot refuses to touch it.
   - Or export them: `export GITHUB_TOKEN=... GITHUB_REPO=owner/name ANTHROPIC_API_KEY=...` (Windows PowerShell: `$env:GITHUB_TOKEN="..."`).
5. Make sure the root `.gitignore` contains `.env` (add the line if not).
6. Try it safely first: `python -m bot1 --demo` (fake repo, no token) and `python -m bot1 --dry-run "..."`.

| Variable | Required | Meaning |
|---|---|---|
| `GITHUB_TOKEN` | yes | fine-grained token (never hard-coded) |
| `GITHUB_REPO` | yes | `owner/name` |
| `ANTHROPIC_API_KEY` | no | enables full AI; without it, the offline parser is used |
| `BOT_MODEL` | no | model name (default `claude-sonnet-5-5`) |
| `BOT_BRANCH` | no | work on this branch instead of the default branch |

## Running on GitHub Actions (optional)

1. Repo -> Settings -> Secrets and variables -> Actions -> add `ANTHROPIC_API_KEY` (optional).
   `GITHUB_TOKEN` is provided automatically.
2. Settings -> Actions -> General -> Workflow permissions -> **Read and write**.
3. Actions tab -> **Bot 1 - File Management** -> *Run workflow* -> type the command.
   `dry_run` is on by default; untick it to really commit. Tick `confirm` to pre-approve risky deletions.

The built-in token cannot edit `.github/workflows/*` - GitHub rejects it, which is a good thing.

## Command line

```
python -m bot1                       # interactive
python -m bot1 "command"             # one-shot
  --dry-run   show the plan, change nothing
  --yes       auto-confirm risky operations (blocked paths stay blocked)
  --no-ai     force the offline parser
  --info      show the repository analysis
  --branch X  use branch X        --repo owner/name   --demo
```

## Example commands and expected responses

```
$ python -m bot1 "Create a login.java file"
Plan (1 action(s)) on you/repo@main:
  - CREATE login.java
[+] CREATED   Created 'login.java' (6 lines).
            commit 3fa91c2  https://github.com/you/repo/commit/3fa91c2...

$ python -m bot1 "Create a Python calculator program"
  - CREATE src/calculator.py          <- placed where your other .py files live

$ python -m bot1 "Add a CSS file inside the styles folder"
  - CREATE styles/style.css

$ python -m bot1 "Delete the test file"
Confirmation required:
  - 'tests/test_app.py' was guessed from your description and will be DELETED
Proceed with ALL of these actions?
Type 'yes' to confirm: yes
[-] DELETED   Deleted 'tests/test_app.py'.

$ python -m bot1 "Update my existing README"      # needs the AI key
[~] UPDATED   Updated 'README.md'.     (+ a short diff is printed)

$ python -m bot1 'append "Run tests with pytest" to README.md'     # works offline
$ python -m bot1 'replace "localhost" with "db.example.com" in src/db.py'    # works offline
$ python -m bot1 "Change the database connection code"   # AI: edits only the needed lines
```

Other responses: `[=] SKIPPED` (file already exists - never overwritten; or edit made no change, so no empty commit),
`[x] CANCELLED` (you said no), `[!] FAILED` (reason is printed), `[?] DRY-RUN`.

## Safety and security

- **No secrets in code.** Token and keys come only from environment variables / GitHub Secrets. Logs, errors and results
  pass through a redactor that masks known secrets and `ghp_...`, `github_pat_...`, `sk-ant-...` patterns.
- **Least privilege.** Fine-grained token with *Contents* only, limited to one repo; the workflow requests only `contents: write`.
- **Path validation.** Rejects `..`, absolute paths, backslashes, odd characters and folders. `.git/` and `.env*`
  (except `.env.example`) can never be touched.
- **Confirmation** before deleting: more than one file, an important file (README, LICENSE, `.github/`, `bot1/`, dependency
  manifests, entry points), or a file the bot only *guessed* from a vague description. Workflow files also need confirmation.
- **Never overwrites.** `create` refuses existing files. Updates are exact search/replace edits (each match must be
  unique), so unrelated code is preserved; an edit that would empty a file is refused. Max 10 files per command.
- **Conflicts.** On a stale-SHA/409 conflict the bot re-reads the file, re-applies the edit and retries (3 times).
- **No duplicate commits.** Unchanged content produces no commit.
- **Prompt-injection aware.** File contents are sent to the AI as data, and every AI plan goes through the same
  validation and confirmation as a typed command.
- **Errors** are translated into safe messages (bad token, no permission, rate limit, protected branch, not found).
- Pushing to a protected default branch fails with a clear message; set `BOT_BRANCH` to a feature branch (it must already exist).

## Tests

```
python -m pytest bot1/tests -q
```
Covers security rules, the offline parser, create/update/delete, confirmations, conflicts, the Claude engine (mocked),
and the real GitHub client against a local fake GitHub API.

## Limits

Text files up to 1 MB. Binary files can be deleted but not edited. Large repos: the tree listing may be truncated by GitHub.
Offline mode understands only simple commands; use an API key for open-ended ones like "change the database connection code".

## Automatic mode (no command line): GitHub Issues

Once the bot files are in your repo, just **open an issue** titled `bot: <your command>`, for example
`bot: create a login.java file`. GitHub Actions runs the bot, commits the change, replies on the issue and closes it.

- Risky actions (deleting important or several files) get a comment asking you to reply `confirm`.
- Only the repo owner and collaborators can trigger it (checked in the workflow and again in Python).
- Needs no token of yours: the workflow uses GitHub's built-in token. Add `ANTHROPIC_API_KEY` as a repo secret for free-form commands.
- One-time setup: `python3 -m bot1 --install` uploads the bot files to the repo (needs a token with Contents + Workflows: Read and write).
