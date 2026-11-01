# Inbox - drop files here

Upload any file into this `inbox/` folder (GitHub: **Add file -> Upload files -> Commit changes**).
Bot 2 will automatically move it to the right folder, commit it, and post an update in the
**"Repo Activity Log (Bot 2)"** issue.

- Want a specific place? Upload into a sub-folder, e.g. `inbox/docs/report.pdf` goes to `docs/report.pdf`.
- A file with the same name but different content is kept as `name_2.ext` (nothing is overwritten).
- Files that look like passwords/keys (`.env`, private keys, tokens) are deleted, never filed.
