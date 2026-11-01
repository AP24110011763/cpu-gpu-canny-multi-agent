# Bot 1 - repo-ready setup (no token, no terminal needed)

This zip is already laid out for a repository root:
  .github/workflows/   <- the automation
  bot1/                <- the bot
  .gitignore

1. Upload everything to the ROOT of your repo (see steps in chat), commit to main.
2. Repo -> Settings -> Actions -> General -> Workflow permissions -> "Read and write permissions" -> Save.
3. (Optional) Settings -> Secrets and variables -> Actions -> New secret: ANTHROPIC_API_KEY
4. Issues -> New issue, title:  bot: create a login.java file   -> Submit.
5. Watch the Actions tab. The bot commits the change, comments and closes the issue.
