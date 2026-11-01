#!/bin/bash
# Double-click me. Anything you drop into the "BotDrop" folder is pushed into your GitHub repo,
# placed in the right folder automatically.
cd "$(dirname "$0")" || exit 1
export PYTHONPATH="$(pwd)/../1-UPLOAD-THESE-TO-GITHUB"
ENVFILE="$HOME/.bot2.env"
DROP="$HOME/BotDrop"

echo "=== Bot 2: drop-folder pusher ==="
if ! python3 -c "import github, dotenv" 2>/dev/null; then
  echo "Installing helper packages (one time)..."
  python3 -m pip install --quiet PyGithub python-dotenv anthropic 2>/dev/null \
    || python3 -m pip install --quiet --user --break-system-packages PyGithub python-dotenv anthropic
fi

if [ ! -f "$ENVFILE" ]; then
  echo "One-time setup. Saved only on this Mac, in $ENVFILE (private file)."
  read -r -p "Your repo (looks like username/repo): " REPO
  read -r -s -p "Paste your GitHub token (hidden while typing): " TOKEN; echo
  ( umask 077; printf 'GITHUB_REPO=%s\nGITHUB_TOKEN=%s\n' "$REPO" "$TOKEN" > "$ENVFILE" )
fi

mkdir -p "$DROP"
command -v open >/dev/null && open "$DROP"
echo "Drop files into: $DROP   (Press Ctrl+C to stop)"
python3 -m bot2 watch "$DROP"
echo
echo "Stopped. To change repo/token later, delete $ENVFILE and run me again."
read -r -p "Press Enter to close." _
