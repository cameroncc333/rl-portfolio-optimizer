#!/bin/bash
# Runs RL inference and pushes fresh signal to sector-command-live repo.
# Scheduled daily at 8:30 AM via launchd (see com.sectorcommand.rl.plist).

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$HOME/Downloads/sector-command-live-2"
SIGNAL_OUT="$REPO_DIR/data/rl_signal.json"
VENV="$SCRIPT_DIR/rl_env/bin/python3"
LOG="$SCRIPT_DIR/logs/auto_signal.log"

mkdir -p "$SCRIPT_DIR/logs"
exec >> "$LOG" 2>&1
echo ""
echo "=== $(date) ==="

# Run RL inference
echo "[1/3] Running RL inference..."
RL_SIGNAL_OUT="$SIGNAL_OUT" "$VENV" "$SCRIPT_DIR/generate_rl_signal.py"

# Commit and push
echo "[2/3] Pushing to GitHub..."
cd "$REPO_DIR"
git add data/rl_signal.json
git diff --cached --quiet && echo "No change in signal — skip commit" && exit 0
git commit -m "rl: auto-signal $(date -u '+%Y-%m-%d %H:%M UTC')"
git push

echo "[3/3] Done — fresh signal pushed."
