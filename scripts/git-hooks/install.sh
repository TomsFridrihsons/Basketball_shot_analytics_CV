#!/bin/sh
# Copy the HPAI ticket hooks into .git/hooks (keeps the Git LFS hooks untouched).
set -e
ROOT=$(git rev-parse --show-toplevel)
HOOKS=$(git rev-parse --git-path hooks)
for hook in prepare-commit-msg commit-msg; do
  cp "$ROOT/scripts/git-hooks/$hook" "$HOOKS/$hook"
  chmod +x "$HOOKS/$hook"
  echo "installed $hook"
done
