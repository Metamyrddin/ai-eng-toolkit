#!/usr/bin/env bash
# This script puts the toolkit on GitHub in one command, after checking that nothing dangerous or huge is included.
#
# How it fits in the project:
#   The .gitignore file already keeps the virtual environment, model caches and secret files out of Git.
#   This script adds two safety checks on top (no file larger than 50 MB, no secret-looking text),
#   makes a Git commit, and then uses the GitHub command line ("gh") to create the repository and upload the code.
#
# Usage:   bash tools/publish_github.sh <repository-name> [--private]
# Example: bash tools/publish_github.sh ai-eng-toolkit
#
# Before the first run you must log in once with:  gh auth login   (the GitHub CLI is free: https://cli.github.com)
# Nothing is uploaded until all checks pass. Use --private if you do not want the repository to be public yet
# (the free GPU notebooks in tools/gpu need a PUBLIC repository, so make it public before using them).
#
# STATUS: the checks (steps 1 and 2) were run on this repository. The upload (step 4) has NOT been run.
set -euo pipefail
cd "$(dirname "$0")/.."                                  # go to the repository root

NAME="${1:?Please give a repository name, for example ai-eng-toolkit}"
VISIBILITY="--public"; [ "${2:-}" = "--private" ] && VISIBILITY="--private"

# Step 1: make sure Git knows which files belong in the repository, and refuse files over 50 MB.
[ -d .git ] || git init -q
git add -A .
TOO_BIG="$(git ls-files -z | xargs -0 du -b | awk '$1 > 52428800' || true)"   # du prints "bytes filename"
if [ -n "$TOO_BIG" ]; then
  echo "STOP: these files are larger than 50 MB and must not be committed:"; echo "$TOO_BIG"; exit 1
fi

# Step 2: look for text that looks like a secret (GitHub, Hugging Face, Kaggle, OpenAI-style keys, private keys).
if git ls-files -z | xargs -0 grep -nIE "hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|KGAT_[A-Za-z0-9]+|BEGIN [A-Z ]*PRIVATE KEY"; then
  echo "STOP: the lines above look like secrets. Remove them first."; exit 1
fi
echo "Checks passed: $(git ls-files | wc -l) files, $(git ls-files -z | xargs -0 du -cb | tail -1 | cut -f1) bytes in total."

# Step 3: create a commit (this only changes your own computer).
git -c user.name="${GIT_AUTHOR_NAME:-ai-eng-toolkit}" -c user.email="${GIT_AUTHOR_EMAIL:-noreply@example.com}" \
    commit -q -m "AI engineering toolkit: eight areas with laptop and cloud-GPU paths" || echo "Nothing new to commit."
git branch -M main

# Step 4: create the GitHub repository and upload (this is the only step that talks to GitHub).
command -v gh >/dev/null || { echo "The GitHub CLI 'gh' is not installed: https://cli.github.com"; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "You are not logged in. Run: gh auth login"; exit 1; }
gh repo create "$NAME" $VISIBILITY --source . --remote origin --push
echo "Done. Your repository is at: $(gh repo view --json url -q .url)"
