#!/bin/sh
# The state HANDOFF.md claims, read from git and CI; run it at the start of a session (ORCHESTRATOR.md §0).
cd "$(git rev-parse --show-toplevel)" || exit 1
git fetch -q origin 2>/dev/null
echo "main checkout: $(git branch --show-current) @ $(git rev-parse --short HEAD); changes: $(git status --short | wc -l | tr -d ' ')"
for b in turbotab-next main; do
  echo "$b $(git rev-parse --short "$b") | origin $(git rev-parse --short "origin/$b" 2>/dev/null) | ahead behind: $(git rev-list --left-right --count "$b...origin/$b" 2>/dev/null)"
done
echo "branches not in turbotab-next, newest first:"
git for-each-ref --sort=-committerdate --no-merged turbotab-next --format='  %(refname:short) %(committerdate:short)' refs/heads | head -8
echo "worktrees: $(git worktree list | wc -l | tr -d ' ')"
echo "CI on turbotab-next:"
curl -s "https://api.github.com/repos/hedglinnolan/tabular-ml-lab/commits/$(git rev-parse turbotab-next)/check-runs?per_page=100" |
  python3 -c "import json,sys; [print('  %-11s %-8s %s' % (r['status'], r['conclusion'], r['name'])) for r in json.load(sys.stdin).get('check_runs', [])]"
