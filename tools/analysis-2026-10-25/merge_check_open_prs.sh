#!/bin/bash
# Does each of our open 1Cat PRs still merge cleanly into the current origin/main? Uses git merge-tree (no checkout) in the PR worktree.
# Output: one line per PR: number, head ref, result (clean or conflicting files).
WT=/home/mp/Projekte/vllm-research/1Cat-vLLM-pr-sched
OUT=/tmp/claude-1000/-home-mp-Projekte-vllm-research/f6c0fa1a-4e12-435e-a578-839446e867c6/scratchpad/merge_check_open_prs.txt
git -C "$WT" fetch origin
echo "origin/main = $(git -C "$WT" rev-parse --short=9 origin/main)" > "$OUT"
gh pr list --repo 1CatAI/1Cat-vLLM --author Peuqui --state open --limit 100 --json number,headRefName --jq '.[] | "\(.number) \(.headRefName)"' | sort -n > "$OUT.list"
while read -r number ref; do
  git -C "$WT" fetch --quiet origin "pull/$number/head:refs/prcheck/$number" 2>/dev/null || git -C "$WT" fetch --quiet --force origin "pull/$number/head:refs/prcheck/$number"
  result=$(git -C "$WT" merge-tree --write-tree --name-only origin/main "refs/prcheck/$number" 2>&1)
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "#$number $ref: sauber" >> "$OUT"
  else
    files=$(echo "$result" | grep -vE '^[0-9a-f]{40}$|^$|^Auto-merging|^CONFLICT' | tr '\n' ' ')
    echo "#$number $ref: KONFLIKT rc=$rc Dateien: $files" >> "$OUT"
  fi
done < "$OUT.list"
cat "$OUT"
