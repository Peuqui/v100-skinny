#!/bin/bash
# Compare the live body of an issue with the draft file: prints IDENTISCH or the diff. Usage: compare_issue_body.sh <issue> <draft file>
ISSUE=$1
DRAFT=$2
LIVE=/tmp/claude-1000/-home-mp-Projekte-vllm-research/f6c0fa1a-4e12-435e-a578-839446e867c6/scratchpad/live_issue_$ISSUE.txt
gh issue view "$ISSUE" --repo 1CatAI/1Cat-vLLM --json body --jq .body > "$LIVE"
if diff <(tr -d '\r' < "$LIVE") "$DRAFT" > /dev/null; then
  echo IDENTISCH
else
  diff <(tr -d '\r' < "$LIVE") "$DRAFT"
fi
