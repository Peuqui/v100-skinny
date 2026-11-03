#!/bin/bash
# Compare the live body of a PR with the draft file: prints "IDENTISCH" or the diff.
PR=$1
DRAFT=$2
LIVE=/tmp/claude-1000/-home-mp-Projekte-vllm-research/f6c0fa1a-4e12-435e-a578-839446e867c6/scratchpad/live_body_$PR.txt
gh pr view "$PR" --repo 1CatAI/1Cat-vLLM --json body --jq .body > "$LIVE"
if diff <(tr -d '\r' < "$LIVE") "$DRAFT" > /dev/null; then
  echo IDENTISCH
else
  diff <(tr -d '\r' < "$LIVE") "$DRAFT"
fi
