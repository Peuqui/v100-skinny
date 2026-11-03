#!/bin/bash
# Which pre-commit hooks failed in the CI run of a PR? Usage: ci_failed_hooks.sh <run id>
RUN=$1
OUT=/tmp/claude-1000/-home-mp-Projekte-vllm-research/f6c0fa1a-4e12-435e-a578-839446e867c6/scratchpad/ci_log_$RUN.txt
gh run view "$RUN" --repo 1CatAI/1Cat-vLLM --log-failed > "$OUT" 2>&1
echo "Logzeilen: $(wc -l < "$OUT")"
grep -nE "\.\.\.\.Failed|Failed$|hook id:|platform coupling|exit code" "$OUT" | head -40
