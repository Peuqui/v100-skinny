#!/bin/bash
# pre-commit over the PR diff (origin/main..HEAD) in the GDN gate PR worktree, as the 1Cat development guide asks.
cd /home/mp/Projekte/vllm-research/1Cat-vLLM-pr-gdn || exit 1
/home/mp/.venv/precommit/bin/pre-commit run --from-ref origin/main --to-ref HEAD
echo "pre-commit rc=$?"
