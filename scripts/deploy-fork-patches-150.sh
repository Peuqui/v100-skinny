#!/usr/bin/env bash
# Deploy fork_patches_150/ over the installed 1Cat-vLLM 1.5.0 wheel
# (.venv-sm70-150 by default). Pairs come from DEPLOY-TARGETS.txt; entries
# whose tracked file is absent were retired upstream and are skipped.
# Keeps a .pre_deploy backup of every replaced file.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-$REPO_ROOT/.venv-sm70-150}"
PY="$ENV_PREFIX/bin/python"
SP="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
[ -d "$SP/vllm" ] || { echo "vllm not found in $SP" >&2; exit 1; }
PATCHES="$REPO_ROOT/fork_patches_150"
deployed=0; skipped=0
while read -r src dst; do
  [ -n "$src" ] || continue
  if [ ! -f "$PATCHES/$src" ]; then skipped=$((skipped+1)); continue; fi
  if [ -f "$SP/$dst" ]; then
    [ -f "$SP/$dst.pre_deploy" ] || cp -p "$SP/$dst" "$SP/$dst.pre_deploy"
  else
    # fork-own file that does not exist in the wheel: create it
    mkdir -p "$(dirname "$SP/$dst")"
  fi
  cp -p "$PATCHES/$src" "$SP/$dst"
  deployed=$((deployed+1))
done < "$PATCHES/DEPLOY-TARGETS.txt"
echo "deployed: $deployed, retired/skipped: $skipped (site-packages: $SP)"
