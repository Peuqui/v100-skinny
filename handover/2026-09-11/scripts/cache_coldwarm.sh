#!/usr/bin/env bash
# Beleg fuer den Compile-Cache-PR: 0DOT3-Graph + aktiver Cache, frischer
# Cache-Root, erst Kaltstart, dann Warmstart. Erwartung: Warmstart laedt die
# AOT-Artefakte ("Directly load"), Text-SHA und Annahmelaenge unveraendert.
#   cache_coldwarm.sh <DEVS> <tag>
set -uo pipefail
DEVS=$1; TAG=$2
D=/home/mp/.cache/huggingface/hub/models--maurienne-ai--Qwen3.8-27B-DFlash2-NVFP4-RTNcal/snapshots/bd7a934213c47a9e7ef69eef36bb3325f47fd1f1
ROOT=$HOME/.cache/vllm-cache-coldwarm-$TAG; rm -rf "$ROOT"
cd /home/mp/Projekte/vllm-research/v100-skinny
for PHASE in cold warm; do
  NAME="cache_${TAG}_$PHASE"
  for i in $(seq 1 60); do
    u=$(nvidia-smi --id=$DEVS --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)
    if [ "${u:-0}" -le 500 ] && ! pgrep -f '[a]pi_server' > /dev/null; then break; fi
    sleep 5
  done
  VLLM_DISABLE_COMPILE_CACHE=0 VLLM_CACHE_ROOT=$ROOT DEVS=$DEVS DRAFT=$D \
    bash tools/mtp-diagnostics/speed_dflash.sh "$NAME" fork dflash > "$HOME/.cache/mtp-diagnostics/sweep_$NAME.log" 2>&1
  R=$HOME/.cache/mtp-diagnostics/qual_$NAME
  echo "$PHASE: $(grep -o 'STATUS up_[0-9]*s\|STATUS [a-z]*' $HOME/.cache/mtp-diagnostics/sweep_$NAME.log | head -1) | $(grep -o 'MEDIAN.*' $HOME/.cache/mtp-diagnostics/sweep_$NAME.log) | SHA $(python3 -c "import json;print(json.load(open('$R/result.json'))['sha256'])" 2>/dev/null || echo KEIN_ERGEBNIS)"
  echo "   cache: $(grep -o -m1 'Using cache directory: [^ ]*\|compile cache is disabled' $R/boot.log) | $(grep -c 'Directly load' $R/boot.log) x Directly-load | $(grep -o -m1 'torch.compile takes [0-9.]* s' $R/boot.log)"
done
echo COLDWARM-ENDE
