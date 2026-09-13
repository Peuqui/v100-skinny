#!/usr/bin/env bash
# Tail-Cudagraphs auf Turing im FORK (work-main = main + unsere offenen PRs, DFlash2 laeuft dort auf der RTX):
# Patch als Overlay, dann RTX Produktionskopf: Tail an (Default) gegen Tail aus, plus V100-Gegenprobe.
set -uo pipefail
cd /home/mp/Projekte/vllm-research/v100-skinny
for i in $(seq 1 240); do grep -qE "TAILGRAPHS-ENDE|ABBRUCH" handover/2026-09-12/tailgraphs_chain2.out 2>/dev/null && break; sleep 30; done
W=/home/mp/Projekte/vllm-research/1Cat-vLLM-work
python3 /tmp/claude-1000/-home-mp-Projekte-AIfred-Intelligence/a3963e2c-ddcd-4607-85e4-f3d182a6897c/scratchpad/apply_tailgraphs_patch.py $W
(cd $W && git diff --stat | tail -1)
export DRAFT=$(ls -d /home/mp/.cache/huggingface/hub/models--maurienne-ai--Qwen3.8-27B-DFlash2-NVFP4-RTNcal/snapshots/*/ | head -1)
export VENV=/home/mp/vllm/venv
S=tools/mtp-diagnostics/speed_dflash.sh
wait_free() { for i in $(seq 1 120); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc); [ "${u:-0}" -le 1000 ] && ! pgrep -f "^bash tools/mtp-diagnostics/speed_dflash.sh" >/dev/null && return 0; sleep 5; done; echo "ABBRUCH: Karten nicht frei"; exit 1; }
show() { echo "   sha: $(grep -h -o '"sha256": "[0-9a-f]*"' ~/.cache/mtp-diagnostics/qual_$1/result.json 2>/dev/null) | tail: $(grep -a -c 'Capturing SM70 DFlash2 target tail' ~/.cache/mtp-diagnostics/qual_$1/boot.log) | fa2: $(grep -a -o -m1 'Loaded FA2 library [^ ]*' ~/.cache/mtp-diagnostics/qual_$1/boot.log)"; }
for P in "tf_rtx_tail1 0,2 1" "tf_rtx_tail0 0,2 0" "tf_rtx_tail1b 0,2 1" "tf_v100_tail1 1,3 1"; do
  set -- $P; wait_free; echo "== $1 (DEVS $2, TAIL=$3, Produktionskopf, Fork gepatcht)"
  DEVS=$2 VLLM_SM70_DFLASH2_TAIL_CUDAGRAPHS=$3 bash $S $1 fork dflash 2>&1 | grep -E "STATUS|MEDIAN|Error"; show $1
done
echo TAILFORK-ENDE
