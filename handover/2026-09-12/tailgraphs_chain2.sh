#!/usr/bin/env bash
# E2E Tail-Cudagraphs/MTP-Split auf Turing, incoai-Entwurfskopf (der maurienne-Kopf braucht PR #592, nicht in main).
set -uo pipefail
cd /home/mp/Projekte/vllm-research/v100-skinny
T=/home/mp/Projekte/vllm-research/1Cat-vLLM-pr-turing-ops
export KV_DTYPE=float16 VENV=/home/mp/Projekte/vllm-research/v100-skinny/.venv-pr-turing DEVS=0,2
unset DRAFT
S=tools/mtp-diagnostics/speed_dflash.sh
wait_free() { for i in $(seq 1 120); do u=$(nvidia-smi --id=0,2 --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc); [ "${u:-0}" -le 1000 ] && ! pgrep -f "^bash tools/mtp-diagnostics/speed_dflash.sh" >/dev/null && return 0; sleep 5; done; echo "ABBRUCH: Karten nicht frei"; exit 1; }
show() { echo "   sha: $(grep -h -o '"sha256": "[0-9a-f]*"' ~/.cache/mtp-diagnostics/qual_$1/result.json 2>/dev/null) | tail: $(grep -a -c 'Capturing SM70 DFlash2 target tail' ~/.cache/mtp-diagnostics/qual_$1/boot.log) | split: $(grep -a -c -i 'split.*draft.*graph\|SM70 MTP split' ~/.cache/mtp-diagnostics/qual_$1/boot.log) | fa2: $(grep -a -o -m1 'Loaded FA2 library [^ ]*' ~/.cache/mtp-diagnostics/qual_$1/boot.log)"; }
wait_free; echo "== basis mtp (aus Vorlauf, ungepatcht, SPLIT=1)"; grep -h "MEDIAN\|STATUS" handover/2026-09-12/tailgraphs_chain.out | tail -2; show tg_mtp_base_rtx
echo "== basis dflash incoai (ungepatcht)"
bash $S tg_base_rtx2 fork dflash 2>&1 | grep -E "STATUS|MEDIAN|Error"; show tg_base_rtx2
python3 /tmp/claude-1000/-home-mp-Projekte-AIfred-Intelligence/a3963e2c-ddcd-4607-85e4-f3d182a6897c/scratchpad/apply_tailgraphs_patch.py $T
wait_free; echo "== patch dflash (Tail-Graphen auf Turing)"
bash $S tg_patch_rtx fork dflash 2>&1 | grep -E "STATUS|MEDIAN|Error"; show tg_patch_rtx
wait_free; echo "== patch mtp k=3, SPLIT=1"
VLLM_SM70_MTP_SPLIT_DRAFT_CUDAGRAPHS=1 bash $S tg_mtp_patch_rtx fork mtp 2>&1 | grep -E "STATUS|MEDIAN|Error"; show tg_mtp_patch_rtx
wait_free; echo "== patch dflash Wiederholung"
bash $S tg_patch_rtx2 fork dflash 2>&1 | grep -E "STATUS|MEDIAN|Error"; show tg_patch_rtx2
echo TAILGRAPHS-ENDE
