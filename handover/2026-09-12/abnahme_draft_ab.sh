#!/usr/bin/env bash
# Nachlauf 4: DFlash2 RTX mit dem QUANTISIERTEN Entwurfskopf (maurienne RTNcal), wie die 77-tok/s-Laeufe vom 11.09.; neu und alt.
set -uo pipefail
cd /home/mp/Projekte/vllm-research/v100-skinny
for i in $(seq 1 720); do grep -q "PROD-ENDE" handover/2026-09-12/abnahme_prod_rerun.out 2>/dev/null && break; sleep 30; done
export DRAFT=$(ls -d /home/mp/.cache/huggingface/hub/models--maurienne-ai--Qwen3.8-27B-DFlash2-NVFP4-RTNcal/snapshots/*/ | head -1)
S=tools/mtp-diagnostics/speed_dflash.sh
wait_free() { for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc); [ "${u:-0}" -le 1000 ] && return 0; sleep 5; done; echo "ABBRUCH: Karten nicht frei"; exit 1; }
for P in "rb_rtx_qdraft_new /home/mp/vllm/venv 0,2" "rb_rtx_qdraft_old /home/mp/Projekte/vllm-research/v100-skinny/.venv-sm70-old 0,2" "rb_v100_qdraft_new /home/mp/vllm/venv 1,3"; do
  set -- $P; wait_free; echo "== $1 (venv $2, DEVS $3, DRAFT quantisiert)"
  VENV=$2 DEVS=$3 bash $S $1 fork dflash 2>&1 | grep -E "STATUS|MEDIAN|Error|ABBRUCH"
  echo "   sha: $(grep -h -o '"sha256": "[0-9a-f]*"' ~/.cache/mtp-diagnostics/qual_$1/result.json 2>/dev/null)"
done
echo DRAFTAB-ENDE
