#!/usr/bin/env bash
set -uo pipefail
cd /home/mp/Projekte/vllm-research/v100-skinny
wait_free() { for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc); [ "${u:-0}" -le 1000 ] && ! pgrep -f '^[^ ]*python[^ ]* -m vllm.entrypoints.openai.api_server' >/dev/null && return 0; sleep 5; done; echo "ABBRUCH: Karten nicht frei"; exit 1; }
for P in "fa2_fp8 FLASH_ATTN auto" "tri_fp16 TRITON_ATTN float16" "fa2_fp16 FLASH_ATTN float16"; do
  set -- $P; wait_free; echo "== $1"; bash handover/2026-09-12/fa2_probe.sh $1 $2 $3 2>&1 | grep -E "STATUS|ERGEBNIS|attn:|Error|FERTIG"
done
echo FA2-ENDE
