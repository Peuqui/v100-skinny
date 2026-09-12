#!/usr/bin/env bash
cd /home/mp/Projekte/vllm-research/v100-skinny
wait_free() { for i in $(seq 1 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc); [ "${u:-0}" -le 1000 ] && ! pgrep -f '^[^ ]*python[^ ]* -m vllm.entrypoints.openai.api_server' >/dev/null && return 0; sleep 5; done; echo "ABBRUCH: Karten nicht frei"; exit 1; }
wait_free; GPUS=1,3 bash handover/2026-09-12/kvpolicy_probe.sh v100_patch /home/mp/Projekte/vllm-research/1Cat-vLLM-pr-kvpolicy
wait_free; GPUS=1,3 bash handover/2026-09-12/kvpolicy_probe.sh v100_control /home/mp/Projekte/vllm-research/1Cat-vLLM-pr-fa2sm75
echo KVPOLICY-ENDE
