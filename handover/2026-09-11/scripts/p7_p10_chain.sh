#!/usr/bin/env bash
# (1) Punkt 7: DeepSeek PP5 + 27B-GDN mit tilelang 0.1.14 (.venv-sm70-tltest)
# (2) Punkt 10: echter Tool-Call gegen den DeepSeek-Eintrag von llama-swap (Prod-venv)
set -uo pipefail
REPO=/home/mp/Projekte/vllm-research/v100-skinny
TL=$REPO/.venv-sm70-tltest
D=/home/mp/.cache/huggingface/hub/models--maurienne-ai--Qwen3.8-27B-DFlash2-NVFP4-RTNcal/snapshots/bd7a934213c47a9e7ef69eef36bb3325f47fd1f1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_HOME=/home/mp/vllm/cuda PATH=/home/mp/vllm/cuda/bin:$PATH
cd $REPO
echo "== DS-ACCEPT tl014"
VENV=$TL bash handover/2026-09-11/scripts/ds_accept.sh tl014 2>&1 | grep -a -E "STATUS|tok/s|Kohaerenz|coheren|byteident|identisch|FERTIG|Traceback|Error" | head -12
echo "== 27B-GDN tl014 (RTX-Paar, DFlash2)"
for i in $(seq 1 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc); [ "${u:-0}" -le 1000 ] && ! pgrep -f '[a]pi_server' >/dev/null && break; sleep 5; done
VENV=$TL DEVS=0,2 DRAFT=$D bash tools/mtp-diagnostics/speed_dflash.sh tl014_27b fork dflash > $HOME/.cache/mtp-diagnostics/sweep_tl014_27b.log 2>&1
R=$HOME/.cache/mtp-diagnostics/qual_tl014_27b
echo "27B: $(grep -o 'STATUS up_[0-9]*s\|STATUS [a-z]*' $HOME/.cache/mtp-diagnostics/sweep_tl014_27b.log | head -1) | $(grep -o 'MEDIAN.*' $HOME/.cache/mtp-diagnostics/sweep_tl014_27b.log) | SHA $(python3 -c "import json;print(json.load(open('$R/result.json'))['sha256'])" 2>/dev/null || echo KEIN_ERGEBNIS)"
echo "   tilelang: $(grep -a -o -m1 'tilelang [0-9.]*\|TileLang[^ ]* [0-9.]*' $R/boot.log)"
echo "== TOOLCALL DeepSeek via llama-swap"
for i in $(seq 1 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc); [ "${u:-0}" -le 1000 ] && ! pgrep -f '[a]pi_server' >/dev/null && break; sleep 5; done
/home/mp/vllm/venv/bin/python handover/2026-09-11/scripts/abnahme2/toolcall_probe.py 11435 DeepSeek-V4-Flash-nvfp4-DSpark-vllm handover/2026-09-11/ergebnisse/toolcall_deepseek 2>&1 | tail -12
echo CHAIN-ENDE
