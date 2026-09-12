#!/usr/bin/env bash
# Punkt 15: Skinny-Build pro Architektur. RTX-Paar (sm75-Build) und V100-Paar
# (sm70-Build), DFlash2 27B mit maurienne-Kopf, danach cuobjdump der sm75-.so.
set -uo pipefail
D=/home/mp/.cache/huggingface/hub/models--maurienne-ai--Qwen3.8-27B-DFlash2-NVFP4-RTNcal/snapshots/bd7a934213c47a9e7ef69eef36bb3325f47fd1f1
cd /home/mp/Projekte/vllm-research/v100-skinny
for P in "0,2 p15_rtx" "1,3 p15_v100"; do
  set -- $P; DEVS=$1; NAME=$2
  for i in $(seq 1 60); do
    u=$(nvidia-smi --id=$DEVS --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)
    if [ "${u:-0}" -le 500 ] && ! pgrep -f '[a]pi_server' > /dev/null; then break; fi; sleep 5
  done
  DEVS=$DEVS DRAFT=$D bash tools/mtp-diagnostics/speed_dflash.sh "$NAME" fork dflash > "$HOME/.cache/mtp-diagnostics/sweep_$NAME.log" 2>&1
  R=$HOME/.cache/mtp-diagnostics/qual_$NAME
  echo "$NAME: $(grep -o 'STATUS up_[0-9]*s\|STATUS [a-z]*' $HOME/.cache/mtp-diagnostics/sweep_$NAME.log | head -1) | $(grep -o 'MEDIAN.*' $HOME/.cache/mtp-diagnostics/sweep_$NAME.log) | SHA $(python3 -c "import json;print(json.load(open('$R/result.json'))['sha256'])" 2>/dev/null || echo KEIN_ERGEBNIS)"
  echo "   kernel: $(grep -a -o -m1 'Skinny NVFP4 kernel for sm_[0-9]* loaded\|SM70 skinny NVFP4 kernel loaded' $R/boot.log)"
done
for so in $(find $HOME/.cache/torch_extensions -maxdepth 2 -name "skinny_nvfp4_v11_sm*" -type d); do
  echo "cuobjdump $(basename $so): $(cuobjdump -lelf $so/*.so 2>/dev/null | grep -o 'sm_[0-9]*' | sort | uniq -c | tr '\n' ' ')"
done
echo P15-ENDE
