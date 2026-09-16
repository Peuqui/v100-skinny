#!/usr/bin/env bash
# Paket 2, Abnahme-Kette: zwei Kaskaden-Boots mit verschiedenen Host-Anteilen,
# danach der Kontroll-Boot mit unveraendertem llama-swap-Eintrag.
# Laeuft abgekoppelt (setsid nohup); Ende markiert p2_driver.DONE.
set -uo pipefail
HERE=$(dirname "$(readlink -f "$0")")
cd /tmp
SWAP_MODEL=Qwen3.8-Flash-Next-180B-A4B-NVFP4-MTP-vllm-vlm-qwen3vl4b
rm -f "$HERE/p2_driver.DONE"

run() {  # $1=outdir $2=cascade-env $3=skip_control
  rm -rf "$HERE/$1"
  echo "=== $1 start $(date +%H:%M:%S)"
  CASCADE_ENV="$2" timeout 3000 bash "$HERE/ple_cascade_boot.sh" \
    "$HERE/$1" "$SWAP_MODEL" ref_full.json "$3" > "$HERE/$1.run.log" 2>&1
  echo "=== $1 ende $(date +%H:%M:%S) rc=$?"
}

run p2_host2 "VLLM_QWEN4EXP_PLE_STORE_DEVICE=4 VLLM_QWEN4EXP_PLE_STORE_GIB=16 VLLM_QWEN4EXP_PLE_HOST_GIB=2" 1
run p2_host0 "VLLM_QWEN4EXP_PLE_STORE_DEVICE=4 VLLM_QWEN4EXP_PLE_STORE_GIB=16 VLLM_QWEN4EXP_PLE_HOST_GIB=0" 0
date > "$HERE/p2_driver.DONE"
echo "FERTIG $(date +%H:%M:%S)"
