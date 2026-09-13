#!/usr/bin/env bash
# Reproduziert der AOT-Ladefehler? (a) warm ohne Zwang auf dem bestehenden Root, (b) frischer Root kalt, (c) danach warm ohne Zwang.
set -uo pipefail
cd /home/mp/Projekte/vllm-research/v100-skinny
export VENV=/home/mp/vllm/venv VLLM_DISABLE_COMPILE_CACHE=0 DEVS=1,3
S=tools/mtp-diagnostics/speed_dflash.sh
wait_free() { for i in $(seq 1 120); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc); [ "${u:-0}" -le 1000 ] && ! pgrep -f "^bash tools/mtp-diagnostics/speed_dflash.sh" >/dev/null && return 0; sleep 5; done; echo "ABBRUCH"; exit 1; }
show() { B=~/.cache/mtp-diagnostics/qual_$1/boot.log; echo "   sha: $(grep -h -o '"sha256": "[0-9a-f]*"' ~/.cache/mtp-diagnostics/qual_$1/result.json 2>/dev/null) | aot-load: $(grep -a -c 'Directly load AOT' $B) | load-failure: $(grep -a -c 'load failure' $B) | saved: $(grep -a -c 'saved AOT' $B)"; grep -a -o -m2 "load failure from [^ ]* reason: .*" $B | cut -c1-200; }
wait_free; echo "== a warm ohne Zwang, bestehender Root"
VLLM_CACHE_ROOT=/home/mp/.cache/vllm-devtype-test bash $S rl_warm_a fork dflash 2>&1 | grep -E "STATUS|MEDIAN|Error"; show rl_warm_a
R=/home/mp/.cache/vllm-reload-test; rm -rf $R; mkdir -p $R
wait_free; echo "== b frischer Root kalt"
VLLM_CACHE_ROOT=$R bash $S rl_cold_b fork dflash 2>&1 | grep -E "STATUS|MEDIAN|Error"; show rl_cold_b
wait_free; echo "== c derselbe Root warm ohne Zwang"
VLLM_CACHE_ROOT=$R bash $S rl_warm_c fork dflash 2>&1 | grep -E "STATUS|MEDIAN|Error"; show rl_warm_c
wait_free; echo "== d noch einmal warm"
VLLM_CACHE_ROOT=$R bash $S rl_warm_d fork dflash 2>&1 | grep -E "STATUS|MEDIAN|Error"; show rl_warm_d
echo RELOAD-ENDE
