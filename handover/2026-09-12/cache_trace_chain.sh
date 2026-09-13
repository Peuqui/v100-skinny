#!/usr/bin/env bash
# Traceback des AOT-Ladefehlers: frischer Root kalt, dann Warmstart mit VLLM_FORCE_AOT_LOAD=1 (re-raise).
set -uo pipefail
cd /home/mp/Projekte/vllm-research/v100-skinny
for i in $(seq 1 120); do grep -qE "RELOAD-ENDE|ABBRUCH" handover/2026-09-12/cache_reload_chain.out 2>/dev/null && break; sleep 20; done
export VENV=/home/mp/vllm/venv VLLM_DISABLE_COMPILE_CACHE=0 DEVS=1,3
S=tools/mtp-diagnostics/speed_dflash.sh
R=/home/mp/.cache/vllm-trace-test; rm -rf $R; mkdir -p $R
wait_free() { for i in $(seq 1 120); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc); [ "${u:-0}" -le 1000 ] && ! pgrep -f "^bash tools/mtp-diagnostics/speed_dflash.sh" >/dev/null && return 0; sleep 5; done; echo "ABBRUCH"; exit 1; }
wait_free; echo "== kalt"; VLLM_CACHE_ROOT=$R bash $S tr_cold fork dflash 2>&1 | grep -E "STATUS|MEDIAN"
wait_free; echo "== warm mit Zwang (erwartet Traceback)"; VLLM_CACHE_ROOT=$R VLLM_FORCE_AOT_LOAD=1 bash $S tr_force fork dflash 2>&1 | grep -E "STATUS|MEDIAN"
B=~/.cache/mtp-diagnostics/qual_tr_force/boot.log
echo "   aot-load: $(grep -a -c 'Directly load AOT' $B) | traceback: $(grep -a -c 'Traceback' $B)"
grep -a -A40 -m1 "Traceback (most recent call last)" $B | sed 's/^([^)]*) //; s/ERROR [0-9-]* [0-9:]* \[[a-z_.]*:[0-9]*\] //' | grep -a -E "Error|raise |File \"" | grep -a -v "^\s*$" | tail -14 | cut -c1-200
echo TRACE-ENDE
