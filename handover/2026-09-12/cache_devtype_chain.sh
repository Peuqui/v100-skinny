#!/usr/bin/env bash
# Kartentyp-Test Compile-Cache: Artefakt auf V100 erzeugen, mit identischer Umgebung auf RTX booten. Laedt die RTX das V100-Artefakt?
set -uo pipefail
cd /home/mp/Projekte/vllm-research/v100-skinny
R=/home/mp/.cache/vllm-devtype-test; rm -rf $R; mkdir -p $R
export VENV=/home/mp/vllm/venv VLLM_DISABLE_COMPILE_CACHE=0
S=tools/mtp-diagnostics/speed_dflash.sh
wait_free() { for i in $(seq 1 120); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc); [ "${u:-0}" -le 1000 ] && ! pgrep -f "^bash tools/mtp-diagnostics/speed_dflash.sh" >/dev/null && return 0; sleep 5; done; echo "ABBRUCH: Karten nicht frei"; exit 1; }
show() { B=~/.cache/mtp-diagnostics/qual_$1/boot.log; echo "   sha: $(grep -h -o '"sha256": "[0-9a-f]*"' ~/.cache/mtp-diagnostics/qual_$1/result.json 2>/dev/null) | aot-load: $(grep -a -c 'Directly load AOT' $B) | compiling: $(grep -a -c 'Compiling a graph' $B) | cache-off: $(grep -a -c 'compile cache is disabled' $B) | artefakte: $(ls $R/torch_compile_cache/torch_aot_compile 2>/dev/null | wc -l)"; grep -a -o -m1 'torch_aot_compile/[0-9a-f]*' $B | head -1; }
wait_free; echo "== 1 V100-Paar, frischer Root (kompiliert, schreibt Artefakt)"
VLLM_CACHE_ROOT=$R DEVS=1,3 bash $S dt_v100_cold fork dflash 2>&1 | grep -E "STATUS|MEDIAN|Error"; show dt_v100_cold
wait_free; echo "== 2 RTX-Paar, DERSELBE Root, identische Umgebung (laedt es das V100-Artefakt?)"
VLLM_CACHE_ROOT=$R DEVS=0,2 bash $S dt_rtx_after_v100 fork dflash 2>&1 | grep -E "STATUS|MEDIAN|Error"; show dt_rtx_after_v100
wait_free; echo "== 3 RTX-Paar, frischer Root (Kontrolle kalt)"
rm -rf $R-rtx; mkdir -p $R-rtx
VLLM_CACHE_ROOT=$R-rtx DEVS=0,2 bash $S dt_rtx_cold fork dflash 2>&1 | grep -E "STATUS|MEDIAN|Error"; R=$R-rtx show dt_rtx_cold
wait_free; echo "== 4 V100-Paar, Root aus 1 erneut (Kontrolle: laedt V100 sein eigenes Artefakt?)"
VLLM_CACHE_ROOT=$R DEVS=1,3 bash $S dt_v100_warm fork dflash 2>&1 | grep -E "STATUS|MEDIAN|Error"; show dt_v100_warm
echo DEVTYPE-ENDE
