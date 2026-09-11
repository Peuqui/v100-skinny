#!/usr/bin/env bash
# Abnahme 11.09. spaet: E5-Ausbau (V1-Runner: 27B-MTP, DeepSeek) und Befund 2
# index_share=True (Flash-Next, V2), DFlash2-V100 als Sicherheitsnetz. GPU-seriell.
# Aufruf: driver2.sh <OUT>
set -uo pipefail
S=/home/mp/Projekte/v100-skinny/handover/2026-09-11/scripts
REPO=/home/mp/Projekte/vllm-research/v100-skinny
OUT=${1:?OUT fehlt}; mkdir -p "$OUT"
VENV=/home/mp/Projekte/vllm-research/v100-skinny/.venv-sm70-main
D=/home/mp/.cache/huggingface/hub/models--maurienne-ai--Qwen3.8-27B-DFlash2-NVFP4-RTNcal/snapshots/bd7a934213c47a9e7ef69eef36bb3325f47fd1f1
stamp() { echo "===== $(date '+%F %T') $*"; }
sha_of() { grep -o '"sha256": "[0-9a-f]*"' "$HOME/.cache/mtp-diagnostics/qual_$1/result.json" | sed "s/^/$1 /"; }

stamp "A) 27B-MTP (V1-Runner, E5 entfernt), exakter Produktionsbefehl"
bash "$S/prod_accept_p1.sh" "$OUT/prod_e5out"
grep -c "e5-" "$OUT/prod_e5out/Qwen3.8-27B-NVFP4-vllm.boot.log" | sed 's/^/  e5-Zeilen im Boot-Log (muss 0 sein): /'
"$VENV/bin/python" - "$S/prod/Qwen3.8-27B-NVFP4-vllm.p1.json" "$OUT/prod_e5out/Qwen3.8-27B-NVFP4-vllm.p1.json" <<'PY'
import json, sys
try:
    a, b = (json.load(open(p))["choices"][0]["message"]["content"] for p in sys.argv[1:3])
except Exception as e:
    print(f"27B-MTP: VERGLEICH NICHT MOEGLICH: {e}"); sys.exit(0)
print("27B-MTP: Text gleich der Referenz" if a == b else f"27B-MTP: ABWEICHUNG\nREF: {a[:300]}\nNEU: {b[:300]}")
PY

cd "$REPO"
stamp "B) 27B DFlash2 V100 (V2-Runner, Sicherheitsnetz speculative.py)"
DEVS=1,3 DRAFT=$D bash tools/mtp-diagnostics/speed_dflash.sh acc3_v100 fork dflash 2>&1 | tail -2; sha_of acc3_v100

stamp "C) Flash-Next heterogen, index_share=True (Befund 2), 3 Laeufe Rohtext-Sonde"
for r in 1 2 3; do
  stamp "C.$r"
  VENV=$VENV bash tools/mtp-diagnostics/flashnext_qual.sh acc3_ishare_$r 4 2>&1 | grep "q[123]:\|UP on\|FERTIG\|Traceback\|Error"
done

stamp "D) DeepSeek PP5 (V1-Runner, E5 entfernt)"
VENV=$VENV bash "$S/ds_accept.sh" acc3_e5out > "$OUT/ds_acc3.out" 2>&1
norm() { grep "^\[" "$1" | sed -E 's/ [0-9]+ tok in [0-9.]+s \([0-9.]+ tok\/s\): / /'; }
if diff <(norm "$S/ds_main.out") <(norm "$OUT/ds_acc3.out") > /dev/null; then
  echo "DeepSeek: alle Antworten gleich der Referenz"
else
  echo "DeepSeek: ABWEICHUNG"; diff <(norm "$S/ds_main.out") <(norm "$OUT/ds_acc3.out") | head -10
fi
grep "STATUS\|tok/s" "$OUT/ds_acc3.out" | head -12
stamp "ABNAHME3-ENDE"
