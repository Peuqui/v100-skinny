#!/usr/bin/env bash
# Abnahme-Treiber 11.09. nachmittags: Rest der Aufraeum-Abnahme (Befunde 1,3-8),
# dazu A/B fuer DFlash2-V100 und Flash-Next gegen zurueckgenommenes Aufraeumen,
# am Ende DeepSeek PP5. GPU-seriell. Aufruf: driver.sh <OUT-Verzeichnis>
set -uo pipefail
S=/home/mp/Projekte/v100-skinny/handover/2026-09-11/scripts
REPO=/home/mp/Projekte/vllm-research/v100-skinny
WT=/home/mp/Projekte/vllm-research/1Cat-vLLM-work
OUT=${1:?OUT fehlt}; mkdir -p "$OUT"
VENV=/home/mp/Projekte/vllm-research/v100-skinny/.venv-sm70-main
D=/home/mp/.cache/huggingface/hub/models--maurienne-ai--Qwen3.8-27B-DFlash2-NVFP4-RTNcal/snapshots/bd7a934213c47a9e7ef69eef36bb3325f47fd1f1
FILES="vllm/_custom_ops.py vllm/config/vllm.py vllm/model_executor/models/config.py vllm/v1/core/kv_cache_utils.py vllm/v1/core/single_type_kv_cache_manager.py vllm/v1/worker/gpu/model_states/mamba_hybrid.py vllm/v1/worker/gpu_model_runner.py"
ROLLED=0

stamp() { echo "===== $(date '+%F %T') $*"; }

# Das Aufraeumen exakt wiederherstellen (aus dem vorher gesicherten Diff).
# Laeuft auch beim Abbruch ueber den EXIT-Trap, damit die Produktion nie im
# zurueckgenommenen Zustand stehen bleibt.
restore() {
  if [ "$ROLLED" = 1 ]; then
    if git -C "$WT" apply "$OUT/cleanup_before.diff"; then ROLLED=0; fi
    git -C "$WT" diff -- $FILES > "$OUT/cleanup_after.diff"
    if cmp -s "$OUT/cleanup_before.diff" "$OUT/cleanup_after.diff"; then
      echo "RESTORE: Aufraeumen exakt wiederhergestellt"
    else
      echo "RESTORE: ABWEICHUNG - Worktree von Hand pruefen (cleanup_before.diff)"
    fi
  fi
}
trap restore EXIT

sha_of() { grep -o '"sha256": "[0-9a-f]*"' "$HOME/.cache/mtp-diagnostics/qual_$1/result.json" | sed "s/^/$1 /"; }
fnq() { VENV=$VENV bash tools/mtp-diagnostics/flashnext_qual.sh "$1" 4 2>&1 | grep "q[123]:\|UP on\|FERTIG\|Traceback\|Error"; }
dflash_v100() { DEVS=1,3 DRAFT=$D bash tools/mtp-diagnostics/speed_dflash.sh "$1" fork dflash 2>&1 | tail -2; sha_of "$1"; }

stamp "A) 27B-MTP (V1-Runner), exakter Produktionsbefehl, MIT Aufraeumen"
bash "$S/prod_accept_p1.sh" "$OUT/prod_clean"
"$VENV/bin/python" - "$S/prod/Qwen3.8-27B-NVFP4-vllm.p1.json" "$OUT/prod_clean/Qwen3.8-27B-NVFP4-vllm.p1.json" <<'PY'
import json, sys
try:
    a, b = (json.load(open(p))["choices"][0]["message"]["content"] for p in sys.argv[1:3])
except Exception as e:
    print(f"27B-MTP: VERGLEICH NICHT MOEGLICH: {e}"); sys.exit(0)
print("27B-MTP: Text gleich der Referenz" if a == b else f"27B-MTP: ABWEICHUNG\nREF: {a[:300]}\nNEU: {b[:300]}")
PY

cd "$REPO"
stamp "B) 27B DFlash2 V100 (V2-Runner), MIT Aufraeumen"
dflash_v100 acc2_clean_v100

stamp "C) Flash-Next heterogen, MIT Aufraeumen, 3 Laeufe"
for r in 1 2 3; do stamp "C.$r"; fnq "acc2_clean_$r"; done

stamp "D) Aufraeumen ZURUECKNEHMEN (7 Dateien im Produktions-Worktree)"
git -C "$WT" diff -- $FILES > "$OUT/cleanup_before.diff"
if git -C "$WT" checkout HEAD -- $FILES; then ROLLED=1; fi
git -C "$WT" status --short
stamp "D1) 27B DFlash2 V100, OHNE Aufraeumen"
dflash_v100 acc2_base_v100
stamp "D2) Flash-Next heterogen, OHNE Aufraeumen, 3 Laeufe"
for r in 1 2 3; do stamp "D2.$r"; fnq "acc2_base_$r"; done
stamp "D3) Aufraeumen WIEDER ANWENDEN"
restore
git -C "$WT" status --short

stamp "E) DeepSeek PP5 (V1-Runner), MIT Aufraeumen"
VENV=$VENV bash "$S/ds_accept.sh" acc2_clean > "$OUT/ds_acc2_clean.out" 2>&1
norm() { grep "^\[" "$1" | sed -E 's/ [0-9]+ tok in [0-9.]+s \([0-9.]+ tok\/s\): / /'; }
if diff <(norm "$S/ds_main.out") <(norm "$OUT/ds_acc2_clean.out") > /dev/null; then
  echo "DeepSeek: alle Antworten gleich der Referenz"
else
  echo "DeepSeek: ABWEICHUNG"; diff <(norm "$S/ds_main.out") <(norm "$OUT/ds_acc2_clean.out") | head -10
fi
grep "STATUS\|tok/s" "$OUT/ds_acc2_clean.out" | head -20
stamp "ABNAHME-ENDE"
