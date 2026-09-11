#!/usr/bin/env bash
# Paket 11a End-zu-End: 1Cat main (fe67339d) + #572, RTX-Paar, DFlash2 k=7 mit dem
# BF16-Entwurfskopf (incoai, kein DRAFT=), greedy. Arm A ohne Fix, Arm B mit Fix.
# Metrik: Annahmelaenge (frueher auf dem Overlay 1,015 -> 3,353), Text-SHA, tok/s.
# Dazu die zwei CUDA-Faelle test_sm70_dflash2_exact_rerank_matches_gathered_bmm[7/8]
# auf einer freien V100 (GPU 4), main gegen Fix. Aufruf: e2e_11a.sh <OUT>
set -uo pipefail
OUT=${1:?OUT fehlt}; mkdir -p "$OUT"
E2E=/home/mp/Projekte/vllm-research/1Cat-vLLM-e2e-572
REPO=/home/mp/Projekte/vllm-research/v100-skinny
PY=/home/mp/vllm/venv/bin/python
FIX=$(dirname "$0")/pr11a_fix.diff
F=vllm/model_executor/models/qwen3_dflash2.py
stamp() { echo "===== $(date '+%F %T') $*"; }

# Nachweis, welcher Code laeuft: vllm.__file__ und der Quelltext des Gates.
probe() {
  ( cd "$OUT" && CUDA_VISIBLE_DEVICES="" PYTHONPATH=$E2E "$PY" - <<'PY' 2>/dev/null | grep -v "^W0"
import vllm, inspect
from vllm.model_executor.models import qwen3_dflash2 as m
print("vllm:", vllm.__file__)
src = inspect.getsource(m._use_sm70_bf16_emulation)
print("gate:", "has_device_capability(80, worker)" if "has_device_capability" in src else "is_device_capability(70), Geraet 0")
PY
  )
}

restore() { git -C "$E2E" checkout -- "$F"; }
trap restore EXIT

git -C "$E2E" diff --quiet -- "$F" || { echo "ABBRUCH: e2e-Worktree hat lokale Aenderungen an $F"; exit 1; }
echo "e2e HEAD: $(git -C "$E2E" log --oneline -1)"

stamp "CUDA-Tests auf GPU 4 (freie V100): main+#572 OHNE Fix"
( cd "$E2E" && CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 PYTHONPATH=$E2E "$PY" -m pytest -q -p no:cacheprovider \
    "tests/v1/spec_decode/test_dflash2.py::test_sm70_dflash2_exact_rerank_matches_gathered_bmm" \
    tests/v1/spec_decode/test_dflash2_pre_ampere_gate.py 2>&1 | tail -4 ) || true
git -C "$E2E" apply "$FIX"
stamp "CUDA-Tests auf GPU 4: main+#572 MIT Fix"
( cd "$E2E" && CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 PYTHONPATH=$E2E "$PY" -m pytest -q -p no:cacheprovider \
    "tests/v1/spec_decode/test_dflash2.py::test_sm70_dflash2_exact_rerank_matches_gathered_bmm" 2>&1 | tail -4 ) || true
restore

cd "$REPO"
export PYTHONPATH=$E2E
stamp "Arm A: main+#572 OHNE Fix, RTX-Paar, DFlash2 k=7, BF16-Kopf"
probe
DEVS=0,2 bash tools/mtp-diagnostics/speed_dflash.sh e2e572_nofix fork dflash 2>&1 | tail -3
grep -o '"sha256": "[0-9a-f]*"' "$HOME/.cache/mtp-diagnostics/qual_e2e572_nofix/result.json" | sed 's/^/e2e572_nofix /'
grep -c "Traceback" "$HOME/.cache/mtp-diagnostics/qual_e2e572_nofix/boot.log" | sed 's/^/Tracebacks im Boot-Log: /'

stamp "Arm B: main+#572 MIT Fix"
git -C "$E2E" apply "$FIX" || { echo "ABBRUCH: Fix laesst sich nicht anwenden"; exit 1; }
probe
DEVS=0,2 bash tools/mtp-diagnostics/speed_dflash.sh e2e572_fix fork dflash 2>&1 | tail -3
grep -o '"sha256": "[0-9a-f]*"' "$HOME/.cache/mtp-diagnostics/qual_e2e572_fix/result.json" | sed 's/^/e2e572_fix /'
grep -c "Traceback" "$HOME/.cache/mtp-diagnostics/qual_e2e572_fix/boot.log" | sed 's/^/Tracebacks im Boot-Log: /'
restore
stamp "E2E-11A-ENDE"
