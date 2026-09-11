#!/usr/bin/env bash
# 11a: die drei CUDA-Faelle test_sm70_dflash2_exact_rerank_matches_gathered_bmm
# auf einer echten Tesla V100 (PCI-Index 4), main+#572 ohne und mit Fix.
# CUDA_DEVICE_ORDER=PCI_BUS_ID ist Pflicht: ohne die Variable ist Index 4 eine
# RTX 8000 und der Test ueberspringt sich selbst als "SM70-only" (11.09.).
set -uo pipefail
E2E=/home/mp/Projekte/vllm-research/1Cat-vLLM-e2e-572
PY=/home/mp/vllm/venv/bin/python
FIX=$(dirname "$0")/pr11a_fix.diff
F=vllm/model_executor/models/qwen3_dflash2.py
T="tests/v1/spec_decode/test_dflash2.py::test_sm70_dflash2_exact_rerank_matches_gathered_bmm"
run() { ( cd "$E2E" && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 PYTHONPATH=$E2E "$PY" -m pytest -q -rs -p no:cacheprovider "$T" 2>&1 | grep -i "skipped\|passed\|failed" | tail -3 ); }
git -C "$E2E" diff --quiet -- "$F" || { echo "ABBRUCH: e2e-Worktree nicht sauber"; exit 1; }
echo "== Karte hinter CUDA_VISIBLE_DEVICES=4 (PCI_BUS_ID): $(CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 $PY -c 'import torch;print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))' 2>/dev/null | tail -1)"
echo "== OHNE Fix"; run
git -C "$E2E" apply "$FIX" && { echo "== MIT Fix"; run; git -C "$E2E" checkout -- "$F"; }
echo "RERANK-ENDE"
