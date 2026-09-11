#!/usr/bin/env bash
# Abnahme nach dem Aufraeumen der Merge-Reste (Befunde 1,3,4,5,6,7,8):
# alle vier Produktionsmodelle unter den Bedingungen der Referenzlaeufe vom 10.09.
set -uo pipefail
S=$(dirname "$0"); TAG=${1:-clean}
D=/home/mp/.cache/huggingface/hub/models--maurienne-ai--Qwen3.8-27B-DFlash2-NVFP4-RTNcal/snapshots/bd7a934213c47a9e7ef69eef36bb3325f47fd1f1
echo "##### 27B-MTP (V1-Runner), exakter Produktionsbefehl"
bash "$S/prod_accept_p1.sh" "$S/prod_$TAG"
cd /home/mp/Projekte/vllm-research/v100-skinny
echo "##### 27B DFlash2 RTX / V100 (V2-Runner)"
DEVS=0,2 DRAFT=$D bash tools/mtp-diagnostics/speed_dflash.sh ${TAG}_rtx fork dflash 2>&1 | tail -2
DEVS=1,3 DRAFT=$D bash tools/mtp-diagnostics/speed_dflash.sh ${TAG}_v100 fork dflash 2>&1 | tail -2
for n in ${TAG}_rtx ${TAG}_v100; do command grep -o '"sha256": "[0-9a-f]*"' $HOME/.cache/mtp-diagnostics/qual_$n/result.json | sed "s/^/$n /"; done
echo "##### Flash-Next heterogen (V2-Runner)"
VENV=/home/mp/Projekte/vllm-research/v100-skinny/.venv-sm70-main bash tools/mtp-diagnostics/flashnext_qual.sh ${TAG}_hetero 4 2>&1 | command grep "q[123]:\|UP on\|FERTIG\|Traceback\|Error"
echo "##### DeepSeek PP5 (V1-Runner)"
VENV=/home/mp/Projekte/vllm-research/v100-skinny/.venv-sm70-main bash "$S/ds_accept.sh" "$TAG" > "$S/ds_$TAG.out" 2>&1
diff <(command grep "^\[" "$S/ds_main.out" | sed -E 's/ [0-9]+ tok in [0-9.]+s \([0-9.]+ tok\/s\): / /') <(command grep "^\[" "$S/ds_$TAG.out" | sed -E 's/ [0-9]+ tok in [0-9.]+s \([0-9.]+ tok\/s\): / /') > /dev/null && echo "DeepSeek: alle Antworten gleich der Referenz" || { echo "DeepSeek: ABWEICHUNG"; diff <(command grep "^\[" "$S/ds_main.out" | sed -E 's/ [0-9]+ tok in [0-9.]+s \([0-9.]+ tok\/s\): / /') <(command grep "^\[" "$S/ds_$TAG.out" | sed -E 's/ [0-9]+ tok in [0-9.]+s \([0-9.]+ tok\/s\): / /') | head -10; }
echo "##### 27B-MTP Textvergleich"
/home/mp/vllm/venv/bin/python - "$S/prod/Qwen3.8-27B-NVFP4-vllm.p1.json" "$S/prod_$TAG/Qwen3.8-27B-NVFP4-vllm.p1.json" <<'PY'
import json, sys
a, b = (json.load(open(p))["choices"][0]["message"]["content"] for p in sys.argv[1:3])
print("27B-MTP: Text gleich der Referenz" if a == b else f"27B-MTP: ABWEICHUNG\n{a[:200]}\n{b[:200]}")
PY
echo ABNAHME-ENDE
