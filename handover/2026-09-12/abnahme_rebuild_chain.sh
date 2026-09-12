#!/usr/bin/env bash
# Abnahme nach dem work-main-Merge auf main dfef3342 (12.09. nachts). Laeuft erst nach erfolgreichem Neubau.
set -uo pipefail
cd /home/mp/Projekte/vllm-research/v100-skinny
L=handover/2026-09-12/rebuild_work_main.log
for i in $(seq 1 360); do grep -q "REBUILD-ENDE" $L && break; sleep 30; done
grep -q "PIP-EXIT 0" $L || { echo "ABBRUCH: Neubau nicht erfolgreich"; exit 1; }
export VENV=/home/mp/vllm/venv
S=tools/mtp-diagnostics/speed_dflash.sh
echo "== 1 DFlash2 V100-Paar (Referenz SHA 0106659946c064b1, ~76,3 tok/s, Annahme ~3,3)"
DEVS=1,3 bash $S rb_v100 fork dflash 2>&1 | grep -E "STATUS|MEDIAN|SHA|Annahme|Error|ABBRUCH"
echo "== 2 DFlash2 V100-Paar, Tail-Cudagraphs AUS (alter Default)"
DEVS=1,3 VLLM_SM70_DFLASH2_TAIL_CUDAGRAPHS=0 bash $S rb_v100_tail0 fork dflash 2>&1 | grep -E "STATUS|MEDIAN|SHA|Annahme|Error|ABBRUCH"
echo "== 3 DFlash2 RTX-Paar (Referenz ~77,1 tok/s, SHA gleich)"
DEVS=0,2 bash $S rb_rtx fork dflash 2>&1 | grep -E "STATUS|MEDIAN|SHA|Annahme|Error|ABBRUCH"
echo "== 4 DeepSeek PP5 Kohaerenz zweimal (Referenz 8/8, byteidentisch)"
bash handover/2026-09-11/scripts/ds_accept.sh rebuild 2>&1 | grep -E "STATUS|Lauf|/8|PASS|FAIL|verdict|Error|FERTIG" | head -30
echo "== 5 Flash-Next Chat-Qualitaet k=4 (Referenz 3/3)"
bash handover/2026-09-11/scripts/abnahme2/flashnext_qual_chat.sh rb 4 2>&1 | grep -E "STATUS|q[123]|tok/s|bestanden|PASS|FAIL|Error|FERTIG|ABBRUCH" | head -20
echo "== 6 Produktive llama-swap-Eintraege kalt und warm (4 Eintraege)"
bash handover/2026-09-12/prod_accept4.sh handover/2026-09-12/prod 2>&1 | grep -E "^==|^#####|tok in|Antwort|Traceback|FEHLGESCHLAGEN|Gesamtzeit|FERTIG"
echo ABNAHME-ENDE
