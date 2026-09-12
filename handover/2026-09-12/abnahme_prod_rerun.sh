#!/usr/bin/env bash
# Nachlauf 3: die vier produktiven llama-swap-Eintraege kalt und warm, diesmal mit ABSOLUTEM Ausgabepfad
# (prod_accept.sh wechselt nach /tmp; relativer Pfad = keine Startskripte, keine Antwortdateien).
cd /home/mp/Projekte/vllm-research/v100-skinny
for i in $(seq 1 600); do grep -qE "Q3AB-ENDE" handover/2026-09-12/abnahme_q3_ab.out 2>/dev/null && break; sleep 30; done
bash handover/2026-09-12/prod_accept4.sh /home/mp/Projekte/vllm-research/v100-skinny/handover/2026-09-12/prod 2>&1 | grep -E "^==|^#####|tok in|Antwort|Traceback|FEHLGESCHLAGEN|Gesamtzeit|FERTIG"
echo PROD-ENDE
