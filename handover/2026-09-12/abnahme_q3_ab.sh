#!/usr/bin/env bash
# Nachlauf 2: Kuanda-Frage alt gegen neu, je ein Boot (TP2xPP2 heterogen, k=4), Frage dreimal.
set -uo pipefail
cd /home/mp/Projekte/vllm-research/v100-skinny
for i in $(seq 1 600); do grep -qE "RTXAB-ENDE|^ABBRUCH" handover/2026-09-12/abnahme_rtx_ab.out 2>/dev/null && break; sleep 30; done
OLDV=/home/mp/Projekte/vllm-research/v100-skinny/.venv-sm70-old
for P in "q3_old $OLDV" "q3_new /home/mp/vllm/venv"; do
  set -- $P; echo "== $1 ($2)"
  VENV=$2 bash handover/2026-09-12/flashnext_q3_rate.sh $1 4 2>&1 | grep -E "STATUS|q3[abc]|Error|ABBRUCH|FERTIG" | cut -c1-200
  for t in a b c; do f=~/.cache/mtp-diagnostics/fnq3_$1/raw_q3$t.json; [ -f "$f" ] && /home/mp/vllm/venv/bin/python -c "
import json;d=json.load(open('$f'));m=d['choices'][0]['message'];c=m.get('content') or '';r=m.get('reasoning') or m.get('reasoning_content') or ''
v='bestanden (Coandă)' if 'Coand' in c else ('zurueckgewiesen' if any(k in c for k in ('Kunda','nicht bekannt','Tippfehler','Schreibfehler','kein etablierter','vermutlich')) else 'ERFUNDEN')
print('   q3$t:', v, '| Coand im Denkblock:', 'Coand' in r, '| Zeichen', len(c), '| Anfang:', c.strip()[:90].replace(chr(10),' '))"; done
done
echo Q3AB-ENDE
