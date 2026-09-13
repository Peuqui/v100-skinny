#!/usr/bin/env bash
# Beleg torch-Backport #173556: je Modell kalt, warm1, warm2 mit dem exakten Produktionsbefehl (Port 8093),
# Compile-Cache an (Produktions-Default). Erwartung: warm1 und warm2 laden die Artefakte, Text byteidentisch, kein Absturz.
set -uo pipefail
A=/home/mp/Projekte/vllm-research/v100-skinny/handover/2026-09-12/prod_fa2c; B=$A/beleg_patch; PY=/home/mp/vllm/venv/bin/python
wait_free() { for i in $(seq 1 120); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc); [ "${u:-0}" -le 1000 ] && return 0; sleep 5; done; echo "GPUs nicht frei"; exit 1; }
run() { # $1 name $2 launch $3 model
  wait_free; echo "== $1"; date +%T
  setsid bash $2 > $B/$1.boot.log 2>&1 & PID=$!; T0=$(date +%s); ST=timeout
  for i in $(seq 1 360); do curl -sf -o /dev/null --max-time 2 http://127.0.0.1:8093/v1/models && { ST="up_$(( $(date +%s) - T0 ))s"; break; }; kill -0 $PID 2>/dev/null || { ST=exited; break; }; sleep 5; done
  echo "STATUS $ST"
  if [ "${ST#up_}" != "$ST" ]; then $PY - "$3" "$B/$1" <<'PY'
import json, urllib.request, time, hashlib, sys
model, out = sys.argv[1:3]
b=json.dumps({"model":model,"messages":[{"role":"user","content":"Erklaere in drei Saetzen, wie ein Regenbogen entsteht."}],"max_tokens":400,"temperature":0}).encode()
t0=time.time()
try:
    d=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8093/v1/chat/completions",data=b,headers={"Content-Type":"application/json"}),timeout=1800))
    txt=(d['choices'][0]['message'].get('content') or ''); json.dump(d, open(out+".json","w"), indent=1)
    print(f"  {d['usage']['completion_tokens']} tok in {time.time()-t0:.1f}s sha {hashlib.sha256(txt.encode()).hexdigest()[:16]}")
except Exception as e: print("  ANFRAGE FEHLGESCHLAGEN:", e)
PY
  fi
  echo "  AOT: $(grep -a -c 'Directly load AOT' $B/$1.boot.log) geladen, $(grep -a -c 'load failure' $B/$1.boot.log) Ladefehler, $(grep -a -c 'illegal memory access' $B/$1.boot.log) illegal, $(grep -a -c 'Traceback' $B/$1.boot.log) Traceback"
  kill -TERM -- -$PID 2>/dev/null; sleep 20; kill -KILL -- -$PID 2>/dev/null; wait $PID 2>/dev/null; sleep 5
}
FN=$A/Qwen3.8-Flash-Next-180B-A4B-NVFP4-MTPQ-vllm.launch.sh; Q=$A/Qwen3.8-27B-NVFP4-vllm.launch.sh
for s in kalt warm1 warm2; do run fn_$s $FN Qwen3.8-Flash-Next-180B-A4B-NVFP4-MTPQ-vllm; done
for s in kalt warm1 warm2; do run q27_$s $Q Qwen3.8-27B-NVFP4-vllm; done
echo BELEG-ENDE
