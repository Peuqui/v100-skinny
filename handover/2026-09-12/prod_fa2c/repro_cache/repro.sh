#!/usr/bin/env bash
# Flash-Next warm ueber den exakten Produktionsbefehl (Port 8093), zweimal:
#   a) wie Phase 2: Compile-Cache an, Artefakte des kalten Boots vorhanden -> stuerzt es wieder?
#   b) VLLM_DISABLE_COMPILE_CACHE=1 -> laeuft es ohne Artefakt-Laden?
set -uo pipefail
A=/home/mp/Projekte/vllm-research/v100-skinny/handover/2026-09-12/prod_fa2c; R=$A/repro_cache
L=$A/Qwen3.8-Flash-Next-180B-A4B-NVFP4-MTPQ-vllm.launch.sh; PY=/home/mp/vllm/venv/bin/python
wait_free() { for i in $(seq 1 120); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc); [ "${u:-0}" -le 1000 ] && return 0; sleep 5; done; echo "GPUs nicht frei"; exit 1; }
run() { # $1 name, $2 extra env line
  wait_free; sed "s|^exec |$2\nexec |" $L > $R/$1.launch.sh; echo "== $1"; date +%T
  setsid bash $R/$1.launch.sh > $R/$1.boot.log 2>&1 & PID=$!; T0=$(date +%s); ST=timeout
  for i in $(seq 1 360); do curl -sf -o /dev/null --max-time 2 http://127.0.0.1:8093/v1/models && { ST="up_$(( $(date +%s) - T0 ))s"; break; }; kill -0 $PID 2>/dev/null || { ST=exited; break; }; sleep 5; done
  echo "STATUS $ST"
  if [ "${ST#up_}" != "$ST" ]; then $PY - <<'PY'
import json, urllib.request, time
b=json.dumps({"model":"Qwen3.8-Flash-Next-180B-A4B-NVFP4-MTPQ-vllm","messages":[{"role":"user","content":"Erklaere in drei Saetzen, wie ein Regenbogen entsteht."}],"max_tokens":400,"temperature":0}).encode()
t0=time.time()
try:
    d=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8093/v1/chat/completions",data=b,headers={"Content-Type":"application/json"}),timeout=1800))
    print(f"  {d['usage']['completion_tokens']} tok in {time.time()-t0:.1f}s: {(d['choices'][0]['message'].get('content') or '')[:120]!r}")
except Exception as e: print("  ANFRAGE FEHLGESCHLAGEN:", e)
PY
  fi
  echo "  AOT: $(grep -a -c 'Directly load AOT' $R/$1.boot.log) geladen, $(grep -a -c 'load failure' $R/$1.boot.log) Ladefehler, $(grep -a -c 'compile cache is disabled' $R/$1.boot.log) cache-aus, illegal: $(grep -a -c 'illegal memory access' $R/$1.boot.log)"
  kill -TERM -- -$PID 2>/dev/null; sleep 20; kill -KILL -- -$PID 2>/dev/null; wait $PID 2>/dev/null; sleep 5
}
run a_cache_an "export REPRO=a"
run b_cache_aus "export VLLM_DISABLE_COMPILE_CACHE=1"
echo REPRO-ENDE
