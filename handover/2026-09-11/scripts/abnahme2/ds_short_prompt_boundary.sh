#!/usr/bin/env bash
# Grenzmessung fuer den DeepSeek-Kurzprompt-Absturz (Assertion topk_indices,
# amd/rocm.py:788). Boot 1: Prompts absteigend ab 16 Token bis zum ersten
# Absturz (Obergrenze). Boot 2: aufsteigend ab 4 Token (Untergrenze).
# Server: scripts/serve-deepseek-het-graphs.sh (PP5, Port 19998, Prod-venv).
set -uo pipefail
REPO=/home/mp/Projekte/vllm-research/v100-skinny; cd $REPO
PY=/home/mp/vllm/venv/bin/python
OUT=handover/2026-09-11/ergebnisse/ds_short_prompt_boundary; mkdir -p $OUT
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_HOME=/home/mp/vllm/cuda PATH=/home/mp/vllm/cuda/bin:$PATH
orphans() { ps -eo pid,ppid,comm | awk '$2==1 && $3 ~ /^VLLM::Worker/ {print $1}'; }
cleanup() {
  local p; p=$(orphans); [ -n "$p" ] && { echo "  Waisen: $(echo $p) -> TERM"; kill -TERM $p 2>/dev/null; sleep 12; p=$(orphans); [ -n "$p" ] && kill -KILL $p 2>/dev/null; }
  sleep 3; echo "  VRAM: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd' ')"
}
wait_free() { for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc); [ "${u:-0}" -le 1000 ] && ! pgrep -f '[a]pi_server' >/dev/null && return; sleep 5; done; }
phase() {  # $1 = desc|asc
  local ORDER=$1
  echo "== Boot ($ORDER)"
  wait_free
  setsid env VENV=/home/mp/vllm/venv bash scripts/serve-deepseek-het-graphs.sh > $OUT/boot_$ORDER.log 2>&1 &
  local SPID=$!
  for i in $(seq 1 720); do curl -sf -o /dev/null --max-time 2 http://127.0.0.1:19998/v1/models && { echo "  up nach $((i*5))s"; break; }; kill -0 $SPID 2>/dev/null || { echo "  exited"; break; }; sleep 5; done
  $PY - "$ORDER" "$OUT" <<'PYX'
import json, sys, time, urllib.request, urllib.error
order, out = sys.argv[1:3]
sel = {int(k): v for k, v in json.load(open("handover/2026-09-11/scripts/abnahme2/ds_prompts_by_len.json")).items()}
lens = sorted(sel, reverse=(order == "desc"))
for L in lens:
    body = {"model": "dsv4-manual", "messages": [{"role": "user", "content": sel[L]}], "max_tokens": 8, "temperature": 0, "seed": 1}
    req = urllib.request.Request("http://127.0.0.1:19998/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=340) as r: d = json.load(r)
        u = d["usage"]; print(f"  {L:2d} Token: OK {time.perf_counter()-t0:.1f}s prompt={u['prompt_tokens']} out={u['completion_tokens']} {(d['choices'][0]['message'].get('content') or '')[:40]!r}", flush=True)
    except urllib.error.HTTPError as e:
        print(f"  {L:2d} Token: HTTP {e.code} nach {time.perf_counter()-t0:.0f}s -> ABSTURZ, Phase Ende", flush=True); break
    except Exception as e:
        print(f"  {L:2d} Token: {type(e).__name__} nach {time.perf_counter()-t0:.0f}s -> ABSTURZ, Phase Ende", flush=True); break
PYX
  echo "  Assertion im Log: $(grep -a -c 'assert topk_indices is not None' $OUT/boot_$ORDER.log)"
  kill -TERM $SPID 2>/dev/null; sleep 15; kill -KILL $SPID 2>/dev/null
  cleanup
}
phase desc
phase asc
echo BOUNDARY-ENDE
