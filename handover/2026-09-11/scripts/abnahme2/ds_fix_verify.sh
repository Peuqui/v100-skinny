#!/usr/bin/env bash
# Verifikation des SWA-Schwellen-Fixes: ein Boot (Serve-Skript, PP5), alle
# Prompt-Laengen 4..16 aufsteigend, danach Tool-Call Runde 1 + 2 + Kontrolle.
set -uo pipefail
REPO=/home/mp/Projekte/vllm-research/v100-skinny; cd $REPO
PY=/home/mp/vllm/venv/bin/python
OUT=handover/2026-09-11/ergebnisse/ds_fix_verify; mkdir -p $OUT
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_HOME=/home/mp/vllm/cuda PATH=/home/mp/vllm/cuda/bin:$PATH
orphans() { ps -eo pid,ppid,comm | awk '$2==1 && $3 ~ /^VLLM::Worker/ {print $1}'; }
for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc); [ "${u:-0}" -le 1000 ] && ! pgrep -f '[a]pi_server' >/dev/null && break; sleep 5; done
echo "== Boot mit Fix"
setsid env VENV=/home/mp/vllm/venv bash scripts/serve-deepseek-het-graphs.sh > $OUT/boot.log 2>&1 &
SPID=$!
for i in $(seq 1 720); do curl -sf -o /dev/null --max-time 2 http://127.0.0.1:19998/v1/models && { echo "  up nach $((i*5))s"; break; }; kill -0 $SPID 2>/dev/null || { echo "  exited"; break; }; sleep 5; done
$PY - "$OUT" <<'PYX'
import json, sys, time, urllib.request, urllib.error
out = sys.argv[1]
sel = {int(k): v for k, v in json.load(open("handover/2026-09-11/scripts/abnahme2/ds_prompts_by_len.json")).items()}
ok = 0
for L in sorted(sel):
    body = {"model": "dsv4-manual", "messages": [{"role": "user", "content": sel[L]}], "max_tokens": 8, "temperature": 0, "seed": 1}
    req = urllib.request.Request("http://127.0.0.1:19998/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=340) as r: d = json.load(r)
        ok += 1; print(f"  {L:2d} Token: OK {time.perf_counter()-t0:.1f}s out={d['usage']['completion_tokens']}", flush=True)
    except Exception as e:
        print(f"  {L:2d} Token: {type(e).__name__} nach {time.perf_counter()-t0:.0f}s -> ABSTURZ", flush=True); break
print(f"LAENGEN {ok}/{len(sel)} ok", flush=True)
PYX
echo "== Tool-Call gegen denselben Server"
$PY handover/2026-09-11/scripts/abnahme2/toolcall_probe.py 19998 dsv4-manual $OUT/toolcall 2>&1 | tail -8
echo "  Assertion im Log: $(grep -a -c 'assert topk_indices is not None' $OUT/boot.log)"
kill -TERM $SPID 2>/dev/null; sleep 15; kill -KILL $SPID 2>/dev/null
p=$(orphans); [ -n "$p" ] && { kill -TERM $p 2>/dev/null; sleep 12; p=$(orphans); [ -n "$p" ] && kill -KILL $p 2>/dev/null; }
sleep 3; echo "  VRAM: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd' ')"
echo VERIFY-ENDE
