#!/usr/bin/env bash
# A/B fuer den DeepSeek-Absturz bei kurzem Chat-Prompt (12.09., AssertionError
# topk_indices in amd/rocm.py:788): (A) Serve-Skript ohne Prefix-Cache,
# (B) llama-swap-Eintrag mit Prefix-Cache. Gleicher Prompt, gleiche Sonde.
set -uo pipefail
REPO=/home/mp/Projekte/vllm-research/v100-skinny; cd $REPO
PY=/home/mp/vllm/venv/bin/python
OUT=handover/2026-09-11/ergebnisse/ds_short_prompt_ab; mkdir -p $OUT
probe() {  # $1 port $2 model $3 tag
  $PY - "$1" "$2" "$OUT/$3" <<'PYX'
import json, sys, time, urllib.request, urllib.error
port, model, out = sys.argv[1:4]
def ask(tag, msg, mt):
    body = {"model": model, "messages": [{"role": "user", "content": msg}], "max_tokens": mt, "temperature": 0, "seed": 1}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r: d = json.load(r)
    except urllib.error.HTTPError as e:
        print(f"  {tag}: HTTP {e.code} nach {time.perf_counter()-t0:.0f}s", flush=True); return False
    except Exception as e:
        print(f"  {tag}: {type(e).__name__} nach {time.perf_counter()-t0:.0f}s", flush=True); return False
    u = d["usage"]; c = d["choices"][0]["message"].get("content") or ""
    print(f"  {tag}: ok {time.perf_counter()-t0:.1f}s prompt={u['prompt_tokens']} out={u['completion_tokens']} {c[:60]!r}", flush=True)
    json.dump(d, open(f"{out}_{tag}.json", "w"), ensure_ascii=False, indent=1); return True
ask("kurz8", "Guten Tag.", 8) and ask("kurz10", "Nenne drei Farben.", 40) and ask("lang", "Erklaere in fuenf Saetzen, wie ein Regenbogen entsteht.", 120)
PYX
}
wait_free() { for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc); [ "${u:-0}" -le 1000 ] && ! pgrep -f '[a]pi_server' >/dev/null && return; sleep 5; done; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_HOME=/home/mp/vllm/cuda PATH=/home/mp/vllm/cuda/bin:$PATH
echo "== A: Serve-Skript ohne Prefix-Cache (Port 19998)"
wait_free
setsid env VENV=/home/mp/vllm/venv bash scripts/serve-deepseek-het-graphs.sh > $OUT/A_boot.log 2>&1 &
APID=$!; APG=$(ps -o pgid= -p $APID | tr -d ' ')
for i in $(seq 1 720); do curl -sf -o /dev/null --max-time 2 http://127.0.0.1:19998/v1/models && { echo "  up nach $((i*5))s"; break; }; kill -0 $APID 2>/dev/null || { echo "  exited"; break; }; sleep 5; done
probe 19998 dsv4-manual A
kill -TERM -- -$APG 2>/dev/null; sleep 15; kill -KILL -- -$APG 2>/dev/null
echo "  A-Leichen: $(ps -eo pid,pgid,args | awk -v g=$APG '$2==g && /VLLM::Worker/' | wc -l)"
echo "== B: llama-swap-Eintrag mit Prefix-Cache (Port 11435)"
wait_free
probe 11435 DeepSeek-V4-Flash-nvfp4-DSpark-vllm B
sleep 5
# Aufraeumen: Worker, deren Prozessgruppe keinen lebenden api_server mehr hat
for pg in $(ps -eo pgid,args | awk '/[V]LLM::Worker/ {print $1}' | sort -u); do
  ps -eo pgid,args | awk -v g=$pg '$1==g && /[a]pi_server/' | grep -q . || { echo "  Leichen pgid $pg beendet"; kill -TERM -- -$pg 2>/dev/null; sleep 10; kill -KILL -- -$pg 2>/dev/null; }
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | paste -sd' '
echo AB-ENDE
