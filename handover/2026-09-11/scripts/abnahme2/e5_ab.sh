#!/usr/bin/env bash
# E5-Cache A/B auf dem V1-Runner: exakter Produktionsbefehl 27B-MTP (Port 8093,
# RTX-Paar), einmal VLLM_SM70_E5_CACHE=0, einmal =1. Je Arm 6 Anfragen, greedy,
# 600 Token; Decode-Rate aus dem Streaming (erstes bis letztes Token), Median
# der Laeufe 2-6, Text-SHA je Anfrage. Nachweis, dass E5 wirklich feuert: die
# [e5-*]-Zeilen im Boot-Log. Aufruf: e5_ab.sh <OUT>   (NUR nach Freigabe)
set -uo pipefail
OUT=${1:?OUT fehlt}; mkdir -p "$OUT"
L=$(dirname "$0")/prod_clean/Qwen3.8-27B-NVFP4-vllm.launch.sh
PY=/home/mp/vllm/venv/bin/python
stamp() { echo "===== $(date '+%F %T') $*"; }
gpu_free() { for i in $(seq 1 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc); [ "${u:-0}" -le 1000 ] && return 0; sleep 5; done; return 1; }

[ -f "$L" ] || { echo "ABBRUCH: $L fehlt (aus prod_accept_p1.sh)"; exit 1; }
for ARM in e5off e5on; do
  V=0; [ "$ARM" = e5on ] && V=1
  stamp "Arm $ARM (VLLM_SM70_E5_CACHE=$V)"
  gpu_free || { echo "GPUs nicht frei"; exit 1; }
  sed "s/^export VLLM_SM70_E5_CACHE=0$/export VLLM_SM70_E5_CACHE=$V/" "$L" > "$OUT/$ARM.launch.sh"
  grep -q "E5_CACHE=$V" "$OUT/$ARM.launch.sh" || { echo "ABBRUCH: Schalter nicht gesetzt"; exit 1; }
  cd /tmp; setsid bash "$OUT/$ARM.launch.sh" > "$OUT/$ARM.boot.log" 2>&1 &
  PID=$!; ST=timeout
  for i in $(seq 1 720); do
    curl -sf -o /dev/null --max-time 2 http://127.0.0.1:8093/v1/models && { ST="up_$((i*5))s"; break; }
    kill -0 $PID 2>/dev/null || { ST=exited; break; }
    sleep 5
  done
  echo "STATUS $ST"
  if [ "${ST#up_}" != "$ST" ]; then
    "$PY" - "$OUT/$ARM" <<'PY'
import hashlib, json, sys, time, urllib.request, statistics
out = sys.argv[1]
body = {"model": "Qwen3.8-27B-NVFP4-vllm",
        "messages": [{"role": "user", "content": "Erklaere die Quantenphysik in 30 Saetzen."}],
        "max_tokens": 600, "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True}}
rates, shas = [], []
for r in range(1, 7):
    req = urllib.request.Request("http://127.0.0.1:8093/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    text, t_first, t_last, usage = [], None, None, {}
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices", []):
                delta = ch.get("delta", {}) or {}
                piece = delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content") or ""
                if piece:
                    if t_first is None:
                        t_first = time.time()
                    t_last = time.time(); text.append(piece)
    n = usage.get("completion_tokens", 0)
    rate = n / (t_last - t_first) if t_first and t_last and t_last > t_first else float("nan")
    sha = hashlib.sha256("".join(text).encode()).hexdigest()[:16]
    rates.append(rate); shas.append(sha)
    open(f"{out}.run{r}.txt", "w").write("".join(text))
    print(f"  run{r}: {n} tok, decode {rate:.2f} tok/s, sha {sha}")
print(f"MEDIAN(run2-6) {statistics.median(rates[1:]):.2f} tok/s  SHAs {'alle gleich' if len(set(shas)) == 1 else 'VERSCHIEDEN: ' + ' '.join(shas)}")
PY
  fi
  echo "  Tracebacks: $(grep -c Traceback "$OUT/$ARM.boot.log")  e5-Zeilen: $(grep -c '\[e5-' "$OUT/$ARM.boot.log")"
  grep '\[e5-' "$OUT/$ARM.boot.log" | tail -3 | cut -c1-160
  kill -TERM -$PID 2>/dev/null; sleep 20; kill -KILL -$PID 2>/dev/null
done
stamp "E5-AB-ENDE"
