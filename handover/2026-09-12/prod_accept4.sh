#!/usr/bin/env bash
# Produktive llama-swap-vLLM-Eintraege auf der neuen venv pruefen.
# Phase 1: exakter Befehl + Umgebung aus config.yaml, eigener Port 8093, lange
#          Geduld (waermt den Compile-Cache, der seit dem Loeschen kalt ist).
# Phase 2: dieselben Eintraege warm ueber llama-swap (Port 11435).
set -uo pipefail
OUT=$1; mkdir -p "$OUT"; cd /tmp
PY=/home/mp/vllm/venv/bin/python
ENTRIES=(Qwen3.8-27B-NVFP4-vllm Qwen3.8-27B-NVFP4-DFlash2-vllm Qwen3.8-Flash-Next-180B-A4B-NVFP4-MTPQ-vllm DeepSeek-V4-Flash-nvfp4-DSpark-vllm)
PROMPT='Erklaere in drei Saetzen, wie ein Regenbogen entsteht.'

gpu_free() { for i in $(seq 1 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc); [ "${u:-0}" -le 1000 ] && return 0; sleep 5; done; return 1; }
ask() {  # $1=url $2=model $3=outfile
  $PY - "$1" "$2" "$PROMPT" "$3" <<'PY'
import json, sys, time, urllib.request
url, model, prompt, out = sys.argv[1:5]
body = {"model": model, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 400, "temperature": 0}
t0 = time.time()
req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.load(r)
except Exception as e:
    print(f"  ANFRAGE FEHLGESCHLAGEN: {e}"); sys.exit(1)
dt = time.time() - t0
msg = d["choices"][0]["message"]; u = d.get("usage", {})
json.dump(d, open(out, "w"), indent=1)
text = (msg.get("content") or "").strip().replace("\n", " ")
print(f"  {u.get('completion_tokens')} tok in {dt:.1f}s, finish={d['choices'][0].get('finish_reason')}")
print(f"  Antwort: {text[:300]}")
PY
}

echo "##### Phase 1: exakte Befehle, Port 8093"
for E in "${ENTRIES[@]}"; do
  gpu_free || { echo "$E: GPUs nicht frei, Abbruch"; exit 1; }
  $PY - "$E" "$OUT/$E.launch.sh" <<'PY'
import shlex, sys, yaml
name, out = sys.argv[1], sys.argv[2]
m = yaml.safe_load(open("/home/mp/.config/llama-swap/config.yaml"))["models"][name]
cmd = " ".join(m["cmd"].split()).replace("${PORT}", "8093")
# Nur Geduld: PP>1 ohne Wachhund-Fix reisst beim kalten Compile nach 600 s.
if "--pipeline-parallel-size 1" not in cmd and "--distributed-timeout-seconds" not in cmd:
    cmd += " --distributed-timeout-seconds 3600"
with open(out, "w") as f:
    f.write("#!/usr/bin/env bash\n")
    for e in m.get("env", []):
        k, v = e.split("=", 1)
        f.write(f"export {k}={shlex.quote(v)}\n")
    f.write("exec " + cmd + "\n")
PY
  setsid bash "$OUT/$E.launch.sh" > "$OUT/$E.boot.log" 2>&1 &
  PID=$!; T0=$(date +%s); ST=timeout
  for i in $(seq 1 720); do
    curl -sf -o /dev/null --max-time 2 http://127.0.0.1:8093/v1/models && { ST="up_$(( $(date +%s) - T0 ))s"; break; }
    kill -0 $PID 2>/dev/null || { ST=exited; break; }
    sleep 5
  done
  echo "== $E: $ST (pgid $PID)"
  [ "${ST#up_}" != "$ST" ] && ask http://127.0.0.1:8093 "$E" "$OUT/$E.p1.json"
  echo "  Traceback: $(grep -c Traceback "$OUT/$E.boot.log")  FA2: $(grep -o 'Loaded FA2 library [^ ]*' "$OUT/$E.boot.log" | sort | uniq -c | tr '\n' ' ')"
  kill -TERM -$PID 2>/dev/null; sleep 20; kill -KILL -$PID 2>/dev/null
done

echo "##### Phase 2: warm ueber llama-swap"
for E in "${ENTRIES[@]}"; do
  gpu_free || { echo "$E: GPUs nicht frei, Abbruch"; exit 1; }
  T0=$(date +%s)
  echo "== $E ueber llama-swap"
  ask http://127.0.0.1:11435 "$E" "$OUT/$E.p2.json"
  echo "  Gesamtzeit inkl. Boot: $(( $(date +%s) - T0 ))s"
  curl -s -o /dev/null --max-time 120 http://127.0.0.1:11435/unload; sleep 30
done
echo "FERTIG"
