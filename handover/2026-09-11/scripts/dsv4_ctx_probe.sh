#!/usr/bin/env bash
# Kontextsuche DeepSeek PP5: exakter YAML-Befehl auf Port 8093, nur
# --max-model-len und --num-gpu-blocks-override ersetzt. Bootet, liest die
# KV-Zeilen bzw. die Fehlermeldung aus, laesst den Server bei Erfolg laufen
# (KEEP=1) oder beendet die eigene Prozessgruppe.
# Aufruf: dsv4_ctx_probe.sh <outdir> <max_model_len> <blocks> [KEEP]
set -uo pipefail
OUT=$1; MML=$2; BLK=$3; KEEP=${4:-0}; mkdir -p "$OUT"; cd /tmp
PY=/home/mp/vllm/venv/bin/python
E=DeepSeek-V4-Flash-nvfp4-DSpark-vllm
$PY - "$E" "$OUT/launch.sh" "$MML" "$BLK" <<'PY'
import os, re, shlex, sys, yaml
name, out, mml, blk = sys.argv[1:5]
m = yaml.safe_load(open("/home/mp/.config/llama-swap/config.yaml"))["models"][name]
cmd = " ".join(m["cmd"].split()).replace("${PORT}", "8093")
cmd, n1 = re.subn(r"--max-model-len \S+", f"--max-model-len {mml}", cmd)
cmd, n2 = re.subn(r"--num-gpu-blocks-override \S+", f"--num-gpu-blocks-override {blk}", cmd)
assert n1 == 1 and n2 == 1, (n1, n2)
with open(out, "w") as f:
    f.write("#!/usr/bin/env bash\n")
    for e in m.get("env", []):
        k, v = e.split("=", 1)
        f.write(f"export {k}={shlex.quote(v)}\n")
    for e in os.environ.get("EXTRA_ENV", "").split():
        k, v = e.split("=", 1)
        f.write(f"export {k}={shlex.quote(v)}\n")
    f.write("exec " + cmd + "\n")
PY
command grep -o -- "--max-model-len [0-9]*\|--num-gpu-blocks-override [0-9]*\|INDEXER_PREFILL_TILE_MB=[0-9]*" "$OUT/launch.sh"
setsid bash "$OUT/launch.sh" > "$OUT/boot.log" 2>&1 &
PID=$!; T0=$(date +%s); ST=timeout
for i in $(seq 1 720); do
  curl -sf -o /dev/null --max-time 2 http://127.0.0.1:8093/v1/models && { ST="up_$(( $(date +%s) - T0 ))s"; break; }
  kill -0 $PID 2>/dev/null || { ST="exited_$(( $(date +%s) - T0 ))s"; break; }
  sleep 5
done
echo "== $ST (pgid $PID)"
command grep -o "Overriding num_gpu_blocks=[-0-9]* with num_gpu_blocks_override=[0-9]*\|GPU KV cache size: [0-9,]* tokens\|Maximum concurrency for [0-9,]* tokens per request: [0-9.]*x\|To serve at least one request.*estimated maximum model length is [0-9]*\|No available memory[^.]*\|CUDA out of memory[^.]*" "$OUT/boot.log" | sort | uniq -c
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo
if [ "$KEEP" != "1" ] || [ "${ST#up_}" = "$ST" ]; then
  kill -TERM -$PID 2>/dev/null; sleep 20; kill -KILL -$PID 2>/dev/null; echo "beendet"
else
  echo "laeuft weiter, pgid $PID"
fi
