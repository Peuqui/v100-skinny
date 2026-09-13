#!/usr/bin/env bash
# Paket C, Beleg 3: gemischtes Volta+Turing-Wheel (PR-C auf main dfef3342 + #604) im Betrieb.
# Arm v100:      27B-Produktionsbefehl auf dem V100-Paar (PCI 1,3)  -> laedt _vllm_fa2_C
# Arm rtx:       27B-Produktionsbefehl auf dem RTX-Paar  (PCI 0,2)  -> laedt _vllm_fa2_C_sm75
# Arm flashnext: Flash-Next TP2 PP2 exakt wie Produktion (0,2,1,3) -> beide Bibliotheken in einem Modell
# Python: .venv-pr-fa2sm75 (gemischtes Wheel), Code: Worktree 1Cat-vLLM-e2e-fa2mixed via PYTHONPATH,
# dessen .so aus dem Wheel kopiert sind. Aufruf: e2e_fa2mixed.sh <arm> <OUT>
set -uo pipefail
ARM=$1; OUT=$2; mkdir -p "$OUT"; cd /tmp
E=/home/mp/Projekte/vllm-research/1Cat-vLLM-e2e-fa2mixed
V=/home/mp/Projekte/vllm-research/v100-skinny/.venv-pr-fa2sm75
PY=$V/bin/python; PORT=8094
case $ARM in
  # v100: der Produktionsbefehl erzwingt fuer den MTP-Entwurfskopf FLASH_ATTN; das nimmt nur der Fork-Overlay
  #       auf Volta an. main waehlt auf 7.0 FLASH_ATTN_V100 selbst -> Angabe entfernen (auto).
  # rtx:  der 27B deklariert FP8-KV; ohne #613 im Stapel muss fp16 explizit gesetzt werden (sm75-FA2 ist fp16-only).
  # v100/mixed27b: der Produktionsbefehl (262k, 98 %) ist auf 48-GB-RTX ausgelegt; auf 32-GB-V100 laeuft er in der
  #       Inductor-Autotune-Phase in OOM -> 32k Kontext und 90 % wie in der Nachtmessung (fa2_probe.sh).
  # mixed27b: 27B als TP1 PP2 ueber RTX (Stufe 0, PCI 0) und V100 (Stufe 1, PCI 1) -> beide FA2-Bibliotheken
  #       in einem Modell. Flash-Next geht auf main nicht: NVFP4-MoE auf Volta (SM70_SKINNY) ist Fork-Overlay.
  v100)      ENTRY=Qwen3.8-27B-NVFP4-vllm; DEVS=1,3; ARCH=7.0; SPEC_STRIP=1; EXTRA=""; SUBST="v100mem" ;;
  rtx)       ENTRY=Qwen3.8-27B-NVFP4-vllm; DEVS=0,2; ARCH=7.5; SPEC_STRIP=0; EXTRA="--kv-cache-dtype float16"; SUBST="" ;;
  # mixed27b: der 27B kann kein PP (kein SupportsPP) -> TP2 ueber RTX (Rang 0, PCI 0) und V100 (Rang 1, PCI 1).
  mixed27b)  ENTRY=Qwen3.8-27B-NVFP4-vllm; DEVS=0,1; ARCH="7.0;7.5"; SPEC_STRIP=1; EXTRA="--kv-cache-dtype float16"; SUBST="v100mem" ;;
  flashnext) ENTRY=Qwen3.8-Flash-Next-180B-A4B-NVFP4-MTPQ-vllm; DEVS=0,2,1,3; ARCH="7.0;7.5"; SPEC_STRIP=0; EXTRA=""; SUBST="" ;;
  *) echo "unbekannter Arm $ARM"; exit 2 ;;
esac
# Fremde Server (llama-swap) stoeren nicht, solange die Ziel-GPUs frei sind und Port $PORT unbelegt ist.
curl -sf -o /dev/null --max-time 2 http://127.0.0.1:$PORT/v1/models && { echo "ABBRUCH: Port $PORT belegt"; exit 1; }
for d in $(echo $DEVS | tr , ' '); do u=$(nvidia-smi -i $d --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -gt 1000 ] && { echo "ABBRUCH: GPU $d belegt ($u MiB)"; exit 1; }; done
$PY - "$ENTRY" "$OUT/$ARM.launch.sh" "$PORT" "$DEVS" "$ARCH" "$V" "$E" "$SPEC_STRIP" "$EXTRA" "$SUBST" <<'PY'
import shlex, sys, yaml
name, out, port, devs, arch, venv, tree, spec_strip, extra, subst = sys.argv[1:11]
m = yaml.safe_load(open("/home/mp/.config/llama-swap/config.yaml"))["models"][name]
cmd = " ".join(m["cmd"].split()).replace("${PORT}", port).replace("/home/mp/vllm/venv/bin/python", venv + "/bin/python")
if "--pipeline-parallel-size 1" not in cmd and "--distributed-timeout-seconds" not in cmd:
    cmd += " --distributed-timeout-seconds 3600"
if spec_strip == "1":
    assert ',"attention_backend":"FLASH_ATTN"' in cmd, "erwartete Backend-Angabe im Spekulations-JSON fehlt"
    cmd = cmd.replace(',"attention_backend":"FLASH_ATTN"', "")
if extra:
    cmd += " " + extra
SUBST = {"v100mem": [("--max-model-len 262144", "--max-model-len 32768"), ("--gpu-memory-utilization 0.98", "--gpu-memory-utilization 0.90")],
         "pp2": [("--tensor-parallel-size 2 --pipeline-parallel-size 1", "--tensor-parallel-size 1 --pipeline-parallel-size 2 --distributed-timeout-seconds 3600")]}
for key in subst.split():
    for a, b in SUBST[key]:
        assert a in cmd, f"{key}: '{a}' nicht im Befehl"
        cmd = cmd.replace(a, b)
override = {"PATH": venv + "/bin:/home/mp/vllm/cuda/bin:/usr/local/bin:/usr/bin:/bin",
            "CUDA_VISIBLE_DEVICES": devs, "TORCH_CUDA_ARCH_LIST": arch,
            "VLLM_CACHE_ROOT": "/home/mp/.cache/vllm-e2e-fa2mixed", "PYTHONPATH": tree}
with open(out, "w") as f:
    f.write("#!/usr/bin/env bash\n")
    seen = set()
    for e in m.get("env", []):
        k, v = e.split("=", 1); seen.add(k)
        f.write(f"export {k}={shlex.quote(override.get(k, v))}\n")
    for k, v in override.items():
        if k not in seen: f.write(f"export {k}={shlex.quote(v)}\n")
    f.write("exec " + cmd + "\n")
PY
echo "== $ARM ($ENTRY auf $DEVS)"; date
setsid bash "$OUT/$ARM.launch.sh" > "$OUT/$ARM.boot.log" 2>&1 &
PID=$!; T0=$(date +%s); ST=timeout
for i in $(seq 1 720); do
  curl -sf -o /dev/null --max-time 2 http://127.0.0.1:$PORT/v1/models && { ST="up_$(( $(date +%s) - T0 ))s"; break; }
  kill -0 $PID 2>/dev/null || { ST=exited; break; }
  grep -q "Traceback (most recent call last)" "$OUT/$ARM.boot.log" && ! grep -q "Application startup complete" "$OUT/$ARM.boot.log" && sleep 20 && ! kill -0 $PID 2>/dev/null && { ST=traceback; break; }
  sleep 5
done
echo "STATUS $ST"
if [ "${ST#up_}" != "$ST" ]; then
  $PY - "http://127.0.0.1:$PORT" "$ENTRY" "$OUT/$ARM" <<'PY'
import hashlib, json, sys, time, urllib.request
url, model, out = sys.argv[1:4]
def ask(prompt, mt, tag):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": mt, "temperature": 0}
    t0 = time.time()
    with urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=3600) as r:
        d = json.load(r)
    dt = time.time() - t0; msg = d["choices"][0]["message"]; u = d["usage"]
    text = (msg.get("content") or "").strip()
    json.dump(d, open(f"{out}.{tag}.json", "w"), indent=1)
    print(f"{tag}: {u['completion_tokens']} tok in {dt:.1f}s ({u['completion_tokens']/dt:.1f} tok/s), prompt {u['prompt_tokens']}, finish={d['choices'][0]['finish_reason']}, sha {hashlib.sha256(text.encode()).hexdigest()[:16]}")
    print("   " + text[:220].replace("\n", " "))
ask("Erklaere in drei Saetzen, wie ein Regenbogen entsteht.", 400, "kurz")
ctx = open("/home/mp/Projekte/vllm-research/v100-skinny/tools/mtp-diagnostics/vorkontext.txt").read()
ask(ctx + "\n\nAufgabe, unabhaengig vom Hintergrundmaterial oben: Erklaere in drei Saetzen, wie ein Regenbogen entsteht.", 200, "lang")
PY
fi
echo "-- Lader-Zeilen je Worker:"; grep -a -o '([A-Za-z_0-9 ]*pid=[0-9]*) .*Loaded FA2 library [^ ]* for compute capability [0-9.]*' "$OUT/$ARM.boot.log" | sed 's/.*(\(.*\)) .*Loaded/\1: Loaded/' | sort -u
grep -a -c 'Loaded FA2 library' "$OUT/$ARM.boot.log" | sed 's/^/-- Lader-Zeilen gesamt: /'
grep -a -o 'Using [A-Z_]* attention backend' "$OUT/$ARM.boot.log" | sort | uniq -c | sed 's/^/-- /'
grep -a -m3 -E "Error:|Error\b|Traceback" "$OUT/$ARM.boot.log" | sed 's/.*\] //' | cut -c1-160
# Nur die eigene Prozessgruppe (setsid -> PGID = PID des Launchers), nie fremde vLLM-Server (llama-swap).
kill -TERM -- -$PID 2>/dev/null; sleep 15; kill -KILL -- -$PID 2>/dev/null; wait $PID 2>/dev/null
sleep 5; echo "-- VRAM danach: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
echo "FERTIG $ARM ($ST)"
