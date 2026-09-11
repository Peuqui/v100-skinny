#!/usr/bin/env bash
# Flash-Next-Abnahme (180B, TP2xPP2, heterogen): A/B des Full-Forward-Wrappers.
#   flashnext_qual.sh <name> <k>       k=0 heisst ohne Spekulation
#
# fix5    = Wrapper nach Geraet (Fix 5): Turing aus, Volta an
# wrapper = VLLM_SM70_QWEN_GDN_FULL_FORWARD=1, erzwingt ihn ueberall (Zustand
#           vor Fix 5). Eine Variable, sonst identisch.
#
# Setzt AIFRED_FORCE_UPSTREAM_GDN=1 -- ohne das fahren die Turing-Stufen das
# Fork-Modul und Fix 5 ist dort wirkungslos.
#
# AIFRED_STATE_FILE: die venv schreibt dort pro Rang Faehigkeit und
# Waechterzustand hin. NOETIG, weil vLLM INFO von Nebenraengen filtert -- eine
# fehlende Logmeldung beweist nichts.
set -uo pipefail
NAME=${1:?name fehlt}; K=${2:?k fehlt}
REPO=/home/mp/Projekte/vllm-research/v100-skinny
CKPT=/home/mp/models/Qwen3.8-Flash-Next-180B-A4B-NVFP4-MTPQ
# venv ueber VENV umschaltbar, wie in speed_dflash.sh: eine neu gebaute venv
# laesst sich so abnehmen, bevor der Produktions-Symlink umgestellt wird.
VENV=${VENV:-/home/mp/vllm/venv}
W=$HOME/.cache/mtp-diagnostics/fnq_$NAME; rm -rf "$W"; mkdir -p "$W"
PORT=8027

for i in $(seq 1 60); do
  u=$(nvidia-smi --id=0,1,2,3 --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)
  [ "${u:-0}" -le 800 ] && break; sleep 5
done
if pgrep -af 'api_server' | grep -v $$ | grep -q .; then echo "ABBRUCH: api_server laeuft"; exit 1; fi

unset VLLM_SM70_QWEN_GDN_FULL_FORWARD
export AIFRED_FORCE_UPSTREAM_GDN=1
export AIFRED_STATE_FILE=$W/state.txt   # ALLE Stufen auf Upstream, sonst faehrt Turing den Fork


cd $REPO
# Kalter Compile unter PP: Stufe 1 wartet in einer Kollektive, bis Stufe 0
# fertig kompiliert hat -- laenger als PyTorchs 600-s-NCCL-Wachhund und die
# 900 s des Serve-Skripts. Beide Werte begrenzen nur die Geduld.
 CUDA_VISIBLE_DEVICES=0,2,1,3 \
TURBOMIND=1 QUANT_BACKEND=turbomind \
ENV_PREFIX="$VENV" \
TP=2 PP=2 K=$K GMU=0.95 MML=262144 PP_PARTITION=24,24 PLE_HOST_GIB=6 \
PORT=$PORT LOG=$W/boot.log BOOT_WAIT_S=2400 \
EXTRA_ARGS='--distributed-timeout-seconds 3600 --compilation-config {"cudagraph_capture_sizes":[1,2,4,5,8]}' \
bash scripts/serve-qwen38-flash-next.sh "$CKPT" 2>&1 | tail -3
RC=$?
PID=$(cat $REPO/.flash-next.pid 2>/dev/null)

if curl -sf -o /dev/null --max-time 5 "http://127.0.0.1:$PORT/v1/models"; then
  "$VENV/bin/python" - "$W" "$PORT" "$REPO/tools/mtp-diagnostics" <<'PY2'
import hashlib, json, sys, time, urllib.request
W, PORT, SCR = sys.argv[1], sys.argv[2], sys.argv[3]
URL = f"http://127.0.0.1:{PORT}/v1/completions"
CTX = open(f"{SCR}/vorkontext.txt").read()
FRAGEN = [
    ("q1", "Erklaere die Quantenphysik in 30 Saetzen."),
    ("q2", "Erklaere den Regenbogeneffekt in 30 Saetzen."),
    ("q3", "Erklaere den Kuanda-Effekt in 30 Saetzen."),
]
def ask(p, mt):
    b = json.dumps({"model": "qwen3.8-flash-next", "prompt": p, "max_tokens": mt,
                    "temperature": 0, "seed": 1}).encode()
    r = urllib.request.Request(URL, data=b, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(r, timeout=3600) as resp:
        d = json.load(resp)
    return time.perf_counter() - t0, d
def spec():
    try:
        m = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=30).read().decode()
    except Exception:
        return None
    a = dr = n = 0.0
    for l in m.splitlines():
        if l.startswith("#"):
            continue
        if l.startswith("vllm:spec_decode_num_accepted_tokens_total"): a = float(l.rsplit(" ", 1)[1])
        elif l.startswith("vllm:spec_decode_num_draft_tokens_total"): dr = float(l.rsplit(" ", 1)[1])
        elif l.startswith("vllm:spec_decode_num_drafts_total"): n = float(l.rsplit(" ", 1)[1])
    return a, dr, n
ask("Guten Tag.", 8)
res, prev = {}, spec()
for tag, frage in FRAGEN:
    prompt = (CTX + "\n\nAufgabe, unabhaengig vom Hintergrundmaterial oben: "
              + frage + "\n\nAntwort:\n")
    t, d = ask(prompt, 1200)
    txt = d["choices"][0]["text"]; u = d["usage"]
    row = {"prompt_tokens": u["prompt_tokens"], "completion_tokens": u["completion_tokens"],
           "s": round(t, 2), "tok_per_s": round(u["completion_tokens"] / t, 2),
           "finish": d["choices"][0].get("finish_reason"),
           "sha256": hashlib.sha256(txt.encode()).hexdigest()[:16]}
    cur = spec()
    if prev and cur:
        da, dd, dr = (x - y for x, y in zip(cur, prev))
        if dd: row["acc_rate"] = round(da / dd, 4)
        if dr: row["acc_len"] = round(da / dr + 1, 3)
    prev = cur
    res[tag] = row
    open(f"{W}/text_{tag}.txt", "w").write(txt)
    print(f"  {tag}: {row}", flush=True)
open(W + "/result.json", "w").write(json.dumps(res, indent=2))
PY2
else
  echo "STATUS nicht_oben"; tail -5 $W/boot.log
fi

# Nur die eigene Prozessgruppe: das Serve-Skript startet den Server per
# setsid, die Worker haengen daran. Ein pgrep auf alle VLLM::-Prozesse
# wuerde auch fremde Server treffen (etwa ein Modell aus llama-swap).
[ -n "${PID:-}" ] && kill -TERM -$PID 2>/dev/null
sleep 20
[ -n "${PID:-}" ] && kill -KILL -$PID 2>/dev/null
echo "   NACHWEIS: armed=$(grep -c 'full-forward guard armed' $W/boot.log) sm75=$(grep -c 'qwen_gdn_linear_attn_sm75' $W/boot.log) upstream=$(grep -c 'cannot run on Turing' $W/boot.log)"
echo "   FA2: sm75=$(grep -c 'Loaded FA2 library _vllm_fa2_C_sm75' $W/boot.log) sm70=$(grep -c 'Loaded FA2 library _vllm_fa2_C.abi3' $W/boot.log) v100_backend=$(grep -c 'Using FLASH_ATTN_V100 attention backend' $W/boot.log) fa2_backend=$(grep -c 'Using FLASH_ATTN attention backend' $W/boot.log)"
echo "FERTIG fnq_$NAME"
