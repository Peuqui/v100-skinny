#!/usr/bin/env bash
# Flash-Next-Abnahme ueber den PRODUKTIONSPFAD: /v1/chat/completions mit dem
# Chat-Template des Checkpoints, enable_thinking=true (chat_template_kwargs wie
# AIfred), Reasoning-Parser qwen3 und Tool-Parser qwen3_coder wie im
# llama-swap-Eintrag. Denkblock landet im Feld `reasoning`, Antwort in `content`.
# Gegenstueck zu tools/mtp-diagnostics/flashnext_qual.sh (Rohtext-Sonde ohne
# Template, dort muss das Modell selbst entscheiden, ob es denkt).
#   flashnext_qual_chat.sh <name> <k>
# Server-Boot, Karten, Betriebspunkt und Abbau identisch zur Rohtext-Sonde.
set -uo pipefail
NAME=${1:?name fehlt}; K=${2:?k fehlt}
REPO=/home/mp/Projekte/vllm-research/v100-skinny
CKPT=/home/mp/models/Qwen3.8-Flash-Next-180B-A4B-NVFP4-MTPQ
VENV=${VENV:-/home/mp/vllm/venv}
W=$HOME/.cache/mtp-diagnostics/fnqc_$NAME; rm -rf "$W"; mkdir -p "$W"
PORT=8027
MAXTOK=${MAXTOK:-16000}  # Denkblock + 30 Saetze; bei MML 262144 kein Deckel mehr noetig (q3 brauchte 4.172 Token)

for i in $(seq 1 60); do
  u=$(nvidia-smi --id=0,1,2,3 --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)
  [ "${u:-0}" -le 800 ] && break; sleep 5
done
if pgrep -af 'api_server' | grep -v $$ | grep -q .; then echo "ABBRUCH: api_server laeuft"; exit 1; fi

unset VLLM_SM70_QWEN_GDN_FULL_FORWARD
export AIFRED_FORCE_UPSTREAM_GDN=1
export AIFRED_STATE_FILE=$W/state.txt

cd $REPO
 CUDA_VISIBLE_DEVICES=0,2,1,3 \
TURBOMIND=1 QUANT_BACKEND=turbomind \
ENV_PREFIX="$VENV" \
TP=2 PP=2 K=$K GMU=0.95 MML=262144 PP_PARTITION=24,24 PLE_HOST_GIB=6 \
PORT=$PORT LOG=$W/boot.log BOOT_WAIT_S=2400 \
EXTRA_ARGS='--distributed-timeout-seconds 3600 --compilation-config {"cudagraph_capture_sizes":[1,2,4,5,8]} --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3' \
bash scripts/serve-qwen38-flash-next.sh "$CKPT" 2>&1 | tail -3
PID=$(cat $REPO/.flash-next.pid 2>/dev/null)

if curl -sf -o /dev/null --max-time 5 "http://127.0.0.1:$PORT/v1/models"; then
  "$VENV/bin/python" - "$W" "$PORT" "$REPO/tools/mtp-diagnostics" "$MAXTOK" <<'PY2'
import hashlib, json, sys, time, urllib.request
W, PORT, SCR, MAXTOK = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
CTX = open(f"{SCR}/vorkontext.txt").read()
FRAGEN = [
    ("q1", "Erklaere die Quantenphysik in 30 Saetzen."),
    ("q2", "Erklaere den Regenbogeneffekt in 30 Saetzen."),
    ("q3", "Erklaere den Kuanda-Effekt in 30 Saetzen."),   # Verschreiber ABSICHTLICH, nicht aendern
]
def ask(user, mt):
    b = json.dumps({"model": "qwen3.8-flash-next", "max_tokens": mt, "temperature": 0, "seed": 1,
                    "messages": [{"role": "user", "content": user}],
                    "chat_template_kwargs": {"enable_thinking": True}}).encode()
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
        if l.startswith("vllm:spec_decode_num_accepted_tokens_total"): a = float(l.rsplit(" ", 1)[1])
        elif l.startswith("vllm:spec_decode_num_draft_tokens_total"): dr = float(l.rsplit(" ", 1)[1])
        elif l.startswith("vllm:spec_decode_num_drafts_total"): n = float(l.rsplit(" ", 1)[1])
    return a, dr, n
ask("Guten Tag.", 8)
res, prev = {}, spec()
for tag, frage in FRAGEN:
    user = CTX + "\n\nAufgabe, unabhaengig vom Hintergrundmaterial oben: " + frage
    t, d = ask(user, MAXTOK)
    open(f"{W}/raw_{tag}.json", "w").write(json.dumps(d, ensure_ascii=False, indent=1))
    msg = d["choices"][0]["message"]; u = d["usage"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    row = {"prompt_tokens": u["prompt_tokens"], "completion_tokens": u["completion_tokens"],
           "s": round(t, 2), "tok_per_s": round(u["completion_tokens"] / t, 2),
           "finish": d["choices"][0].get("finish_reason"),
           "reasoning_chars": len(reasoning), "content_chars": len(content),
           "sha256": hashlib.sha256(content.encode()).hexdigest()[:16]}
    cur = spec()
    if prev and cur:
        da, dd, dr = (x - y for x, y in zip(cur, prev))
        if dd: row["acc_rate"] = round(da / dd, 4)
        if dr: row["acc_len"] = round(da / dr + 1, 3)
    prev = cur
    res[tag] = row
    open(f"{W}/text_{tag}.txt", "w").write(f"<think>{reasoning}</think>\n{content}")
    print(f"  {tag}: {row}", flush=True)
open(W + "/result.json", "w").write(json.dumps(res, indent=2))
PY2
else
  echo "STATUS nicht_oben"; tail -5 $W/boot.log
fi

[ -n "${PID:-}" ] && kill -TERM -$PID 2>/dev/null
sleep 20
[ -n "${PID:-}" ] && kill -KILL -$PID 2>/dev/null
echo "   PARSER: $(grep -o 'reasoning_parser=[^,)]*\|tool_call_parser=[^,)]*\|Using reasoning parser[^\n]\{0,40\}' $W/boot.log | sort -u | tr '\n' ' ')"
echo "FERTIG fnqc_$NAME"
