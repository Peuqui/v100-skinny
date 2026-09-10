#!/usr/bin/env bash
# Flash-Next-Abnahme (180B, TP2xPP2, heterogen): A/B des Full-Forward-Wrappers.
#   flashnext_ab.sh <name> <fix5|wrapper>
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
NAME=${1:?name fehlt}; MODE=${2:?fix5|wrapper fehlt}
REPO=/home/mp/Projekte/vllm-research/v100-skinny
CKPT=/home/mp/models/Qwen3.8-Flash-Next-180B-A4B-NVFP4-MTPQ
W=$HOME/.cache/mtp-diagnostics/fn_$NAME; rm -rf "$W"; mkdir -p "$W"
PORT=8027

for i in $(seq 1 60); do
  u=$(nvidia-smi --id=0,1,2,3 --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)
  [ "${u:-0}" -le 800 ] && break; sleep 5
done
if pgrep -af 'api_server' | grep -v $$ | grep -q .; then echo "ABBRUCH: api_server laeuft"; exit 1; fi

unset VLLM_SM70_QWEN_GDN_FULL_FORWARD
export AIFRED_FORCE_UPSTREAM_GDN=1
export AIFRED_STATE_FILE=$W/state.txt   # ALLE Stufen auf Upstream, sonst faehrt Turing den Fork
[ "$MODE" = "wrapper" ] && export VLLM_SM70_QWEN_GDN_FULL_FORWARD=1

cd $REPO
VLLM_SM70_E5_CACHE=0 CUDA_VISIBLE_DEVICES=0,2,1,3 \
TURBOMIND=1 QUANT_BACKEND=turbomind \
ENV_PREFIX=/home/mp/vllm/venv \
TP=2 PP=2 K=4 GMU=0.95 MML=16384 PP_PARTITION=24,24 PLE_HOST_GIB=6 \
PORT=$PORT LOG=$W/boot.log \
EXTRA_ARGS='--compilation-config {"cudagraph_capture_sizes":[1,2,4,5,8]}' \
bash scripts/serve-qwen38-flash-next.sh "$CKPT" 2>&1 | tail -3
RC=$?
PID=$(cat $REPO/.flash-next.pid 2>/dev/null)

if curl -sf -o /dev/null --max-time 5 "http://127.0.0.1:$PORT/v1/models"; then
  /home/mp/vllm/venv/bin/python - "$W" "$PORT" <<'PY'
import hashlib, json, sys, time, urllib.request
W, PORT = sys.argv[1], sys.argv[2]
URL = f"http://127.0.0.1:{PORT}/v1/completions"
def ask(p, mt):
    b=json.dumps({"model":"qwen3.8-flash-next","prompt":p,"max_tokens":mt,
                  "temperature":0,"seed":1}).encode()
    r=urllib.request.Request(URL,data=b,headers={"Content-Type":"application/json"})
    t0=time.perf_counter()
    with urllib.request.urlopen(r,timeout=1800) as resp: d=json.load(resp)
    return time.perf_counter()-t0, d
def spec():
    try: m=urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics",timeout=30).read().decode()
    except Exception: return None
    a=dr=n=0.0
    for l in m.splitlines():
        if l.startswith("#"): continue
        if l.startswith("vllm:spec_decode_num_accepted_tokens_total"): a=float(l.rsplit(" ",1)[1])
        elif l.startswith("vllm:spec_decode_num_draft_tokens_total"): dr=float(l.rsplit(" ",1)[1])
        elif l.startswith("vllm:spec_decode_num_drafts_total"): n=float(l.rsplit(" ",1)[1])
    return a,dr,n
ask("Guten Tag.",8)
prev=spec(); rows=[]; txt=""
for i in range(3):
    t,d=ask("Erklaere die Quantenphysik in 20 Saetzen.",300)
    txt=d["choices"][0]["text"]; c=d["usage"]["completion_tokens"]
    row={"tok":c,"s":round(t,3),"tok_per_s":round(c/t,2)}
    cur=spec()
    if prev and cur:
        da,dd,dr=(x-y for x,y in zip(cur,prev))
        if dd: row["acc_rate"]=round(da/dd,4)
        if dr: row["acc_len"]=round(da/dr+1,3)
    prev=cur; rows.append(row); print(f"  #{i+1}: {row}",flush=True)
sp=sorted(r["tok_per_s"] for r in rows); med=sp[len(sp)//2]
al=[r["acc_len"] for r in rows if "acc_len" in r]
print(f"MEDIAN {med:.2f} tok/s  Spanne {min(sp):.2f}-{max(sp):.2f}"+(f"  Annahmelaenge {sum(al)/len(al):.3f}" if al else ""))
open(f"{W}/text.txt","w").write(txt)
open(f"{W}/result.json","w").write(json.dumps({"rows":rows,"median_tok_per_s":med,
    "sha256":hashlib.sha256(txt.encode()).hexdigest()[:16]},indent=2))
PY
else
  echo "STATUS nicht_oben (rc=$RC)"; tail -5 $W/boot.log
fi

[ -n "${PID:-}" ] && kill -TERM -$PID 2>/dev/null
for p in $(pgrep -f 'VLLM[:]:'); do kill -TERM $p 2>/dev/null; done
sleep 20
[ -n "${PID:-}" ] && kill -KILL -$PID 2>/dev/null
for p in $(pgrep -f 'VLLM[:]:'); do kill -KILL $p 2>/dev/null; done
echo "   NACHWEIS: armed=$(grep -c 'full-forward guard armed' $W/boot.log) route=$(grep -c 'full-forward route enabled' $W/boot.log) sm75-Modul=$(grep -c 'qwen_gdn_linear_attn_sm75' $W/boot.log) upstream=$(grep -c 'cannot run on Turing' $W/boot.log)"
echo "   ZUSTAND je Rang:"; sort -u $W/state.txt 2>/dev/null | sed "s/^/     /"
echo "FERTIG fn_$NAME"
