#!/usr/bin/env bash
# QUASAR-NVFP4 unter 1Cats eigenem Produktionsprofil nachfahren.
#   quasar_1cat.sh <name> <dflash|0>
#
# Zweck: Unser speed_dflash.sh fuhr QUASAR mit dem RadixArk-Profil, und dabei
# degenerierte die Ausgabe in eine Wiederholungsschleife. Dieses Skript aendert
# genau eine Sache -- es stellt 1Cats validierte Bedingungen her (RELEASE.md,
# "Recommended QUASAR NVFP4 + DFlash2 serving example"), damit sich Checkpoint
# und Konfiguration als Ursache trennen lassen.
#
# Unterschiede zu speed_dflash.sh, alle aus 1Cats Aufrufzeile:
#   FLASH_ATTN_V100 statt Auto-Backend, KV fp8_e5m2 statt auto, Prefix-Caching
#   AN, mamba-cache-mode align, Chat-Endpoint mit enable_thinking und
#   Reasoning-Parser statt rohem /v1/completions, draft_sample_method
#   probabilistic statt greedy, 4096 statt 2048 Batch-Token, GMU 0.80.
# Bewusst NICHT gesetzt: unsere VLLM_SM70_*-Schalter. 1Cat faehrt die Defaults
# des Forks; jeder eigene Schalter waere eine zweite geaenderte Variable.
#
# TP2 statt 1Cats TP4: QUASAR laedt auf den RTX-Karten nicht (Marlin verlangt
# Ausgabebreiten als Vielfache von 64, eine Schicht hat 8240), und frei sind
# nur drei V100. Fuer die Qualitaetsfrage ist der TP-Grad nicht die Variable.
#
# --disable-custom-all-reduce steht NICHT in 1Cats Zeile, ist hier aber
# zwingend: auf diesem Rechner ist P2P aus (Karten an OCuLink/USB4, siehe
# STAND.md), und der Custom-Allreduce-Kernel setzt direkte GPU-zu-GPU-Zugriffe
# voraus. Ohne das Flag warten beide TP-Raenge im Allreduce -- 100% GPU-Last
# bei 46 W, kein Fortschritt (09.09., zwei Laeufe verloren). 1Cat faehrt vier
# V100 in einem Server mit funktionierendem P2P.
#
# ACHTUNG Prefix-Caching ist AN, wie bei 1Cat -- die zweite Anfrage trifft den
# Cache. Dieses Skript beantwortet die QUALITAETSfrage, nicht die Tempofrage.
set -uo pipefail
NAME=${1:?name fehlt}; MODE=${2:?dflash|0 fehlt}
DEVS=${DEVS:-1,3}
BASE=${WORKDIR:-$HOME/.cache/mtp-diagnostics}
W=$BASE/qual_$NAME; rm -rf "$W"; mkdir -p "$W"; cd "$BASE" || exit 1

export PATH=/home/mp/vllm/venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_HOME=/home/mp/vllm/cuda
export TORCHINDUCTOR_CACHE_DIR=/home/mp/.cache/torchinductor VLLM_NO_USAGE_STATS=1
export VLLM_CACHE_ROOT=/home/mp/.cache/vllm-calibration HOME=/home/mp
export CUDA_VISIBLE_DEVICES=$DEVS
export NCCL_P2P_DISABLE=1

CKPT=${CKPT:-/home/mp/.cache/huggingface/hub/models--QUASAR-QAT--Qwen3.8-27B-QUASAR-NVFP4/snapshots/d8e6fbfa3e3a78899b440222b827430045a05b44}
DRAFT=${DRAFT:-/home/mp/.cache/huggingface/hub/models--incoai--Qwen3.8-27B-DFlash2/snapshots/dedf8df68adfb1afeaf7b7480c0a0243108177b4}
MML=${MML:-262144}
SPEC=()
case "$MODE" in
  dflash) SPEC=(--speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFT\",\"kv_cache_dtype\":\"auto\",\"draft_sample_method\":\"probabilistic\"}") ;;
  0)      SPEC=() ;;
  *)      echo "unbekannter MODE: $MODE"; exit 1 ;;
esac

if pgrep -af 'api_server' | grep -v $$ | grep -q .; then echo "ABBRUCH: api_server laeuft"; exit 1; fi
used=$(nvidia-smi --id=$DEVS --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)
[ "${used:-0}" -gt 500 ] && { echo "ABBRUCH: $used MiB belegt"; exit 1; }

/home/mp/vllm/venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$CKPT" --served-model-name m --trust-remote-code --dtype half \
  --tensor-parallel-size 2 \
  --disable-custom-all-reduce \
  --attention-backend FLASH_ATTN_V100 \
  --kv-cache-dtype fp8_e5m2 \
  --max-model-len "$MML" \
  --gpu-memory-utilization 0.80 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 4 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking":true}' \
  --host 127.0.0.1 --port 8066 "${SPEC[@]}" > "$W/boot.log" 2>&1 &
S=$!
STATUS=timeout
# 40 min: bei kaltem Inductor-Cache autotunt Triton den Draftkopf beim ersten
# Forward, das allein lief am 09.09. ueber 12 min (py-spy: benchmark_all_configs
# aus qwen3_dflash.py forward). 15 min haben einen arbeitenden Lauf getoetet.
for i in $(seq 1 480); do
  sleep 5
  kill -0 $S 2>/dev/null || { STATUS=exited; break; }
  grep -q "Application startup complete" "$W/boot.log" && { STATUS="up_$((i*5))s"; break; }
  grep -qi "Traceback (most recent call last)" "$W/boot.log" && { STATUS=traceback; break; }
done
echo "STATUS $STATUS  ($NAME: QUASAR/1Cat-Profil, $MODE, DEVS=$DEVS)"

if [ "${STATUS#up_}" != "$STATUS" ]; then
  /home/mp/vllm/venv/bin/python - "$W" <<'PY'
import json, sys, time, urllib.request
W = sys.argv[1]
URL = "http://127.0.0.1:8066/v1/chat/completions"
# Die etablierte Sonde: drei Fragen im selben Kontext, die dritte mit dem
# ABSICHTLICHEN Schreibfehler "Kuanda" (gemeint ist Coanda). Nicht korrigieren.
FRAGEN = [
    "Erkläre die Quantenphysik in 30 Sätzen.",
    "Erkläre den Regenbogeneffekt in 30 Sätzen.",
    "Erkläre den Kuanda-Effekt in 30 Sätzen.",
]

def ask(messages, mt):
    body = json.dumps({"model": "m", "messages": messages, "max_tokens": mt,
                       "temperature": 0}).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    return time.perf_counter() - t0, d

msgs, rows = [], []
for i, frage in enumerate(FRAGEN, 1):
    msgs.append({"role": "user", "content": frage})
    # Das Modell denkt lang: 1600 Token reichten am 09.09. nicht einmal fuer
    # den Denkblock, die Antwort kam nie und die History bekam einen leeren
    # Assistant-Turn -- danach antwortete das Modell nur noch mit zwei Zeichen.
    t, d = ask(msgs, 6000)
    # Die ROHE Antwort zuerst wegschreiben: content/reasoning_content sind nicht
    # die einzigen Felder, in denen Text landen kann (Tool-Parser!). Am 09.09.
    # gingen 1600 erzeugte Token verloren, weil hier nur zwei Felder gelesen
    # wurden und beide leer waren.
    open(f"{W}/roh_q{i}.json", "w").write(
        json.dumps(d, indent=2, ensure_ascii=False))
    msg = d["choices"][0]["message"]
    text = msg.get("content") or ""
    # Das Feld heisst "reasoning", NICHT "reasoning_content" (09.09. geprueft
    # an der Rohantwort). Beide lesen, falls eine Version es anders benennt.
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    comp = d["usage"]["completion_tokens"]
    finish = d["choices"][0].get("finish_reason")
    ntools = len(msg.get("tool_calls") or [])
    msgs.append({"role": "assistant", "content": text})
    rows.append({"frage": i, "tok": comp, "s": round(t, 2),
                 "tok_per_s": round(comp / t, 2),
                 "antwort_zeichen": len(text), "denk_zeichen": len(reasoning),
                 "tool_calls": ntools, "ende": finish})
    print(f"  q{i}: {rows[-1]}", flush=True)
    open(f"{W}/antwort_q{i}.txt", "w").write(text)
    open(f"{W}/denken_q{i}.txt", "w").write(reasoning)
open(W + "/result.json", "w").write(json.dumps({"rows": rows}, indent=2))
print("Texte liegen in", W, "-- MUESSEN gelesen werden, Zaehler beweisen nichts")
PY
fi
for p in $(pgrep -f 'VLLM[:]:'); do kill -TERM $p 2>/dev/null; done
kill -TERM $S 2>/dev/null; sleep 12
for p in $(pgrep -f 'VLLM[:]:'); do kill -KILL $p 2>/dev/null; done
kill -KILL $S 2>/dev/null; wait $S 2>/dev/null
echo "FERTIG qual_$NAME ($STATUS)"
