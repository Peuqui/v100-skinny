#!/usr/bin/env bash
# DFlash2 gegen MTP auf demselben Ziel (Qwen3.8-27B-NVFP4).
#   speed_dflash.sh <name> <fork|upstream> <dflash|mtp|0>
#
# dflash = incoai/Qwen3.8-27B-DFlash2 als Entwurfskopf, 7 Token je Block
#          (block_size 8 - 1, aus dflash_config des Checkpoints)
# mtp    = der eingebaute MTP-Kopf mit k=3, unsere bisherige Referenz
# 0      = ohne Spekulation
# Kartenwahl ueber DEVS (Vorgabe 0,2 = RTX-Paar; 1,3 = V100-Paar).
set -uo pipefail
NAME=${1:?name fehlt}; MODULE=${2:?fork|upstream fehlt}; MODE=${3:?dflash|mtp|0 fehlt}
DEVS=${DEVS:-0,2}
BASE=${WORKDIR:-$HOME/.cache/mtp-diagnostics}
W=$BASE/qual_$NAME; rm -rf "$W"; mkdir -p "$W"; cd "$BASE" || exit 1

# venv ueber VENV umschaltbar: eine neu gebaute venv laesst sich so abnehmen,
# bevor /home/mp/vllm/venv auf sie zeigt. Vorgabe bleibt der Symlink.
VENV=${VENV:-/home/mp/vllm/venv}
export PATH="$VENV/bin":/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_HOME=/home/mp/vllm/cuda TORCH_CUDA_ARCH_LIST=7.0
export NCCL_P2P_DISABLE=1 VLLM_SM70_E5_CACHE=0 VLLM_SM70_NVFP4_TURBOMIND=1
# QUANT_BACKEND ueberschreibbar: auf Turing waehlt "auto" den Skinny-Pfad
# (skinny_nvfp4_qpn2, 1614 ms im Decode-Profil 09.09.), waehrend die V100
# ueber TurboMind laeuft (gemm_kernel, 1396 ms). "marlin" ist der dritte,
# auf Turing nachweislich vorhandene Pfad -- Vorgabe bleibt auto.
export VLLM_SM70_QUANT_BACKEND=${VLLM_SM70_QUANT_BACKEND:-auto}
# Die Skinny-Schalter ebenfalls ueberschreibbar: sie haengen NICHT am
# Quant-Backend und greifen bei kleinem M -- also genau im Decode. Ein
# Marlin-Vergleich ohne SKINNY=0 misst weiter den Skinny-Pfad (Fehllauf
# 09.09.: Boot meldete trotz QUANT_BACKEND=marlin "SM70 skinny NVFP4 path
# enabled for M<=64"). Vorgaben bleiben 1.
export VLLM_SKINNY_NVFP4=${VLLM_SKINNY_NVFP4:-1}
export VLLM_SKINNY_QPN=${VLLM_SKINNY_QPN:-1} VLLM_SKINNY_QPN2=${VLLM_SKINNY_QPN2:-1}
export VLLM_SKINNY_NVFP4_SRC=/home/mp/Projekte/v100-skinny/kernels/skinny_kernels.cu
export TORCHINDUCTOR_CACHE_DIR=/home/mp/.cache/torchinductor VLLM_NO_USAGE_STATS=1
export VLLM_CACHE_ROOT=/home/mp/.cache/vllm-calibration HOME=/home/mp
export CUDA_VISIBLE_DEVICES=$DEVS
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS=1
unset VLLM_SKINNY_PPDIAG
[ "$MODULE" = "upstream" ] && export AIFRED_FORCE_UPSTREAM_GDN=1
export VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH=1
# Seit dem Upstream-Stand vom 10.09. bricht DFlash2 mit quantisiertem
# Ziel-LM-Head ab (RadixArk quantisiert ihn mit). Der Schalter ist 1Cats
# Opt-in fuer die Kandidaten-TopK ueber dichte Logits -- derselbe Rechenweg,
# den aeltere venvs ohne Abfrage nehmen; die kennen den Schalter nicht.
export VLLM_SM70_DFLASH2_QUANT_LM_HEAD=${VLLM_SM70_DFLASH2_QUANT_LM_HEAD:-1}

# Ziel ueber CKPT umschaltbar: RadixArk quantisiert den lm_head mit, 1Cats
# DFlash2-Referenz QUASAR-QAT nimmt ihn per ignore-Liste aus. Vorgabe bleibt
# RadixArk, damit alle bisherigen Zahlen vergleichbar bleiben.
CKPT=${CKPT:-/home/mp/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/319f741cce68d7914884900c138a1fbb70a42f30}
# Entwurfskopf ueber DRAFT umschaltbar, wie in prof_dflash.sh. Vorgabe bleibt
# der unquantisierte incoai-Kopf, damit alle bisherigen Zahlen vergleichbar
# bleiben. Ein quantisierter Kopf aendert die Annahmerate, aber NIE den Text:
# angenommen wird nur, was das Zielmodell ohnehin erzeugt haette.
DRAFT=${DRAFT:-/home/mp/.cache/huggingface/hub/models--incoai--Qwen3.8-27B-DFlash2/snapshots/dedf8df68adfb1afeaf7b7480c0a0243108177b4}
SPEC=()
case "$MODE" in
  mtp)    SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":3,\"draft_sample_method\":\"greedy\"}") ;;
  dflash) SPEC=(--speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":7,\"draft_sample_method\":\"greedy\"}") ;;
  0)      SPEC=() ;;
  *)      echo "unbekannter MODE: $MODE"; exit 1 ;;
esac

if pgrep -af 'api_server' | grep -v $$ | grep -q .; then echo "ABBRUCH: api_server laeuft"; exit 1; fi
used=$(nvidia-smi --id=$DEVS --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)
[ "${used:-0}" -gt 500 ] && { echo "ABBRUCH: $used MiB belegt"; exit 1; }

# SM70TUNE=1 holt die Vorgaben nach, die Turing entgehen, weil
# VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH dort verworfen wird ("not SM70
# CUDA"). Gemessen 09.09. per nsys: die V100 faehrt rms_norm=['vllm_c'] und
# fuse_norm_quant=True, Turing ['native'] und False. Vorgabe bleibt AUS, damit
# alle bisherigen Zahlen vergleichbar bleiben.
COMPCFG='{"cudagraph_capture_sizes":[1,2,4,8]}'
TUNE=()
if [ "${SM70TUNE:-0}" = "1" ]; then
  COMPCFG='{"cudagraph_capture_sizes":[1,2,4,8],"pass_config":{"fuse_norm_quant":true}}'
  TUNE=(--kernel-config '{"ir_op_priority":{"rms_norm":["vllm_c","native"],"fused_add_rms_norm":["vllm_c","native"]}}')
fi

"$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$CKPT" --served-model-name m --trust-remote-code --dtype float16 \
  --disable-custom-all-reduce --no-enable-prefix-caching \
  --tensor-parallel-size 2 --pipeline-parallel-size 1 --gpu-memory-utilization 0.90 \
  --block-size 16 --max-model-len 32768 --max-num-seqs 4 --max-num-batched-tokens 2048 \
  --language-model-only --host 127.0.0.1 --port 8066 "${SPEC[@]}" "${TUNE[@]}" \
  --compilation-config "$COMPCFG" > "$W/boot.log" 2>&1 &
S=$!
STATUS=timeout
for i in $(seq 1 150); do
  sleep 5
  kill -0 $S 2>/dev/null || { STATUS=exited; break; }
  grep -q "Application startup complete" "$W/boot.log" && { STATUS="up_$((i*5))s"; break; }
  grep -qi "Traceback (most recent call last)" "$W/boot.log" && { STATUS=traceback; break; }
done
echo "STATUS $STATUS  ($NAME: $MODULE, $MODE, DEVS=$DEVS)"

if [ "${STATUS#up_}" != "$STATUS" ]; then
  "$VENV/bin/python" - "$W" <<'PY'
import hashlib, json, sys, time, urllib.request
W = sys.argv[1]
URL = "http://127.0.0.1:8066/v1/completions"
PROMPT = "Erklaere die Quantenphysik in 20 Saetzen."
REPS = 5
N_TOK = 400

def ask(mt):
    body = json.dumps({"model": "m", "prompt": PROMPT, "max_tokens": mt,
                       "temperature": 0, "seed": 1}).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    return time.perf_counter() - t0, d

def spec_counters():
    try:
        met = urllib.request.urlopen("http://127.0.0.1:8066/metrics",
                                     timeout=30).read().decode()
    except Exception:
        return None
    acc = dft = drafts = 0.0
    for line in met.splitlines():
        if line.startswith("#"):
            continue
        if line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            acc = float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:spec_decode_num_draft_tokens_total"):
            dft = float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:spec_decode_num_drafts_total"):
            drafts = float(line.rsplit(" ", 1)[1])
    return acc, dft, drafts

ask(8)                                                # warmlaufen
rows, txt, prev = [], None, spec_counters()
for i in range(REPS):
    t, d = ask(N_TOK)
    txt = d["choices"][0]["text"]
    comp = d["usage"]["completion_tokens"]
    row = {"tok": comp, "s": round(t, 3), "tok_per_s": round(comp / t, 2)}
    cur = spec_counters()
    if prev and cur:
        da, dd, dr = (c - p for c, p in zip(cur, prev))
        if dd:
            row["acc_rate"] = round(da / dd, 4)
        if dr:
            row["acc_len"] = round(da / dr + 1, 3)
    prev = cur
    rows.append(row)
    print(f"  #{i+1}: {row}", flush=True)
sp = sorted(r["tok_per_s"] for r in rows)
med = sp[len(sp) // 2]
al = [r["acc_len"] for r in rows if "acc_len" in r]
print(f"MEDIAN {med:.2f} tok/s  Spanne {min(sp):.2f}-{max(sp):.2f}"
      + (f"  Annahmelaenge {sum(al)/len(al):.3f}" if al else ""))
open(f"{W}/text_q1.txt", "w").write(txt or "")
res = {"rows": rows, "median_tok_per_s": med,
       "sha256": hashlib.sha256((txt or "").encode()).hexdigest()[:16]}
open(W + "/result.json", "w").write(json.dumps(res, indent=2))
PY
fi
for p in $(pgrep -f 'VLLM[:]:'); do kill -TERM $p 2>/dev/null; done
kill -TERM $S 2>/dev/null; sleep 12
for p in $(pgrep -f 'VLLM[:]:'); do kill -KILL $p 2>/dev/null; done
kill -KILL $S 2>/dev/null; wait $S 2>/dev/null
echo "FERTIG qual_$NAME ($STATUS)"
