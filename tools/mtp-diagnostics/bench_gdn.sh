#!/usr/bin/env bash
# GDN-Kernelvergleich auf Volta: FlashQLA-SM70 gegen Triton/FLA.
# Misst Prefill (langer Prompt, 1 Token Ausgabe) und Decode getrennt.
#
#   bench_gdn.sh flashqla   # Volta-Default: FlashQLA fuer Prefill UND Decode
#   bench_gdn.sh triton     # erzwungen Triton/FLA fuer beides (= Turing-Lage)
#
# Prefix-Caching ist AUS: sonst trifft die zweite Anfrage den Cache und der
# Prefill wird uebersprungen - die Zahl waere wertlos.
set -uo pipefail
MODE=${1:?mode fehlt: flashqla|triton}

BASE=${WORKDIR:-$HOME/.cache/mtp-diagnostics}
W=$BASE/runs/bench_$MODE
rm -rf "$W"; mkdir -p "$W"
cd "$BASE" || exit 1

export PATH=/home/mp/vllm/venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_HOME=/home/mp/vllm/cuda TORCH_CUDA_ARCH_LIST=7.0
export NCCL_P2P_DISABLE=1 VLLM_SM70_NVFP4_TURBOMIND=1
export VLLM_SM70_QUANT_BACKEND=auto VLLM_SKINNY_NVFP4=1 VLLM_SKINNY_QPN=1 VLLM_SKINNY_QPN2=1
export VLLM_SKINNY_NVFP4_SRC=/home/mp/Projekte/v100-skinny/kernels/skinny_kernels.cu
export TORCHINDUCTOR_CACHE_DIR=/home/mp/.cache/torchinductor VLLM_NO_USAGE_STATS=1
export VLLM_CACHE_ROOT=/home/mp/.cache/vllm-calibration HOME=/home/mp
export CUDA_VISIBLE_DEVICES=1,3            # beide V100 - nur dort gibt es FlashQLA
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS=1

ADDCFG='{}'
if [ "$MODE" = "triton" ]; then
  export VLLM_SM70_GDN_DECODE_FLASHQLA=0
  ADDCFG='{"gdn_prefill_backend":"triton"}'
fi

CKPT=/home/mp/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/319f741cce68d7914884900c138a1fbb70a42f30

if pgrep -af 'api_server' | grep -v $$ | grep -q .; then
  echo "ABBRUCH: laeuft schon ein api_server"; exit 1
fi
used=$(nvidia-smi --id=1,3 --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)
[ "${used:-0}" -gt 500 ] && { echo "ABBRUCH: $used MiB auf GPU 1/3 belegt"; exit 1; }

/home/mp/vllm/venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$CKPT" --served-model-name m --trust-remote-code --dtype float16 \
  --disable-custom-all-reduce --no-enable-prefix-caching \
  --tensor-parallel-size 2 --pipeline-parallel-size 1 --gpu-memory-utilization 0.90 \
  --block-size 16 --max-model-len 32768 --max-num-seqs 4 --max-num-batched-tokens 2048 \
  --language-model-only --host 127.0.0.1 --port 8065 \
  --additional-config "$ADDCFG" \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"greedy"}' \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8]}' > "$W/boot.log" 2>&1 &
S=$!

STATUS=timeout
for i in $(seq 1 120); do
  sleep 5
  kill -0 $S 2>/dev/null || { STATUS=exited; break; }
  grep -q "Application startup complete" "$W/boot.log" && { STATUS="up_$((i*5))s"; break; }
  grep -qi "Traceback (most recent call last)" "$W/boot.log" && { STATUS=traceback; break; }
done
echo "STATUS $STATUS"
echo "--- gewaehlte Kernel ---"
grep -iE "GDN prefill kernel|FlashQLA GDN decode|Auto-setting VLLM_SM70_GDN" "$W/boot.log" | sed 's/.*INFO[^]]*\] //' | head -4

if [ "${STATUS#up_}" != "$STATUS" ]; then
  /home/mp/vllm/venv/bin/python - "$W" <<'PY'
import json, sys, time, urllib.request

W = sys.argv[1]
URL = "http://127.0.0.1:8065/v1/completions"

def ask(prompt, max_tokens):
    body = json.dumps({"model": "m", "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0, "seed": 1}).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    return time.perf_counter() - t0, d["usage"]

# Langer, nicht-repetitiver Prompt: Prefill soll dominieren. Jeder Abschnitt
# traegt eine andere Zahl, damit kein Block dedupliziert werden kann.
para = ("Abschnitt {i}: Die Messung der Verarbeitungsgeschwindigkeit grosser "
        "Sprachmodelle haengt von vielen Faktoren ab, darunter die Belegung des "
        "Zwischenspeichers, die Bandbreite zwischen den Beschleunigern und die "
        "Auswahl der Rechenkerne fuer die einzelnen Schichten. ")
long_prompt = "".join(para.format(i=i) for i in range(260))

results = {}
# Warmlaufen (nicht gewertet)
ask("Kurzer Aufwaermsatz.", 4)

t_prefill, u1 = ask(long_prompt, 1)
t_total, u2 = ask(long_prompt + " Fortsetzung.", 96)

prompt_toks = u1["prompt_tokens"]
gen_toks = u2["completion_tokens"]
decode_s = max(t_total - t_prefill, 1e-6)

results = {
    "prompt_tokens": prompt_toks,
    "prefill_s": round(t_prefill, 3),
    "prefill_tok_per_s": round(prompt_toks / t_prefill, 1),
    "generated_tokens": gen_toks,
    "total_s_96tok": round(t_total, 3),
    "decode_tok_per_s": round((gen_toks - 1) / decode_s, 2),
}
print(json.dumps(results, indent=2))
open(W + "/bench.json", "w").write(json.dumps(results, indent=2))
PY
fi

for p in $(pgrep -f 'VLLM[:]:'); do kill -TERM $p 2>/dev/null; done
kill -TERM $S 2>/dev/null; sleep 12
for p in $(pgrep -f 'VLLM[:]:'); do kill -KILL $p 2>/dev/null; done
kill -KILL $S 2>/dev/null
echo "FERTIG bench_$MODE ($STATUS)"
