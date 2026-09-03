#!/usr/bin/env bash
# Boot-Versuch: Qwen3.8-Flash-Next (Qwen4Exp) NVFP4 auf 2x RTX 8000 + 2x V100.
#
# Anders als serve-qwen38-mini.sh ist dies KEIN Gate-Skript: der Port ist noch
# nicht lauffähig, das Skript startet den Server und meldet, wo er stirbt.
# Es existiert, damit die Boot-Umgebung reproduzierbar ist und nicht bei jedem
# Versuch neu aus der Kommandozeile zusammengesetzt wird.
#
#   bash scripts/serve-qwen38-flash-next.sh <checkpoint-dir>
#
# --language-model-only laesst den Vision-Tower weg (1,1 GB) und umgeht den
# Zweig, der `_init_video_pruning` ruft -- eine Methode, die das PR-Modell von
# einer upstream-Basisklasse erwartet, die 1Cat nicht hat. Fuer Text-Betrieb
# ist das die vorgesehene Konfiguration, kein Workaround.
#
# PLE_HOST_GIB legt einen Teil der PLE-Tabelle je Rang in den Host-RAM (UVA,
# zero-copy ueber PCIe). Die Tabelle ist 51,2 GB gross, traegt aber nur 2,5 KB
# Verkehr pro Token -- im VRAM verdraengt sie den KV-Cache und damit den
# nutzbaren Kontext.
#   auto (Default) = so viel im VRAM lassen wie moeglich und nur das auslagern,
#                    was MML an KV-Cache braucht
#   <zahl>         = feste GiB je Rang
#   0              = alles im VRAM (Stand vor dieser Kaskade)
#
# Überschreibbar: ENV_PREFIX TP PP K GMU MML MNS MBT PORT PP_PARTITION LOG
#                 PLE_HOST_GIB
#                 EXTRA_ARGS (zusätzliche Argumente für den Server)
#                 SPEC_CONFIG (kompletter --speculative-config-JSON; ersetzt den
#                 aus K gebauten Default, z.B. um enforce_eager nur für den
#                 Drafter zu setzen)
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="${1:-}"
[ -n "$CKPT" ] || { echo "usage: $0 <checkpoint-dir>" >&2; exit 2; }
[ -f "$CKPT/config.json" ] || { echo "ERROR: no config.json in $CKPT" >&2; exit 2; }
CKPT="$(cd "$CKPT" && pwd)"

ENV_PREFIX="${ENV_PREFIX:-$REPO_ROOT/.venv-sm70-130}"
PY="$ENV_PREFIX/bin/python"
[ -x "$PY" ] || { echo "ERROR: no environment at $ENV_PREFIX" >&2; exit 2; }

TP="${TP:-2}"
PP="${PP:-2}"
K="${K:-0}"
GMU="${GMU:-0.90}"
MML="${MML:-4096}"
MNS="${MNS:-4}"
MBT="${MBT:-2048}"
PORT="${PORT:-8026}"
LOG="${LOG:-$REPO_ROOT/serve-flash-next.log}"
# Stufe 0 trägt die PLE-Tabelle und muss jeden ple_layer_id enthalten
# (ple_layer_ids = [2]); 18/30 lässt Stufe 0 bei 18 Layern.
PP_PARTITION="${PP_PARTITION:-18,30}"

read -r -a EXTRA_ARGS_ARR <<< "${EXTRA_ARGS:-}"

SPEC_ARGS=()
if [ "$K" -gt 0 ]; then
  # Kein ${VAR:-{...}}: Bash matcht die schliessende Klammer der Expansion
  # falsch, sobald der Default selbst geschweifte Klammern enthaelt.
  if [ -z "${SPEC_CONFIG:-}" ]; then
    SPEC_CONFIG="{\"method\":\"mtp\",\"num_speculative_tokens\":$K,\"draft_sample_method\":\"greedy\"}"
  fi
  SPEC_ARGS=(--speculative-config "$SPEC_CONFIG")
fi

# Die QSA-Ringkapazität muss die Attention-Blockgröße teilen
# (QSAKeyStateCache.get_kv_cache_spec). Sie wächst mit k:
#   capacity = compress_ratio * ceil((compress_ratio + k) / compress_ratio)
# Bei k=7 und compress_ratio=4 sind das 12, was die Standardblockgröße 16
# nicht teilt. Deshalb hier das kleinste gemeinsame Vielfache setzen.
BLOCK_SIZE="${BLOCK_SIZE:-$(python3 - "$CKPT" "$K" <<'PY'
import json, math, sys
cfg = json.load(open(sys.argv[1] + "/config.json"))
text = cfg.get("text_config", cfg)
ratio = int(text.get("indexer_compress_ratio", 1))
k = int(sys.argv[2])
capacity = ratio * math.ceil((ratio + k) / ratio) if ratio > 1 else 1
print(math.lcm(16, capacity))
PY
)}"

rm -f "$LOG"
echo "==> booting $CKPT  (TP=$TP PP=$PP k=$K GMU=$GMU MML=$MML partition=$PP_PARTITION block=$BLOCK_SIZE ple_host=${PLE_HOST_GIB:-auto})"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2,1,4}" \
CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}" \
CUDA_HOME="${CUDA_HOME:-$REPO_ROOT/.cuda-nvcc-deb/usr/local/cuda-12.8}" \
TORCH_CUDA_ARCH_LIST=7.0 \
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" \
VLLM_PP_LAYER_PARTITION="$PP_PARTITION" \
VLLM_SM70_NVFP4_TURBOMIND="${TURBOMIND:-0}" \
VLLM_SM70_QUANT_BACKEND="${QUANT_BACKEND:-marlin}" \
VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS=$([ "$K" -gt 0 ] && echo 1 || echo 0) \
VLLM_SKINNY_NVFP4=1 \
VLLM_SKINNY_QPN=1 \
VLLM_SKINNY_QPN2=1 \
VLLM_SKINNY_NVFP4_SRC="$REPO_ROOT/kernels/skinny_kernels.cu" \
VLLM_QWEN4EXP_PLE_HOST_GIB="${PLE_HOST_GIB:-auto}" \
setsid "$PY" -m vllm.entrypoints.openai.api_server \
  --model "$CKPT" \
  --served-model-name qwen3.8-flash-next \
  --trust-remote-code \
  --dtype float16 \
  --tensor-parallel-size "$TP" \
  --pipeline-parallel-size "$PP" \
  --disable-custom-all-reduce \
  --gpu-memory-utilization "$GMU" \
  --block-size "$BLOCK_SIZE" \
  --max-model-len "$MML" \
  --max-num-seqs "$MNS" \
  --max-num-batched-tokens "$MBT" \
  --language-model-only \
  ${SPEC_ARGS[@]+"${SPEC_ARGS[@]}"} \
  ${EXTRA_ARGS_ARR[@]+"${EXTRA_ARGS_ARR[@]}"} \
  --host 127.0.0.1 --port "$PORT" > "$LOG" 2>&1 < /dev/null &
SERVER_PID=$!
echo "$SERVER_PID" > "$REPO_ROOT/.flash-next.pid"
echo "==> pid $SERVER_PID, log $LOG"

for i in $(seq 1 900); do
  if curl -sf -o /dev/null --max-time 2 "http://127.0.0.1:$PORT/v1/models"; then
    echo "==> UP on port $PORT (pid $SERVER_PID)"; exit 0
  fi
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "==> SERVER DIED after ${i}s"; exit 1; }
  sleep 1
done
echo "==> TIMEOUT after 900s (pid $SERVER_PID still alive)"; exit 2
