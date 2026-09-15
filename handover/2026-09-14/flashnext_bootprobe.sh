#!/usr/bin/env bash
# Boot probe for the nvidia Flash-Next entry crash (TM gemm tuner illegal memory access):
# serve-qwen38-flash-next.sh with extra flags, then one chat request.
#   flashnext_bootprobe.sh <name> "<extra args>"
set -uo pipefail
NAME=$1; EXTRA=$2
REPO=/home/mp/Projekte/vllm-research/v100-skinny
CKPT=/home/mp/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47
W=$HOME/.cache/mtp-diagnostics/fnprobe_$NAME; rm -rf "$W"; mkdir -p "$W"; PORT=8028
cd $REPO
CUDA_VISIBLE_DEVICES=0,2,1,3 TURBOMIND=1 QUANT_BACKEND=turbomind ENV_PREFIX=/home/mp/vllm/venv \
TP=2 PP=2 K=4 GMU=0.95 MML=262144 MNS=${MNS:-4} PP_PARTITION=24,24 PLE_HOST_GIB=6 \
PORT=$PORT LOG=$W/boot.log BOOT_WAIT_S=2400 \
EXTRA_ARGS="--distributed-timeout-seconds 3600 --compilation-config {\"cudagraph_capture_sizes\":[1,2,4,5,8]} --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 $EXTRA" \
bash scripts/serve-qwen38-flash-next.sh "$CKPT" 2>&1 | tail -2
PID=$(cat $REPO/.flash-next.pid 2>/dev/null)
if curl -sf -o /dev/null --max-time 5 "http://127.0.0.1:$PORT/v1/models"; then
  curl -s -m 600 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.8-flash-next","max_tokens":2000,"messages":[{"role":"user","content":"Wie viel ist 17*23? Antworte nur mit der Zahl."}]}' | head -c 400; echo
  echo "RESULT $NAME: UP"
else
  echo "RESULT $NAME: DIED"
fi
grep -m2 -E "TM\]\[FATAL|Error:" $W/boot.log | cut -c1-200
[ -n "${PID:-}" ] && kill -TERM -$PID 2>/dev/null; sleep 20; [ -n "${PID:-}" ] && kill -KILL -$PID 2>/dev/null
