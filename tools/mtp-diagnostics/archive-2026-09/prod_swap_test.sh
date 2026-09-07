#!/usr/bin/env bash
# Produktionstest ueber llama-swap: exakt der Eintrag, den AIfred benutzt.
# Kaltstart -> llama-swap-restart (entlaedt) -> Warmstart. Zeit + Text vergleichen.
set -uo pipefail
SP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ask () {
  local label=$1
  local t0=$(date +%s)
  curl -s -m 1800 http://127.0.0.1:11435/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen3.8-27B-NVFP4-vllm","prompt":"Nenne die ersten zehn Primzahlen und erklaere kurz, warum 1 keine Primzahl ist.","temperature":0,"max_tokens":80,"seed":7}' \
    > "$SP/swap_$label.json" 2>&1
  echo "$label: $(( $(date +%s)-t0 ))s bis zur Antwort"
}
echo "=== Kaltstart (Modell nicht geladen)"; ask cold
echo "=== entladen via llama-swap-restart"; llama-swap-restart > "$SP/swap_restart.log" 2>&1; echo "restart rc=$?"
sleep 5
echo "=== Warmstart"; ask warm
echo "FERTIG-SWAP"
