#!/usr/bin/env bash
# Nacharbeiten nach der Abnahme, GPU-seriell:
#   1. Chat-Sonde Flash-Next 3x (Produktionspfad, Template)
#   2. 11a End-zu-End + CUDA-Tests (main+#572, RTX-Paar, GPU 4)
#   3. E5-A/B (27B-MTP V1-Runner, RTX-Paar)  -- Freigabe Peuqui 11.09.
#   4. Wheel-Bau (CPU) und Wheel-Abnahme (V100-Paar)
set -uo pipefail
S=$(dirname "$0"); OUT=${1:?OUT fehlt}; mkdir -p "$OUT"
REPO=/home/mp/Projekte/vllm-research/v100-skinny
stamp() { echo "===== $(date '+%F %T') $*"; }

stamp "1) Chat-Sonde Flash-Next, 3 Laeufe, Aufraeum-Stand"
cd "$REPO"
for r in 1 2 3; do
  stamp "1.$r"
  bash "$S/flashnext_qual_chat.sh" acc2_chat_$r 4 2>&1 | grep "q[123]:\|UP on\|FERTIG\|Traceback\|Error\|PARSER\|STATUS"
done

stamp "2) Paket 11a End-zu-End"
bash "$S/e2e_11a.sh" "$OUT/e2e11a" 2>&1

stamp "3) E5-A/B"
bash "$S/e5_ab.sh" "$OUT/e5" 2>&1

stamp "4) Wheel-Bau und Wheel-Abnahme"
bash "$S/wheel_test.sh" "$OUT/wheel" all 2>&1

stamp "NACHARBEIT-ENDE"
