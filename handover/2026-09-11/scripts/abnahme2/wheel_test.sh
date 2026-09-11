#!/usr/bin/env bash
# Editable-PR: Wheel-Bau aus dem Belegbau-Worktree (fe67339d + setup.py-Aenderung,
# vorhandener CMake-Build-Baum -> inkrementell) und Abnahme des Wheels:
#   1. bdist_wheel mit der Umgebung des Belegbaus (STAND.md, Abschnitt Bau)
#   2. .so-Namen im Wheel listen, gegen das offizielle 1.5.0-Wheel vergleichen (falls lokal)
#   3. Kopie der Produktions-venv nach /home/mp/vllm/venv-wheeltest, Wheel hinein
#   4. 27B DFlash2 auf dem V100-Paar mit VENV=venv-wheeltest (speed_dflash.sh)
# NICHT parallel zu GPU-Messungen starten (nvcc-Last verfaelscht tok/s).
# Aufruf: wheel_test.sh <OUT> [build|accept|all]
set -uo pipefail
OUT=${1:?OUT fehlt}; STEP=${2:-all}; mkdir -p "$OUT"
WT=/home/mp/Projekte/vllm-research/1Cat-vLLM-editable-pr
VENV=/home/mp/Projekte/vllm-research/v100-skinny/.venv-sm70-main
VT=/home/mp/vllm/venv-wheeltest
REPO=/home/mp/Projekte/vllm-research/v100-skinny
D=/home/mp/.cache/huggingface/hub/models--maurienne-ai--Qwen3.8-27B-DFlash2-NVFP4-RTNcal/snapshots/bd7a934213c47a9e7ef69eef36bb3325f47fd1f1
# Unter systemd-run --user ist der PATH minimal: ninja liegt im venv-bin, nvcc
# im CUDA-Symlink. Beides explizit voranstellen (Fehlschlag 11.09. 19:11).
export PATH=$VENV/bin:/home/mp/vllm/cuda/bin:$PATH
stamp() { echo "===== $(date '+%F %T') $*"; }

if [ "$STEP" = build ] || [ "$STEP" = all ]; then
  stamp "1) bdist_wheel im Belegbau-Worktree ($(git -C "$WT" log --oneline -1 | cut -c1-60), setup.py $(git -C "$WT" diff --stat | tail -1))"
  cd "$WT" && rm -rf dist
  time env -u VLLM_FLASH_ATTN_SRC_DIR \
    CPATH=$VENV/lib/python3.12/site-packages/nvidia/cuda_cccl/include \
    CUDA_HOME=/home/mp/vllm/cuda TORCH_CUDA_ARCH_LIST=7.0 MAX_JOBS=4 \
    "$VENV/bin/python" setup.py bdist_wheel > "$OUT/bdist_wheel.log" 2>&1
  echo "EXIT $?"; tail -3 "$OUT/bdist_wheel.log"
  W=$(ls "$WT"/dist/*.whl 2>/dev/null | head -1)
  [ -n "$W" ] || { echo "ABBRUCH: kein Wheel entstanden"; exit 1; }
  stamp "2) Wheel: $(basename "$W") ($(du -h "$W" | cut -f1))"
  unzip -l "$W" | grep -o 'vllm/[^ ]*\.so$' | sort > "$OUT/wheel_so.txt"
  cat "$OUT/wheel_so.txt"
  OFF=$(ls /home/mp/Projekte/v100-skinny/.wheels/1cat_vllm-1.5.0*.whl /home/mp/vllm/1cat_vllm-1.5.0*.whl 2>/dev/null | head -1)
  if [ -n "$OFF" ]; then
    unzip -l "$OFF" | grep -o 'vllm/[^ ]*\.so$' | sort > "$OUT/official_so.txt"
    echo "--- .so-Namen gegen offizielles Wheel $(basename "$OFF"):"
    diff "$OUT/official_so.txt" "$OUT/wheel_so.txt" && echo "IDENTISCH" || echo "(Abweichungen oben; neue Extensions seit 1.5.0 sind erwartbar)"
  else
    echo "(kein offizielles 1.5.0-Wheel lokal, kein Namensvergleich)"
  fi
fi

if [ "$STEP" = accept ] || [ "$STEP" = all ]; then
  W=$(ls "$WT"/dist/*.whl 2>/dev/null | head -1); [ -n "$W" ] || { echo "ABBRUCH: kein Wheel"; exit 1; }
  stamp "3) venv-Kopie $VT (11 GB) und Wheel-Installation"
  [ -d "$VT" ] || cp -a "$VENV" "$VT"
  # Ein kopiertes venv traegt absolute Pfade in bin/*; pip mit dem Interpreter
  # der Kopie aufrufen und danach pruefen, dass vllm aus site-packages kommt.
  "$VT/bin/python" -m pip install --no-deps --force-reinstall "$W" > "$OUT/pip_wheel.log" 2>&1; echo "pip EXIT $?"; tail -2 "$OUT/pip_wheel.log"
  ( cd "$OUT" && CUDA_VISIBLE_DEVICES="" "$VT/bin/python" -c "import vllm,sys; print('vllm:', vllm.__file__); print('python:', sys.executable)" 2>&1 | grep -v "^W0" )
  for m in _sm70_exact_reduce_C _h3_w8a16_C _h3_flashinfer_C _h3_flashattn_C _sm70_sparse_attention_C _sm70_sampler_C; do
    ( cd "$OUT" && CUDA_VISIBLE_DEVICES="" "$VT/bin/python" -c "import importlib; m=importlib.import_module('vllm.$m'); print('  $m:', m.__file__.split('/')[-1])" 2>&1 | grep -v "^W0" )
  done
  stamp "4) 27B DFlash2 V100-Paar aus dem Wheel (VENV=$VT), incoai-Kopf"
  cd "$REPO"
  # KEIN DRAFT=maurienne: das Wheel ist reines main ohne #592, der NVFP4-Kopf
  # bricht dort mit "mat1 and mat2 shapes cannot be multiplied" (Lauf 19:43).
  # Der unquantisierte incoai-Kopf ist der Stand, den main tragen muss.
  VENV=$VT DEVS=1,3 bash tools/mtp-diagnostics/speed_dflash.sh wheel_v100 fork dflash 2>&1 | tail -3
  grep -o '"sha256": "[0-9a-f]*"' "$HOME/.cache/mtp-diagnostics/qual_wheel_v100/result.json" | sed 's/^/wheel_v100 /'
  echo "Tracebacks im Boot-Log: $(grep -c Traceback "$HOME/.cache/mtp-diagnostics/qual_wheel_v100/boot.log")"
  stamp "WHEEL-TEST-ENDE"
fi
