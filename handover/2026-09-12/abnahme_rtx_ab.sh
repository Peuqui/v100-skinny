#!/usr/bin/env bash
# Nachlauf: RTX 73,15 gegen Referenz 77,1 einordnen. Alter Produktionsbaum (911c259f) wird in einer KOPIE der
# tltest-venv editable gebaut (keine neuen Pakete, --no-deps), dann alt gegen neu am selben Abend auf RTX und V100.
set -uo pipefail
cd /home/mp/Projekte/vllm-research/v100-skinny
OUT=handover/2026-09-12/abnahme_rebuild_chain.out
for i in $(seq 1 600); do grep -qE "ABNAHME-ENDE|^ABBRUCH" $OUT && break; sleep 30; done
OLDV=/home/mp/Projekte/vllm-research/v100-skinny/.venv-sm70-old; OLDT=/home/mp/Projekte/vllm-research/1Cat-vLLM-old-prod
if [ ! -x $OLDV/bin/python ]; then cp -a /home/mp/Projekte/vllm-research/v100-skinny/.venv-sm70-tltest $OLDV; fi
echo "== Bau alter Baum in $OLDV"; date
( cd $OLDT && env -u VLLM_FLASH_ATTN_SRC_DIR CPATH=$OLDV/lib/python3.12/site-packages/nvidia/cuda_cccl/include CUDA_HOME=/home/mp/vllm/cuda TORCH_CUDA_ARCH_LIST=7.0 MAX_JOBS=4 $OLDV/bin/python -m pip install -e . --no-build-isolation --no-deps > handover_old_build.log 2>&1; echo "OLD-PIP-EXIT $?" )
date; $OLDV/bin/python -c "import vllm; print('alt:', vllm.__version__, vllm.__file__)"
S=tools/mtp-diagnostics/speed_dflash.sh
wait_free() { for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc); [ "${u:-0}" -le 1000 ] && return 0; sleep 5; done; echo "ABBRUCH: Karten nicht frei"; exit 1; }
for P in "rb_rtx_new2 /home/mp/vllm/venv 0,2" "rb_rtx_old $OLDV 0,2" "rb_v100_old $OLDV 1,3" "rb_rtx_new3 /home/mp/vllm/venv 0,2"; do
  set -- $P; wait_free; echo "== $1 (venv $2, DEVS $3)"
  VENV=$2 DEVS=$3 bash $S $1 fork dflash 2>&1 | grep -E "STATUS|MEDIAN|Error|ABBRUCH"
  echo "   sha: $(grep -h -o '"sha256": "[0-9a-f]*"' ~/.cache/mtp-diagnostics/qual_$1/result.json 2>/dev/null)  fa2: $(grep -a -o -m1 'Loaded FA2 library [^ ]*' ~/.cache/mtp-diagnostics/qual_$1/boot.log)"
done
echo RTXAB-ENDE
