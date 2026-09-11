#!/usr/bin/env bash
# Hardware-Probe fuer den Plattform-Wurzelfix auf dem gemischten Knoten:
# Sichtbarkeitsliste "0,1" = Quadro RTX 8000 (sm75) + Tesla V100 (sm70).
# Ein Prozess waehlt Geraet 1 (die V100) wie ein Worker mit local_rank 1 und
# fragt danach die Plattform OHNE Geraeteindex. Vorher: Antwort fuer die RTX
# (Index 0 der Liste). Nachher: Antwort fuer die V100. Dazu die Probe ohne
# CUDA_DEVICE_ORDER=PCI_BUS_ID (FASTEST_FIRST): dort weichen NVML- und
# torch-Ordnung ab, der torch-Pfad nach der Initialisierung muss trotzdem die
# gewaehlte Karte melden. Aufruf: devcap_probe.sh <main|fix>   (GPU 0 und 1 frei!)
set -uo pipefail
MODE=${1:?main|fix}
PY=/home/mp/vllm/venv/bin/python
case "$MODE" in
  main) TREE=/home/mp/Projekte/vllm-research/1Cat-vLLM-e2e-572 ;;   # fe67339d + #572, ohne Fix
  fix)  TREE=/home/mp/Projekte/vllm-research/1Cat-vLLM-pr-devcap ;;
esac
probe() {  # $1 = CUDA_DEVICE_ORDER oder "" (Vorgabe FASTEST_FIRST)
  ( cd /tmp && env -u CUDA_DEVICE_ORDER ${1:+CUDA_DEVICE_ORDER=$1} CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=$TREE \
    "$PY" - <<'PY' 2>/dev/null | grep -v "^W0"
import os, torch, vllm
from vllm.platforms import current_platform
order = os.environ.get("CUDA_DEVICE_ORDER", "(Vorgabe FASTEST_FIRST)")
print(f"  Baum {vllm.__file__.split('/')[-3]}  CUDA_DEVICE_ORDER={order}")
print(f"  vor set_device : cap()={current_platform.get_device_capability()}  is_initialized={torch.cuda.is_initialized()}")
torch.cuda.set_device(1)
name = torch.cuda.get_device_name(1)
print(f"  nach set_device(1) [{name}]: cap()={current_platform.get_device_capability()}  has(75)={current_platform.has_device_capability(75)}  is(70)={current_platform.is_device_capability(70)}")
PY
  )
}
echo "===== Probe [$MODE] mit CUDA_DEVICE_ORDER=PCI_BUS_ID"; probe PCI_BUS_ID
echo "===== Probe [$MODE] ohne CUDA_DEVICE_ORDER"; probe ""
