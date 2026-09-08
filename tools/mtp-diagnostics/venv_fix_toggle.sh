#!/usr/bin/env bash
# fix4.sh on|off  — schaltet Fix 4 (Slice statt Materialisierung) im venv um
V=/home/mp/vllm/venv/lib/python3.12/site-packages/vllm/model_executor/layers/mamba/gdn
case "$1" in
  on)  cp -a $V/qwen_gdn_linear_attn.py.MITFIX4  $V/qwen_gdn_linear_attn.py ;;
  off) cp -a $V/qwen_gdn_linear_attn.py.OHNEFIX4 $V/qwen_gdn_linear_attn.py ;;
  *) echo "on|off"; exit 1 ;;
esac
echo "Fix 4: $1  (SLICEDIAG-Marker: $(grep -c SLICEDIAG $V/qwen_gdn_linear_attn.py))"
