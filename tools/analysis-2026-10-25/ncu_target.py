"""One real GLM o_proj (4096x8192) and one qkv-like shape through the row-group QPN8 GEMV, M=1, for ncu."""
import json, sys
import torch
from safetensors import safe_open
from torch import nn
from vllm.model_executor.kernels.linear.qpn import pre_ampere  # noqa: F401
from vllm.model_executor.layers.quantization import sm70_turbomind as tm

ROOT = "/home/mp/models/GLM-5.3-Flash-NVFP4-MTP"
index = json.load(open(f"{ROOT}/model.safetensors.index.json"))["weight_map"]
name = sys.argv[1] if len(sys.argv) > 1 else "layers.4.self_attn.o_proj"
t = {}
for suffix in ("weight", "weight_scale"):
    key = f"model.language_model.{name}.{suffix}"
    with safe_open(f"{ROOT}/{index[key]}", "pt", device="cuda") as sf:
        t[suffix] = sf.get_tensor(key)
layer = nn.Module()
tm.prepare_mxfp8_qpn8_linear(layer, t["weight"], t["weight_scale"])
x = torch.randn(1, t["weight"].shape[1], device="cuda").to(torch.float16)
for _ in range(3):
    tm.apply_prepared_fp8_qpn8_linear(layer, x, None)
torch.cuda.synchronize()
tm.apply_prepared_fp8_qpn8_linear(layer, x, None)
torch.cuda.synchronize()
print("fertig", name, flush=True)
