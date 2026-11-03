"""q/k_proj of GLM-5.3 KDA layers with E8M0 exponents 102..104 (below the old exact range): bit-exactness of the QPN8 dequantization
against the load-time dequantization, and the decode kernel's error against the exact result."""
import json
import torch
from safetensors import safe_open
from torch import nn

from vllm import _sm70_ops
from vllm.model_executor.kernels.linear.qpn import pre_ampere  # noqa: F401
from vllm.model_executor.layers.quantization import sm70_turbomind as tm
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import dequant_mxfp8_to_bf16

ROOT = "/home/mp/models/GLM-5.3-Flash-NVFP4-MTP"
index = json.load(open(f"{ROOT}/model.safetensors.index.json"))["weight_map"]
print(torch.cuda.get_device_name(0), "Bereich", tm.MXFP8_QPN8_EXPONENT_RANGE, flush=True)
for name in ("layers.1.self_attn.q_proj", "layers.28.self_attn.k_proj", "layers.8.self_attn.q_proj", "layers.41.self_attn.k_proj"):
    t = {}
    for suffix in ("weight", "weight_scale"):
        key = f"model.language_model.{name}.{suffix}"
        with safe_open(f"{ROOT}/{index[key]}", "pt", device="cuda") as sf:
            t[suffix] = sf.get_tensor(key)
    weight, scale = t["weight"], t["weight_scale"]
    n, k = weight.shape
    layer = nn.Module()
    tm.prepare_mxfp8_qpn8_linear(layer, weight, scale)
    state = getattr(layer, tm.FP8_QPN8_STATE_ATTR)
    dense = torch.empty(k, n, device="cuda", dtype=torch.float16)
    _sm70_ops.fp8_qpn8_dequantize_sm70_out(dense, state.codes, state.group_scales)
    reference = dequant_mxfp8_to_bf16(weight, scale).to(torch.float16)
    exact_bits = torch.equal(dense, reference.t())
    low = int((scale < 105).sum())
    x = (torch.randn(4, k, device="cuda") * 0.5).to(torch.float16)
    out = tm.apply_prepared_fp8_qpn8_linear(layer, x, None)
    exact = x.double() @ reference.double().t()
    err = ((out.double() - exact).abs().max() / exact.abs().max()).item()
    print(f"{name:28s} N {n} K {k} | Blöcke < 105: {low:5d} | Dequantisierung bitgleich: {exact_bits} | Decode-Fehler M4 {err:.1e}", flush=True)
