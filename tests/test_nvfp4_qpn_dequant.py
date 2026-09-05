# SPDX-License-Identifier: Apache-2.0
"""QPN prepack -> dense fp16 dequant: must reproduce the checkpoint-native
NVFP4 values exactly, and feed cuBLAS to within fp16 noise of the QPN/marlin
kernels. Run with the fork venv and CUDA_DEVICE_ORDER=PCI_BUS_ID."""

import importlib.util
from pathlib import Path

import pytest
import torch

_FORK = Path(__file__).resolve().parents[1] / "fork_patches_150"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _FORK / filename)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


dq = _load("nvfp4_qpn_dequant", "nvfp4_qpn_dequant.py")
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
DEV = torch.device("cuda:0")


def _native_dequant(codes: torch.Tensor, scales: torch.Tensor, gscale: float) -> torch.Tensor:
    """Checkpoint layout: codes [N][K/2] (low nibble = even k), scales fp8 [N][K/16]."""
    n, k2 = codes.shape
    nib = torch.stack([codes & 0xF, codes >> 4], dim=-1).view(n, k2 * 2)
    mags = E2M1.to(DEV)[(nib & 7).long()]
    vals = mags * torch.where(nib & 8 > 0, -1.0, 1.0)
    sc = scales.view(torch.float8_e4m3fn).to(torch.float32).repeat_interleave(16, dim=1)
    return (vals * sc * gscale).to(torch.float16)


def _random_layer(n: int, k: int, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    codes = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, generator=g).to(DEV)
    # fp8 e4m3 scales in a realistic band (exponents 3..12), no NaN code
    exp = torch.randint(3, 13, (n, k // 16), generator=g)
    mant = torch.randint(0, 8, (n, k // 16), generator=g)
    scales = ((exp << 3) | mant).to(torch.uint8).to(DEV)
    return codes, scales, 1.0 / 2688.0


@pytest.fixture(scope="module")
def prepack():
    from vllm.model_executor.kernels.linear.nvfp4 import marlin as skm  # deployed fork

    return skm._qpn_prepack


@pytest.mark.parametrize(("n", "k"), [(64, 128), (5120, 17408), (34816, 5120)])
def test_kernel_matches_reference_and_native(prepack, n, k):
    codes, scales, gscale = _random_layer(n, k)
    qc, qs = prepack(codes, scales)
    assert qc is not None
    dense = dq.qpn_dequant(qc, qs, gscale, n, k)
    ref = dq.qpn_dequant_reference(qc, qs, gscale, n, k)
    native = _native_dequant(codes, scales, gscale)
    assert torch.equal(dense, ref), "kernel != torch inverse of the prepack"
    assert torch.equal(dense, native), "prepack round trip lost values"


def test_reuses_output_buffer(prepack):
    codes, scales, gscale = _random_layer(64, 128)
    qc, qs = prepack(codes, scales)
    out = torch.empty(64, 128, dtype=torch.float16, device=DEV)
    ret = dq.qpn_dequant(qc, qs, gscale, 64, 128, out=out)
    assert ret.data_ptr() == out.data_ptr()
    assert torch.equal(out, _native_dequant(codes, scales, gscale))


@pytest.mark.parametrize("m", [1, 8, 24, 64, 256, 2048])
def test_dense_gemm_agrees_with_skinny_kernels(prepack, m):
    """cuBLAS on the dequantized weight vs the fork's own routes."""
    from vllm.model_executor.kernels.linear.nvfp4 import marlin as skm

    n, k = 5120, 17408
    codes, scales, gscale = _random_layer(n, k, seed=1)
    qc, qs = prepack(codes, scales)
    w = dq.qpn_dequant(qc, qs, gscale, n, k)
    x = torch.randn(m, k, device=DEV, dtype=torch.float16) * 0.05
    y_dense = torch.nn.functional.linear(x, w)
    ext = skm._get_skinny_ext()
    if m <= 8:
        cfg = skm._qpn2_cfg(k, n)
        y_ref = ext.gemm_qpn2(x, qc, qs, gscale, n, cfg[0], cfg[1])
    elif m <= 16:
        y_ref = ext.gemm_qpn(x, qc, qs, gscale, n)
    else:
        fn = ext.gemm_simt if m <= 7 else ext.gemm_wmma
        if m <= 64:
            y_ref = fn(x, codes, scales.view(torch.uint8), gscale)
        else:
            # marlin serves M > 64 in the current path
            from types import SimpleNamespace

            from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
                apply_fp4_marlin_linear,
                prepare_fp4_layer_for_marlin,
            )

            layer = SimpleNamespace(
                input_size_per_partition=k, output_size_per_partition=n,
                params_dtype=torch.float16,
                weight=torch.nn.Parameter(codes.clone(), requires_grad=False),
                weight_scale=torch.nn.Parameter(scales.view(torch.float8_e4m3fn).clone(), requires_grad=False),
                weight_global_scale=torch.nn.Parameter(torch.tensor(gscale, device=DEV), requires_grad=False),
            )
            prepare_fp4_layer_for_marlin(layer)
            y_ref = apply_fp4_marlin_linear(
                input=x, weight=layer.weight, weight_scale=layer.weight_scale,
                weight_global_scale=layer.weight_global_scale, workspace=layer.workspace,
                size_n=n, size_k=k)
    denom = y_ref.float().abs().max().clamp(min=1e-6)
    rel = ((y_dense.float() - y_ref.float()).abs().max() / denom).item()
    assert rel < 2e-2, f"M={m}: rel err {rel:.3e} vs reference route"
