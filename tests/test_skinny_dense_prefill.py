# SPDX-License-Identifier: Apache-2.0
"""Dense-prefill mode of the skinny NVFP4 kernel (VLLM_SKINNY_DENSE_PREFILL=1):
after process_weights_after_loading only the QPN prepack stays resident, and
apply_weights serves every M against the checkpoint-native values. Uses the
DEPLOYED fork modules from the venv; run with CUDA_DEVICE_ORDER=PCI_BUS_ID."""

import os

os.environ["VLLM_SKINNY_NVFP4"] = "1"
os.environ["VLLM_SKINNY_QPN"] = "1"
os.environ["VLLM_SKINNY_QPN2"] = "1"
os.environ["VLLM_SKINNY_DENSE_PREFILL"] = "1"

import pytest  # noqa: E402
import torch  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
DEV = torch.device("cuda:0")
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _native_dense(codes, scales, gscale):
    n, k2 = codes.shape
    nib = torch.stack([codes & 0xF, codes >> 4], dim=-1).view(n, k2 * 2)
    vals = E2M1.to(DEV)[(nib & 7).long()] * torch.where(nib & 8 > 0, -1.0, 1.0)
    sc = scales.view(torch.float8_e4m3fn).to(torch.float32).repeat_interleave(16, dim=1)
    return (vals * sc * gscale).to(torch.float16)


class _Layer(torch.nn.Module):
    def __init__(self, n, k, seed=0):
        super().__init__()
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.input_size_per_partition = k
        self.output_size_per_partition = n
        self.params_dtype = torch.float16
        codes = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, generator=g).to(DEV)
        exp = torch.randint(3, 13, (n, k // 16), generator=g)
        mant = torch.randint(0, 8, (n, k // 16), generator=g)
        scales = ((exp << 3) | mant).to(torch.uint8).to(DEV).view(torch.float8_e4m3fn)
        self.weight = torch.nn.Parameter(codes, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(scales, requires_grad=False)
        self.weight_global_scale = torch.nn.Parameter(
            torch.tensor(1.0 / 2688.0, device=DEV, dtype=torch.float32), requires_grad=False)
        self.native_ref = _native_dense(codes, scales.view(torch.uint8), 1.0 / 2688.0)


@pytest.fixture(scope="module")
def prepared_layer():
    from vllm.model_executor.kernels.linear.nvfp4 import marlin as skm

    assert skm._DENSE_PREFILL, "test must run with VLLM_SKINNY_DENSE_PREFILL=1"
    layer = _Layer(5120, 17408)
    kernel = skm.MarlinNvFp4LinearKernel.__new__(skm.MarlinNvFp4LinearKernel)
    kernel.process_weights_after_loading(layer)
    return skm, kernel, layer


def test_only_qpn_prepack_stays_resident(prepared_layer):
    _, _, layer = prepared_layer
    n, k = layer.output_size_per_partition, layer.input_size_per_partition
    assert layer.weight.numel() == 0 and layer.weight_scale.numel() == 0
    assert layer.skinny_codes.numel() == 0 and layer.skinny_scales.numel() == 0
    assert layer.workspace.numel() == 0
    assert layer.skinny_qpn_codes.numel() == n * k // 2
    assert layer.skinny_qpn_scales.numel() == n * k // 16


@pytest.mark.parametrize("m", [1, 3, 8, 12, 24, 64, 2048])
def test_apply_weights_matches_native_values(prepared_layer, m):
    _, kernel, layer = prepared_layer
    k = layer.input_size_per_partition
    x = torch.randn(m, k, device=DEV, dtype=torch.float16) * 0.05
    y = kernel.apply_weights(layer, x)
    y_ref = torch.nn.functional.linear(x, layer.native_ref)
    denom = y_ref.float().abs().max().clamp(min=1e-6)
    rel = ((y.float() - y_ref.float()).abs().max() / denom).item()
    assert rel < 2e-2, f"M={m}: rel err {rel:.3e}"
    assert y.shape == (m, layer.output_size_per_partition)


def test_dense_route_is_logged_for_large_m(prepared_layer, caplog):
    skm, kernel, layer = prepared_layer
    skm._route_log_seen.clear()
    x = torch.randn(300, layer.input_size_per_partition, device=DEV, dtype=torch.float16)
    with caplog.at_level("INFO", logger=skm.logger.name):
        kernel.apply_weights(layer, x)
    assert any("M=300" in r.getMessage() and "-> dense" in r.getMessage() for r in caplog.records)
