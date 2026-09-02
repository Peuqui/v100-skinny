# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-expert NVFP4 MoE dispatch over the skinny GEMM family (SM70/SM75).

Fork addition (v100-skinny). The chunked emulation proved the Volta port
end to end but dequantizes every selected expert to fp16 per forward. The
fork already ships benchmarked NVFP4 GEMMs (``skinny_nvfp4_v11``) whose
documented input layout -- codes uint8 [N][K/2], scales uint8 [N][K/16]
fp8-e4m3, one global scale -- is exactly what a compressed-tensors NVFP4
MoE checkpoint stores per expert (proved on real checkpoint bytes by
``scripts/nvfp4_expert_gemm_test.py``: 10/10, rel_err ~6e-4). So the MoE
needs no new kernel and no repacking: each router-selected expert's
weight slice feeds ``gemm_simt``/``gemm_wmma`` unchanged.

Deliberate deviation from the emulation: activations stay fp16 (w4a16),
matching how every NVFP4 *linear* layer in this fork is served. The
emulation's activation quantize-dequantize only loses precision; skipping
it keeps MoE and linear treatment consistent.

Compute order per expert: gather tokens -> gemm(w13 slice) -> activation
(the inherited helper applies swiglu_limit when the model sets one) ->
gemm(w2 slice) -> weighted scatter-add. M-dispatch follows the measured
linear-path frontier: simt M<=7, wmma M<=64, 64-row chunks above (the
prefill band re-reads expert weights once per chunk; acceptable until a
grouped kernel exists).
"""

import os

import torch

from vllm.compilation.breakable_cudagraph import eager_break_during_capture

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe import (
    Nvfp4QuantizationEmulationTritonExperts,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts

logger = init_logger(__name__)

_SKINNY_MOE_ENABLED = os.environ.get("VLLM_SM70_NVFP4_MOE_SKINNY", "1") == "1"
# Measured linear-path frontier (seam_bench): simt owns M<=7, wmma above.
_SIMT_MAX_M = 7
_WMMA_MAX_M = 64


def _expert_gemm(ext, x: torch.Tensor, codes: torch.Tensor,
                 scales: torch.Tensor, gscale: float) -> torch.Tensor:
    """One expert's GEMM at any M, using the existing skinny kernels."""
    m = x.size(0)
    if m <= _SIMT_MAX_M:
        return ext.gemm_simt(x, codes, scales, gscale)
    if m <= _WMMA_MAX_M:
        return ext.gemm_wmma(x, codes, scales, gscale)
    return torch.cat([
        ext.gemm_wmma(x[i:i + _WMMA_MAX_M], codes, scales, gscale)
        for i in range(0, m, _WMMA_MAX_M)
    ])


def skinny_moe_forward(
    ext,
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scales_u8: torch.Tensor,
    w2_scales_u8: torch.Tensor,
    g1: list[float],
    g2: list[float],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inter_dim: int,
    activation_fn,
) -> None:
    """Per-expert MoE forward; module-level so tests can drive it directly.

    ``activation_fn(out, inp)`` fills ``out`` [m, inter_dim] from ``inp``
    [m, N]; ``g1``/``g2`` are the per-expert multiplicative global scales
    (the reciprocal of the checkpoint's ``weight_global_scale``).
    """
    device = hidden_states.device
    top_k = topk_ids.size(1)
    w_flat = topk_weights.reshape(-1)

    # One host sync for the routing table instead of one per expert.
    slots_per_expert: dict[int, list[int]] = {}
    for slot, expert in enumerate(topk_ids.cpu().flatten().tolist()):
        slots_per_expert.setdefault(expert, []).append(slot)

    output.zero_()
    for expert, slots in slots_per_expert.items():
        slot_idx = torch.tensor(slots, dtype=torch.long, device=device)
        rows = slot_idx // top_k
        x_e = hidden_states.index_select(0, rows)

        y13 = _expert_gemm(ext, x_e, w1[expert], w1_scales_u8[expert],
                           g1[expert])
        inter = torch.empty((x_e.size(0), inter_dim), dtype=y13.dtype,
                            device=device)
        activation_fn(inter, y13)
        y = _expert_gemm(ext, inter, w2[expert], w2_scales_u8[expert],
                         g2[expert])

        y.mul_(w_flat.index_select(0, slot_idx).unsqueeze(1).to(y.dtype))
        output.index_add_(0, rows, y)


class Nvfp4SkinnySm70Experts(Nvfp4QuantizationEmulationTritonExperts):
    """NVFP4 MoE via per-expert skinny GEMMs on checkpoint-layout weights."""

    def __init__(self, moe_config, quant_config):
        # Skip the emulation __init__ (its "dequantize on the fly" warnings
        # would be wrong here) but keep its scale stashing: the uint8 scale
        # rasters move out of the quant config so no base-class path
        # mistakes them for fp scales.
        TritonExperts.__init__(self, moe_config, quant_config)
        logger.info_once(
            "Using Nvfp4SkinnySm70Experts MoE backend: per-expert skinny "
            "NVFP4 GEMMs on checkpoint-layout weights, fp16 activations."
        )
        self.w1_scale_val = self.quant_config.w1_scale
        self.w2_scale_val = self.quant_config.w2_scale
        self.quant_config._w1.scale = None
        self.quant_config._w2.scale = None
        self.quantization_emulation = False
        # Lazy caches built on first apply (weights live on the GPU then).
        self._g1: list[float] | None = None
        self._g2: list[float] | None = None
        self._w1_scales_u8: torch.Tensor | None = None
        self._w2_scales_u8: torch.Tensor | None = None

    @property
    def quant_dtype(self) -> torch.dtype | str | None:
        # w4a16: activations are never quantized.
        return None

    @staticmethod
    def _supports_current_device() -> bool:
        if not _SKINNY_MOE_ENABLED:
            return False
        if not (torch.cuda.is_available()
                and TritonExperts._supports_current_device()):
            return False
        # The LOCAL worker device decides, never device 0 of the visibility
        # list (the Session-4 lesson). The skinny NVFP4 extension is built
        # for sm70 and verified correct on sm75 (RTX-solo campaign).
        cap = torch.cuda.get_device_capability(torch.cuda.current_device())
        return cap in ((7, 0), (7, 5))

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # The per-expert loop allocates its own small transients; only the
        # fused output buffer is needed from the framework.
        return ((8,), (8,), (M, K))

    # Fork fix (v100-skinny): the per-expert loop routes on the host
    # (topk_ids.cpu()), which is illegal inside a CUDA graph capture. The
    # breakable capture the fork already uses for DeepSeek V4 ends the
    # current graph segment here, runs the MoE eagerly on the capture
    # stream and resumes capture -- the same mechanism as the attention
    # ops. Address contract: `output`, `hidden_states`, `topk_*` are
    # allocated in the captured segments and written in place; the loop's
    # transients stay local to the eager segment.
    @eager_break_during_capture(ignore_full_mode=True)
    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        assert w1.dtype == torch.uint8 and w2.dtype == torch.uint8
        assert hidden_states.dtype == torch.float16, (
            "skinny NVFP4 kernels take fp16 activations, got "
            f"{hidden_states.dtype}"
        )
        if expert_map is not None:
            raise NotImplementedError(
                "Per-expert skinny NVFP4 MoE does not support expert "
                "parallelism."
            )
        if apply_router_weight_on_input:
            raise NotImplementedError(
                "Per-expert skinny NVFP4 MoE does not support "
                "apply_router_weight_on_input."
            )

        from vllm.model_executor.kernels.linear.nvfp4.marlin import (
            _get_skinny_ext,
        )

        if self._g1 is None:
            self._g1 = self.quant_config.g1_alphas.cpu().tolist()
            self._g2 = self.quant_config.g2_alphas.cpu().tolist()
            self._w1_scales_u8 = self.w1_scale_val.view(torch.uint8)
            self._w2_scales_u8 = self.w2_scale_val.view(torch.uint8)

        inter_dim = self.adjust_N_for_activation(w1.size(1), activation)
        skinny_moe_forward(
            ext=_get_skinny_ext(),
            output=output,
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            w1_scales_u8=self._w1_scales_u8,
            w2_scales_u8=self._w2_scales_u8,
            g1=self._g1,
            g2=self._g2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inter_dim=inter_dim,
            activation_fn=lambda out, inp: self.activation(
                activation, out, inp),
        )
