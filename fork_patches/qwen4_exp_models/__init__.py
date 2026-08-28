# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp model package."""

from typing import TYPE_CHECKING, Any

from .common.hyperconnection import (
    GatedResidual,
    GroupedGemmaRMSNorm,
    HyperConnectionBase,
    HyperConnectionConfig,
)

if TYPE_CHECKING:
    from .nvidia.model import (
        Qwen4ExpForCausalLM,
        Qwen4ExpForConditionalGeneration,
    )
    from .nvidia.mtp import Qwen4ExpMTP


def __getattr__(name: str) -> Any:
    if name in {
        "Qwen4ExpForCausalLM",
        "Qwen4ExpForConditionalGeneration",
        "Qwen4ExpMTP",
    }:
        from vllm.platforms import current_platform

        if current_platform.is_xpu() or current_platform.is_tpu():
            raise NotImplementedError("Qwen4Exp currently supports CUDA and ROCm only")
        # fork: pre-Ampere (V100 sm70 / RTX 8000 sm75) nimmt den Triton-Zweig.
        # Der "amd"-Zweig traegt keine AMD-Abhaengigkeit - er ist reines Triton
        # und erbt von den NVIDIA-Basisklassen (Lektion aus dem DeepSeek-V4-Port).
        # Der nvidia-Zweig braucht CuteDSL/FlashMLA-Kernel ab sm80.
        _pre_ampere = (
            current_platform.is_cuda()
            and current_platform.get_device_capability() is not None
            and current_platform.get_device_capability().to_int() < 80
        )
        if current_platform.is_rocm() or _pre_ampere:
            from .amd.model import (
                Qwen4ExpForCausalLM,
                Qwen4ExpForConditionalGeneration,
            )
            from .amd.mtp import Qwen4ExpMTP
        else:
            from .nvidia.model import (
                Qwen4ExpForCausalLM,
                Qwen4ExpForConditionalGeneration,
            )
            from .nvidia.mtp import Qwen4ExpMTP

        return {
            "Qwen4ExpForCausalLM": Qwen4ExpForCausalLM,
            "Qwen4ExpForConditionalGeneration": (Qwen4ExpForConditionalGeneration),
            "Qwen4ExpMTP": Qwen4ExpMTP,
        }[name]
    raise AttributeError(name)


__all__ = [
    "GatedResidual",
    "GroupedGemmaRMSNorm",
    "HyperConnectionBase",
    "HyperConnectionConfig",
    "Qwen4ExpForCausalLM",
    "Qwen4ExpForConditionalGeneration",
    "Qwen4ExpMTP",
]