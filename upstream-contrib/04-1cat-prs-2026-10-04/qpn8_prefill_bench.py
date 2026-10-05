# Volta block QPN8, rows beyond M=8: TurboMind copy vs dense FP16 prefill.
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.kernels.linear.scaled_mm.qpn8_blk import QPN8Fp8BlockScaledMMLinearKernel
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import FP8ScaledMMLinearLayerConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8Dynamic128Sym, kFp8Static128BlockSym

SHAPES = [(1536, 4096), (32768, 1024), (8192, 1024), (4096, 8192), (4096, 4096), (4096, 2048)]
MS = [9, 16, 64, 256, 512]

def make(n, k, copy):
    torch.manual_seed(n + k)
    raw = torch.randint(0, 256, (n, k), dtype=torch.uint8, device="cuda")
    raw[raw == 0x7F] = 0x7E; raw[raw == 0xFF] = 0xFE
    w = raw.view(torch.float8_e4m3fn)
    s = (2.0 ** (torch.rand(n // 128, k // 128, device="cuda") * 3 - 12)).float()
    ref = w.float() * s.repeat_interleave(128, 0).repeat_interleave(128, 1)
    layer = torch.nn.Module(); layer.prefix = f"b.{n}x{k}"
    layer.weight = torch.nn.Parameter(w.clone(), requires_grad=False)
    layer.weight_scale_inv = torch.nn.Parameter(s.clone(), requires_grad=False)
    cfg = VllmConfig(); cfg.kernel_config.sm70_fp8.block_qpn8_volta_turbomind_prefill = copy
    with set_current_vllm_config(cfg):
        kern = QPN8Fp8BlockScaledMMLinearKernel(FP8ScaledMMLinearLayerConfig(
            weight_quant_key=kFp8Static128BlockSym, activation_quant_key=kFp8Dynamic128Sym,
            weight_shape=(n, k), input_dtype=torch.float16, out_dtype=torch.float16))
        kern.process_weights_after_loading(layer)
    assert hasattr(layer, "_qpn8_fallback_k_ld") == copy
    return kern, layer, ref

def time_ms(fn, reps=50):
    for _ in range(5): fn()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize(); a.record()
    for _ in range(reps): fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / reps

print(torch.cuda.get_device_name(), "| Zeit in ms je Aufruf (weniger ist besser), rel. Fehler gg. FP32")
tot = {True: {}, False: {}}
for n, k in SHAPES:
    arms = {c: make(n, k, c) for c in (True, False)}
    for m in MS:
        x = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.1
        row = []
        for c in (True, False):
            kern, layer, ref = arms[c]
            y = kern.apply_weights(layer, x)
            e = x.float() @ ref.t()
            err = ((y.float() - e).norm() / e.norm()).item()
            t = time_ms(lambda: kern.apply_weights(layer, x))
            tot[c][m] = tot[c].get(m, 0) + t
            row.append(f"{'TM' if c else 'dicht'} {t:7.3f} ms err {err:.2e}")
        print(f"N={n:5d} K={k:5d} M={m:3d}  " + "  |  ".join(row))
print("Summe über alle Formen je M (ms): " + "  ".join(f"M={m}: TM {tot[True][m]:.2f} / dicht {tot[False][m]:.2f}" for m in MS))
