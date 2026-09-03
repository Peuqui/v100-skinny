# Antwort-Entwurf für SabaTech-dev in Issue #441 (flash_attn_v100 64x80-Tile)

**Status:** Entwurf 2026-09-03, wartet auf Freigabe Peuqui.
**Quelle des Diffs:** lokaler Sparse-Clone `~/Projekte/1Cat-vLLM`, Worktree
gegen Tag v1.3.0 (`git diff v1.3.0 -- flash-attention-v100`). Die Änderung
ist vollständig — ein Tile-Makro-Paar in `fused_mha_forward_paged.cu`.

---

@SabaTech-dev Great to hear the QSA dispatch tiles landed cleanly in your
fork — thanks for the attribution, and for the reciprocal numbers. Your
BLOCK_N=128 decode finding and the BM=16/BN=64 smem-cliff data point both
match the pattern we keep hitting on Volta, and a Triton paged-attention
backend that beats the CUDA kernels on decode is a very interesting data
point for us.

Here is the 64x80 prefill tile change you asked for. It is deliberately
tiny — the tile lives in the macros of `kernel/fused_mha_forward_paged.cu`
(source state: the `flash-attention-v100` tree in 1CatAI/1Cat-vLLM at tag
v1.3.0; current main still carries the same stock 32x176 macros, so it
ports 1:1), and a rebuild of the extension picks it up directly:

```diff
--- a/flash-attention-v100/kernel/fused_mha_forward_paged.cu
+++ b/flash-attention-v100/kernel/fused_mha_forward_paged.cu
@@ -62,8 +62,12 @@ int kv_cache_dtype_code_from_string(const std::string& kv_cache_dtype) {
 #define BLOCK_N_64  128
 #define WARPS_64    16
 
-#define BLOCK_M_128 32
-#define BLOCK_N_128 176
+// Tile retuned 2026-08-30: 64x80 measures 14.06 ms vs 16.10 ms
+// for 32x176 on a 2048-token chunk against 31k paged KV (V100, H=4, fp16)
+// -- larger M amortizes KV traffic; 64x80 is the largest tile that fits
+// the 96 KB smem budget with M a multiple of 32.
+#define BLOCK_M_128 64
+#define BLOCK_N_128 80
 #define WARPS_128   16
 
 #define BLOCK_M_256 32
```

Notes from our measurements (V100-PCIE-32GB, H=4, D=128, fp16, 2048-token
chunk against 31k paged KV):

- 64x80: 14.06 ms vs 16.10 ms for the stock 32x176 (-13%). Empirical smem
  balance for this kernel: 272*N + 856*M + 4*M*N bytes against the 96 KB
  carve-out; 64x80 is the largest fit with M a multiple of 32.
- Confirming your note: M values that are not a multiple of 32 (48x112,
  80x48) produced wrong results in our sweep as well. On Turing you would
  additionally hit the 64 KB smem ceiling, which this tile exceeds; it is
  a Volta-only tune.
- Honesty note: end-to-end this was NEUTRAL on our 27B @31k grid — attention
  is not the bottleneck there (NVFP4/QPN8 GEMMs dominate). It should pay off
  where prefill attention actually dominates: long contexts, larger H, and
  chunked prefill.

We have not upstreamed flash_attn_v100 anywhere, so the diff above is the
whole change. Happy to compare notes once your Triton prefill closes in on
the CUDA kernel — and thanks for the mutual measurement offer; we may take
you up on it, e.g. for V100-SXM2/NVLink numbers, since our Voltas are all
PCIE.
