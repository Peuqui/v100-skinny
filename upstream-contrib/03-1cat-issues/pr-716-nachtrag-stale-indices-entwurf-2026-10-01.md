Nachtrag zu #716 (Zweig sm70-dsv4-sparse-bmm, Worktree 1Cat-vLLM-pr-bmm, unkommittet)

Befund 01.10. früh im E2E (1Cat main + unsere PRs + S3, DSv4-Flash PP5, 2x RTX + 3x V100,
VLLM_SM70_DSV4_SPARSE_MLA_BMM_PREFILL=1): erste Anfrage stirbt am Device-Assert
"vectorized gather kernel index out of bounds". Ursache: DeepseekV4SM70SparseImpl._forward_prefill
übergibt combine_topk_swa_indices einen Workspace-Puffer (out=...). Der Kernel schreibt nur die
gültigen Plätze; dahinter bleibt alter Inhalt. Der Triton-Gather-Kernel lädt jenseits der Länge
nichts, die BMM-Prefill sammelt aber alle W Spalten per index_select ein und klemmte nur negative
Indizes. Nicht Turing-spezifisch: der neue Testfall assertet auf V100 und RTX 8000 gleich.

## Commit

[Bugfix][SM70] Mask stale prefill indices past the length before the gather

combine_topk_swa_indices writes only the valid slots when it fills a
caller's workspace, so past each token's length the index tensor holds
whatever the buffer held before. The gathered Triton kernel never loads
those slots; the batched-matmul prefill gathers all of them and only
clamped negative indices, so a stale value past the length read outside
kv and tripped the index_select device assert on the first long prompt.

Unused slots, past the length or -1, now read row 0 and stay masked out of
the softmax as before. The test gains stale out-of-range tails; without
the fix it asserts on V100 and on Quadro RTX 8000.

The decode graph test now takes vLLM's current_stream() as its main
stream. With torch's default stream there, leaving the side-stream
context recorded the default stream as vLLM's current stream, and later
graph captures in the same pytest process failed.

## Kommentar im PR

Fixed a bug in the prefill path, pushed as <sha>.

`combine_topk_swa_indices` writes only the valid slots when it fills a
caller's workspace, which is how `DeepseekV4SM70SparseImpl._forward_prefill`
calls it. Past each token's length the index tensor therefore holds whatever
that buffer held before. The gathered Triton kernel never loads those slots;
`sparse_attn_prefill_bmm` gathers every slot and only clamped negative
indices. A stale value past the length then indexed outside `kv`, and the
first long prompt died at the `index_select` device assert ("vectorized
gather kernel index out of bounds"). I hit it end to end with
`VLLM_SM70_DSV4_SPARSE_MLA_BMM_PREFILL=1`.

Unused slots (past the length, or -1) are now set to 0 before the gather. They
stay masked out of the softmax as before. The prefill test gains a case with
stale out-of-range tails. Without the fix that case fails on both cards.

Separately, the decode graph test took torch's default stream as its main
stream. Leaving its side-stream context then recorded the default stream as
vLLM's current stream (`vllm.utils.torch_utils` patches `set_stream`), and
`TestCUDAGraphWrapper` in `tests/v1/cudagraph/test_cudagraph_dispatch.py`
failed when it ran later in the same process. The test now takes vLLM's
`current_stream()`.

With both changes, and with `test_cudagraph_dispatch.py` last in the same
process:

| | Tesla V100 | Quadro RTX 8000 |
|---|---|---|
| bmm tests + test_deepseek_v4_sm70_routes.py + test_cudagraph_dispatch.py | 88 passed | 76 passed, 12 skipped (SM70-only) |

pre-commit clean.

---
Status: Entwurf, wartet auf Freigabe (Commit + Push auf den PR-Zweig + Kommentar).
