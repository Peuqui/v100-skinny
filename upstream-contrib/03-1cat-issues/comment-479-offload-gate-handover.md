# Kommentar-Entwurf für Issue #479 (Gate 1 abgeben, dritte Scheibe ankündigen)

Status: ENTWURF, nicht gepostet — Freigabe Peuqui. Posten zusammen mit dem Push von
`qwen4exp-ple-split-placement` (Link im Text).

---

Status of the three points, so nobody waits for us on the wrong one:

3 (loader crash) landed as #485. 2 (PLE gate on the partition instead of
the PP size) is #516.

1 (VLLM_PLE_CPU_OFFLOAD refusing PP) we will not take: we cannot test it.
The offload child loads the full table into host memory (51 GB for
Flash Next) and this box has 30 GB, so the path never boots here. From
reading the code, lifting the gate needs three things beyond the
partition check from #516: the workers on later PP stages must not set
up the connector (Worker._ple_offload_enabled is config-based, so today
every rank would register and the child, which waits for exactly
dp*tp registrations, would see pp times as many), the child must drop an
inherited VLLM_PP_LAYER_PARTITION before it builds its PP=1 meta model
(get_pp_indices rejects a two-entry partition against pp_size=1), and
the child's num_workers stays dp*tp because only stage 0 owns PLE
layers. Whoever has the host RAM for it can start from that; it is a
small change, the test is the expensive part.

What we run instead, and propose as the third slice: the pinned-host
table split by capacity -- rows stay in device memory as long as the
stage's weights, the KV cache of the requested context and a reserve
fit, only the remainder is pinned on the host (VLLM_QWEN4EXP_PLE_HOST_GIB
to fix the share). That is how Flash Next serves at TP2/PP2 on our
mixed cards at 51.9 tok/s with MTP. Branch:
https://github.com/Peuqui/1Cat-vLLM/tree/qwen4exp-ple-split-placement --
PR follows if you want it in this shape.
