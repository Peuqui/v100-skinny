# Antwort-Entwurf auf DSYZayn in PR #516 (2026-09-06 02:44Z)

Frage: "#485 is only for qwen4_next construction? I pull up the issue for deepseek v4 flash
TP4 x PP2 and today it is still not available."

Status: GEPOSTET 2026-09-06 (Freigabe Peuqui), Branch gepusht.
Voraussetzung: Branch pp-spec-state-transport auf Peuqui/1Cat-vLLM gepusht.

---

Right: #485 and #516 are Qwen4Exp-specific slices (the mixer weights on non-last
ranks, and the PLE partition gate) and do not touch DeepSeek V4 Flash.

The crash you reported in #439 (`'GPUModelRunner' object has no attribute
'drafter'` during memory profiling) is model-independent; that is what #511
fixes, and it is still open, so on today's main you will keep hitting it.

Two things to expect once #511 is in: the non-last ranks boot and profile, but
speculative decoding under PP (DSpark/MTP on V4 Flash, TP4/PP2 included) is not
usable end to end on main yet -- the non-last stages still need the sampled
tokens, accepted counts and hybrid-state update from the last stage. That is the
transport question in the #439 thread; we serve V4 Flash with DSpark across
five PP stages on our fork with it in place and will propose it upstream once
the maintainers say which transport they want.

If you want to try it before that lands: the branch
https://github.com/Peuqui/1Cat-vLLM/tree/pp-spec-state-transport has #511
plus the two missing pieces rebased onto current main -- the round-state
broadcast (sampled matrix and draft ids from the last stage, accepted
counts derived on the other ranks with the drafter's kernel) and a DSpark
fix for PP, where the drafter's own embedding table was never loaded from
the checkpoint (the boot looks fine, acceptance drops to a few percent).
The changes are Python only, so they drop onto an existing main build.
We could not run TP4 x PP2 here (2x RTX 8000 + 3x V100); TP2 x PP2 with MTP
(Qwen3.8) and a five-stage DSpark pipeline (V4 Flash) work on our side, so
your run would be the first on that layout -- if it hangs in the first decode round, the
transport is the first suspect and I would like to see the log.
