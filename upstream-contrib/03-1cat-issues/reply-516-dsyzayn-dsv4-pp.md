# Antwort-Entwurf auf DSYZayn in PR #516 (2026-09-06 02:44Z)

Frage: "#485 is only for qwen4_next construction? I pull up the issue for deepseek v4 flash
TP4 x PP2 and today it is still not available."

Status: ENTWURF, nicht gepostet — Freigabe Peuqui.

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
transport question in the #439 thread; we serve V4 Flash at TP2/PP2 with DSpark
on our fork with it in place and will propose it upstream once the maintainers
say which transport they want.
