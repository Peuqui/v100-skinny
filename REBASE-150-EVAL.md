# Rebase-Evaluierung: fork_patches (Basis 1.3.0) gegen 1Cat-vLLM v1.5.0

Stand 2026-09-03. Methode: alle 83 Deploy-Ziele aus scripts/bootstrap-sm70.sh
per `git diff v1.3.0 v1.5.0 -- <pfad>` klassifiziert (Repo-Klon
~/Projekte/1Cat-vLLM, beide Tags). Wheel 1.5.0 liegt installiert in
.venv-sm70-150 (A/B-venv des MoE-Vergleichs).

## Kernbefunde

1. **ZIELMODELL-BLOCKER GELOEST: v1.5.0 traegt das komplette
   `vllm/models/qwen4_exp/`-Paket** (nvidia/ + amd/-Baeume, Registry-
   Eintraege Qwen4ExpForCausalLM/-ConditionalGeneration) plus die
   Decode/Prefill-Optimierungen der PRs #345/#361/#390/#393/#398 —
   Upstream faehrt Qwen3.8-Flash-Next produktiv auf V100 (82 tok/s TP4,
   PR #361). KORREKTUR nach Memory-Gegencheck: unser eigener Qwen4Exp-
   Port ist seit 28.08. FERTIG und produktiv (51,9 tok/s k=4, llama-swap-
   Eintrag, SSOT FLASH-NEXT-OPERATING-POINT.md) — der 1.5.0-Wert liegt in
   deren NACHGELAGERTEN Optimierungen: Decode-PRs #345-#398, pooled
   disk-backed PLE (27e437c — loest den 51-GiB-PLE-Klumpen an Layer 1),
   MTP5-Routen. Beim Rebase ersetzt deren offizielles qwen4_exp-Paket
   unsere Port-Dateien.
2. Bilanz der 83 Deploy-Ziele: **18 upstream unveraendert** (Patch 1:1
   portierbar), **58 geaendert** (Sichtung; davon ~18 mit Trivial-Diffs
   <= 20 Zeilen), 5 fork-eigene Dateien (kein Konflikt), 2 Kollisionen
   mit neuen Upstream-Dateien (qwen4_exp.py, spec_decode_qwen4_exp.py —
   unsere deploy_new-Eigenbauten treffen auf offizielle Pendants: unsere
   werden voraussichtlich GELOESCHT statt portiert).
3. Sichere Obsoleszenz-Kandidaten (Upstream hat unsere Backports/Fixes):
   mhc_triton.py (+446 upstream — war unser HEAD-Backport), grosse Teile
   von mhc_tilelang.py (nur Fork-Bloecke mhc_post_fp32/worker-lokal/
   hc_head<sm80 bleiben), vermutlich mehrere Device-0-Fixes (1.5.0-Log
   zeigt Fixes derselben Klasse: "Gate MTP5 NVFP4 by explicit
   capability", "scope KV scale gate to GPU workers") und der
   speculative.py-Idempotenz-Komplex (+269 upstream). Je Patch beim
   Rebase pruefen.
4. Schwere Rebase-Baustellen (Upstream-Basis stark bewegt, unsere
   Patches substanziell): fp8.py (+850), kv_cache_utils.py (+767),
   qwen_gdn_linear_attn.py (+709), gpu_model_runner.py (+621),
   gdn_attn.py (+597), short_conv_attn.py (+529), vllm_config.py (+396).
   Rohdaten: benchmarks/rebase-150-classify-2026-09-03.txt.

## Empfohlener Fahrplan

- **Phase 1 (Smoke, vor jeder Rebase-Arbeit):** Flash-Next-NVFP4-
  Checkpoint auf dem PUREN 1.5.0-Wheel (.venv-sm70-150, ohne unsere
  Patches) auf 2xV100 TP2 booten; Ziel: deren 82-tok/s-Klasse und
  Kohaerenz reproduzieren. Braucht den Checkpoint-Download (~95 GB,
  Platte hat 74 GB frei -> erst Platz schaffen, Peuqui-Freigabe fuer
  Loeschkandidaten einholen). Wenn Phase 1 gruen: Zielmodell laeuft,
  und der Rebase bekommt ein konkretes Serving-Ziel (2x2 mit RTX-Stufen
  = unsere sm75-/Device-0-Patch-Klasse obendrauf).
- **Phase 2:** Rebase nach Matrix: erst die 18+~18 leichten, dann die
  Obsoleszenz-Pruefungen (Punkt 3), zuletzt die schweren Baustellen
  (Punkt 4) — jeweils mit Kohaerenz-Gates. DSv4-PP5 bleibt bis zum
  Abschluss auf der 1.3.0-venv produktiv (kein Regressions-Risiko).
