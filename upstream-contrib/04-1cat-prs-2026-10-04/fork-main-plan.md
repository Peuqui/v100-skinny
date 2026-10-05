# fork-main — Zusammenstellung (Nacht 03./04.10.2026)

Basis: 1Cat main f551e0afe (enthält #856, #857, #831 ple_result_transport, #809 GGUF-Lader).
Worktree `1Cat-vLLM-fork-main`, Branch `fork-main`, nativer Bau sm_70 (`build-fork-main-2026-10-03.log`).

## Obendrauf (Reihenfolge)

1. PR A `pr-qpn8-volta-copy` 8cdfb00e2 — Block-QPN8 auf Volta: Warmup über gehaltenes Layout, Volta-TurboMind-Kopie
   nur wenn Decode > 8 Zeilen (Feld `sm70_fp8.block_qpn8_volta_turbomind_prefill`, None = Automatik).
2. PR B `pr-dsv4-spec-context-buckets` 48f0cb970 — DSv4-Kontext-Buckets auch bei Spec-Decode, cuBLAS-Indexer im Bucket.
3. PR C `pr-ple-disk-fast-path-main` 661935b83, 18fbc21b4, 60ee0e43c — PLE-Plattenstufe: Profil, Seiten gebündelt +
   Seitentabellen-Freigabe (`ple_disk_release_pages`), Zeilen-Cache (`ple_disk_row_cache_gib`).
4. #715 b79ed216c (offen bei 1Cat) — hc_head endlich unter FP16.
5. Fork-intern (kein PR):
   - 2ccdc953c Entwurfsvokabular (Backport vllm#59740, `draft_token_map`) — PR erst, wenn 1Cats #821 entschieden.
   - 841656ec4 FLA-Shared-Memory pro Gerät — auf unserem Rechner wirkungslos, 7 Zeilen (AGENTS.md: kein Einzel-PR).
   - 533feecc9 + 20618af17 Version nur aus vX.Y.Z-Tags (wegen unserer verified-*-Tags).
   - 368e74fdf Doku gemischter Volta/Turing-Rechner.
6. Offen/prüfen:
   - 64ad1de8c Block-Packing QPN2 (#611 → 1Cats #822, Draft): ohne ihn Flash-Next/27B-NVFP4 evtl. langsamer → messen.
   - 399088b93 erzwungener Compile-Cache-Ausstieg Flash-V100-Graph: main hat ihn noch → Wirkung auf Bootzeit prüfen.

## In main bereits enthalten (1Cat-Fassungen)

Kaskade (#806), QPN8-Block (#801), Sparse-BMM + cuBLAS-Indexer (#816), PP-Token-Check (#771), MXFP4-Repack (#767),
SWA-Totblöcke (#765), Graph-Deckel (#762), AOT-Reload-Workspace, ROCm-SWA, FP8-Software-Decode, Test-Skips,
Draft-Zeilen-Assert, QSA-Workspace, Torch-Backport, Recovered-Token-Fix, PLE-Anon-Speicher, Skinny (#742/#837).

## Offene Befunde

- DSv4 Prefill bei 124k: main ~74 s gegen Produktion ~66 s (ab_837n, Gesamtzeit je Nadelanfrage) → zerlegen.
- DSv4 erste Code-Anfrage nach Boot 2,9 s TTFT (main, auch ohne PR B) — einmalig je Prozess, Ursache offen.
- tests/config: 3 Fehlschläge auch auf cef0a2e4b (config_generation empty-vs-unset, dflash2 hash, loaded routes) +
  Absturz des RTX-Laufs → auf fork-main-Bau nachprüfen, ggf. beheben.
- #852 async scheduling bei PP>1: A/B auf Produktion (ab_async.sh fertig).

## Abnahme fork-main gegen Produktion

Alle vier Modelle (DSv4 PP5, Flash-Next PP4, Flash-Next TP2×PP2, 27B TP2): Greedy, Langprompt, Decode, Qualität von Hand.
Einträge ohne Env-Altlasten, Schalter als `--kernel-config`.
