# Morgen-Übersicht 28.09. — was die Nacht gebracht hat, was auf dich wartet

## Produktion
- Läuft auf Fork 3139141b, abgenommen 23:31 (prod_accept4: DSv4/FN/27B 8/8 gleich, Tempo 90/67-68/48 ms, 27B-Warmstart sauber). Tag verified-2026-09-27-pass7 (lokal).
- Seit gestern Mittag rund 360 Zeilen Fork-Code weniger, alles gemessen und abgenommen.
- 27B auf #604 (Warmstart-Fix drin), cuBLAS-Indexer bleibt an (62k: 90 statt 116 ms).

## Fork aufgeräumt (Branch cleanup-fork-2026-09-27, lokal committet, NICHT gepusht)
| Commit | Inhalt |
|---|---|
| c073d883 | Triton-3D-Spec + 64 Segmente zurück auf main (kein Eintrag nutzt TRITON_ATTN) |
| 6aa93144 | kv-Hash, Mamba-Blocktabellen-Kernel, SM75-Zweig, GDN-Chain-Fast-Build, DSpark-fp32-Aux zurück (alle ohne Nutzen gemessen) |
| 6459aede + f2d5d568 | hc_head: Kernel sättigt unter fp16, Torch-Fallback entfällt |
| 21395770 | P9 (Mamba-Kopierfunktionen je Typ, PLE-Spec im V1-Runner) zurück: Flash-Next läuft über V2 |
| 3139141b | klare Fehlermeldung statt einsum-Absturz bei QPN8_BLK=0 |
Tags lokal: verified-2026-09-27-pass6.

## Neue PR-Entwürfe (Branches lokal, Texte in upstream-contrib/03-1cat-issues/) — warten auf dein OK
| Paket | Branch / Worktree | Entwurf | Stand |
|---|---|---|---|
| P1 Indexer-cuBLAS unter Spec-Decode | sm70-indexer-cublas-spec-decode / 1Cat-vLLM-pr-p1 (147a7484) | pr-p1-indexer-cublas-entwurf-2026-09-28.md | V100 25 passed, Gegenprobe 6 failed; Zahlen beider Seiten |
| hc_head fp16 (Ergänzung zu #658) | sm70-hc-head-fp16-saturate / 1Cat-vLLM-pr-hchead (b79ed216) | pr-hchead-fp16-entwurf-2026-09-28.md | V100+RTX je 26 passed, ohne Fix 5 failed (inf) |
| B Sparse-MLA gather+BMM (Schalter, aus) | sm70-dsv4-sparse-bmm / 1Cat-vLLM-pr-bmm (4ed00965) | pr-bmm-sparse-mla-entwurf-2026-09-28.md | V100 34 passed; Decode nur ab ~16 Köpfen besser, Prefill überall 1,3-7x |
| P10 PLE-Offload unter PP (auf #646) | ple-offload-pipeline-parallel / 1Cat-vLLM-pr-plecascade (f996501d) | pr-p10-ple-offload-pp-entwurf-2026-09-28.md | 36 passed (2 GPUs), Gegenprobe 3 failed |
| A Skinny-Kernel nach csrc (ISSUE, kein PR) | – | issue-skinny-kernels-entwurf-2026-09-28.md | Frage an 1Cat vor großem PR |

## Entscheidungen für dich
1. Fork pushen (Branch-Stand 3139141b + Tag)?
2. Welche der vier PRs + das Issue senden? (Texte lesen; ich prüfe vor dem Senden erneut Upstream-Stand + Duplikate.)
3. Paket D (QPN8 Block-FP8): Schalter VLLM_SM70_QPN8_BLK ist im Mischaufbau nirgends ausschaltbar (V100-Stufen: Shape
   mismatch 65536/8192, RTX: jetzt klare Meldung). Entfernen oder A/B-Schalter lassen?
4. P7 (PP-Token-Prüfungen im V1-Runner, DSv4): als gekürzten Diagnose-PR einreichen oder im Fork lassen?

## Befunde der Nacht (Details in pr-plan-2026-09-27.md)
- cuBLAS-Indexer: konstante Kosten (volle Graph-Breite), Schnitt ~18k; an lassen.
- BMM gegen 1Cats besten Decode-Kernel: 8 Köpfe/Karte langsamer, 64x6 bis 4,3x; gegen main-Standard 13-45x.
- Model Runner: nur Flash-Next V2, DSv4/27B V1.
- Warmstart: vLLM lädt AOT-Artefakte auch nach Python-Codeänderungen -> Abnahmen mit eigenem VLLM_CACHE_ROOT.
