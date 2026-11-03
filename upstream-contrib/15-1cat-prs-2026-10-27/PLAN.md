# PR-Runde 27.10.2026 (Peuqui: „unsere PRs zack zack alle einreichen“)

Basis: 1Cat main 6a143197e (seit 23.10. unverändert), unsere Linie `prod-v4` 29c31d0f2 = 97 eigene Commits darüber.
Zuordnung per Patch-ID/Betreff zu den 22 offenen PRs: `scratchpad/map_commits_to_prs.sh` (Sitzung 27.10.).

Regeln je PR (AGENTS.md + neue Vorlage mit „Path coverage“ und „Acceleration and benchmark contract“):
Zweig auf frischem `origin/main`, nur die Commits des Pakets (Konflikte auflösen, keine Upstream-Änderung zurücksetzen),
ganze betroffene Testdateien laufen lassen (CPU; GPU auf V100 PCI 1 und RTX PCI 2 im Fenster), `pre-commit run --from-ref
origin/main --to-ref HEAD`, Duplikatsuche (offene PRs, Stichworte), Text Englisch ohne Markdown-Auszeichnung, jede Zahl aus
einer Messdatei, KI-Hinweis, Autor Peuqui noreply; Zustände direkt vor dem Senden neu abfragen, danach Live-Text prüfen.

## Neue PRs (Reihenfolge = Senden)

| # | Zweig | Inhalt | Commits | Abhängig | Tests |
|---|---|---|---|---|---|
| 1 | pr-mla-kv-codec | Bugfix: MLA-Spezifikationen tragen den Codec ihres Cache-Typs (GLM-5 FP8-KV bricht beim Boot ab, DSv4-SWA läuft still im Paged-Fallback) — vorher auf 1Cat main belegen | 29c31d0f2 (Nettoeffekt mit 16023490c) | – | CPU |
| 2 | pr-test-main-red | Zwei auf main rote Tests (V2-MTP-Profilbericht je TP-Rang; mixed_prefill_budget im Abbruch-Fixture — Doppelung mit #1260 prüfen) | 8f62d5056, 457ba3151 | – | CPU |
| 3 | pr-build-sm75-family | Native sm_75-Code für die SM70-Kernfamilie | 2f70712b0, 5028a3fc4 | – | Bau |
| 4 | pr-qpn8-turing | QPN8-Block-FP8 auf Turing-Tensorkernen, Split-K nach Belegung | 7f343a791, 40cea0e5b, 4da0bcfa4, 6c0ef8efb, 0314585ac | 3 | GPU |
| 5 | pr-skinny-moe-turing | Skinny-MoE: Grid min(S,E), m16n8k8 für SM75, Volta-Kacheln | 79e07bfef, 039c1b6af, 7ee89582f, 993f69f5d | 3 | GPU |
| 6 | pr-dsv4-prefill | DSv4-Prefill: Indexer-Epilog, Sparse-Softmax, Turing-Indexer, Logits je Teilchunk, KV-Gather, Turing-Ein-Pass | 5669da106, e1cb50fd9, d77726c11, 7d83b7032, cb8272ee1, dee5a6e48, a1ac0c89d | 3 | GPU |
| 7 | pr-dsv4-woa-tuning-freeze | DSv4 wo_a gruppiert, TurboMind-Tuning nach Warmup eingefroren | fd5e8a647, e156df3dc, a082b8d19, 1addd22db, 25df27f15 | – | GPU |
| 8 | pr-tp-prefill-overlap | TP-Prefill: Allreduce-Überlappung, Copy-Engine-Host-Reduce, Turing-Split-KV-Prefill, Turing-NVFP4-Kachel, Paged-Prefill hd256 | 88d25b8bd, c1816294c, afe87abd0, f21d1c564, f9ac81b6b, 3856a7b67 | #1216 (Konflikt) | GPU |
| 9 | pr-pp-send-overlap | PP-Senden eines Eager-Schritts überlappt den nächsten | c80e83f8a | – | CPU/GPU |
| 10 | pr-loader-fixes | Index-Shards in Unterordnern, U16/U32 per Direct-I/O, voll genauer lm_head bei ModelOpt | 85b95af2c, 434b350c2, 08fbcdf68 | – | CPU |
| 11 | pr-mla-workspace | MLA-Chunked-Prefill-Workspace und Profil-Reservierung, Indexer-Puffer an max_num_seqs | 7104b7d46, 0cac9351d, 4342ad6b6, 9c3e09130, d975ae5f3 | – | CPU/GPU |
| 12 | pr-glm53 | GLM-5.3-Flash auf SM70/SM75 (Weg B), CSF-Kern, S5-Ebenen, Sparse-MLA-Prefill, Doku PP5+DFlash2 (Überschneidung #1260 prüfen) | 4417a4c04, 5cc9779e1, 58e0e6cde, 37143bf1d, 076e37e67, 422184292, 0b8b98c4f, ba941246d, db216c203, 31f961300, e6483cf55, 3b49b07ba, 64cca01f4, 5fef3ad60, 16d886e8d, ea42af1be | 3, 11 | GPU |
| 13 | pr-mxfp8-qpn8 | MXFP8 als 8 Bit im QPN8 (Skalenmodus, Turing-Kern, GLM-MLA-Projektionen) | 768d1a31b, df65c5ab1, c213a336e, bf807861b | 3, 4, 12 | GPU |
| 14 | pr-lookahead-block-hashes | Lookahead-Block-Hashes (Rückport vllm#61589), DSpark behält Drop; Bezug #1240, #1265 | 4651800f7, 45dc306a5, 23e34b03c | #1240 | CPU |

Nicht dabei: 58415430e (vorübergehende Diagnose), 5bbe3fa84 (Logzeile, ggf. in 11), a3ae1faf3 (gehört zu #1221).

## Stapel (27.10. ~10:50, alle geprüft mit `stack_vs_prod_2026-10-27.py`: 0 unerklärte Abweichungen zum Produktionsbaum)

| Zweig | Basis | eigene Commits | darin offene PRs | Text |
|---|---|---|---|---|
| pr-mla-kv-codec | main | 1 | – | GESENDET #1272 |
| pr-turing-kernels | main | 11 | – | body-turing-kernels.md |
| pr-tp-prefill | pr-turing-kernels | 6 | #1216 (rebased), #1227 | – |
| pr-dsv4-prefill-stacked | pr-tp-prefill | 6 | #1222 | – |
| pr-triton-sm7x | pr-dsv4-prefill-stacked | 10 | – | – (braucht Triton-Fork) |
| pr-glm53 | pr-triton-sm7x | 21 + Loader 3 + MLA 5 | – | – (12 deutsche Nachrichten übersetzen) |
| pr-dsv4-woa-freeze | main | 5 | – | body-dsv4-woa-freeze.md |
| pr-lookahead-on-1240 | main | 3 | #1240 | body-lookahead.md |
| pr-loader-fixes | main | 3 | – | body-loader-fixes.md |
| pr-mla-workspace | main | 5 | – | body-mla-workspace.md |

Noch ohne Zweig: c80e83f8a (PP-Senden überlappt), 5bbe3fa84 (KV-Speicher-Logzeile). Nicht angeboten: 58415430e (Diagnose).
Hilfsskripte: `pr_branch_2026-10-27.sh`, `pick_continue_regen_2026-10-27.sh`, `pick_take_fork_state_2026-10-27.sh`, `reword_commit_2026-10-27.sh`, `stack_vs_prod_2026-10-27.py`, `build_pr_2026-10-27.sh <wt> [FA-Archs]`, `guard_run_2026-10-27.sh`.

## Stand 27.10. ~10:40

- Gesendet: #1272 (MLA-KV-Codec, nach Zweitprüfung korrigiert).
- Turing-Paket: Pakete 3+4+5 sind zu EINEM Zweig `pr-turing-kernels` zusammengelegt (Abhängigkeiten über CMake/ops.h), 11 Commits auf main, gebaut, Tests RTX/V100 grün bis auf einen wackelnden 1Cat-Test (FP8-Laufzeit-Tuning, #1187). Offen: `fp8_qpn8_mma_sm75_out` in `CONFIGURED_OPERATORS` aufnehmen (aus dc4db4903 des wo_a-Pakets hierher verschieben), `test_sm70_native_policy.py` laufen lassen, pre-commit, Zweitprüfung. Text `body-turing-kernels.md`.
- DSv4-Prefill-Paket (6) hängt an Turing-Zweig + f9ac81b6b (TP-Paket) + #1222 → nach dem TP-Paket stapeln. a1ac0c89d gehört zum Triton-Paket.
- wo_a-Paket: auf main übernommen; offen: `fp8_qpn8_mma_sm75_out`-Zeile entfernen, Bau (gemm_runtime.cpp), Tests, Text.
- Loader-, MLA-Workspace-, Lookahead-Zweige auf main übernommen (Loader: Konflikt modelopt.py gelöst; Lookahead: Referenz neu erzeugt); deutsche Commit-Nachrichten vor dem Senden englisch zusammenfassen.
- Rebasete Alt-PRs #1216/#1217/#1218/#1235 (nur Referenzdatei bzw. Ignore-Liste neu): Tests, dann Force-Push mit Archiv-Tag der alten Köpfe.
- 2. Test-PR (8f62d5056) entfällt als Einzel-PR; 457ba3151 ist Teil von #1260.

## Stand 27.10. ~11:10 (neue Sitzung, während der GLM-A/B-Reihe b nur leichte Arbeit)

- Turing-Zweig: `fp8_qpn8_mma_sm75_out` per Fixup in den Commit des Operators (jetzt 77006c020) in `CONFIGURED_OPERATORS`;
  wo_a-Zweig: die Zeile entfernt, Commit-Nachricht auf „the missing operator“ (jetzt d8ce4ca70). Beide nur lokal.
- GLM-Stapel: alle zwölf deutschen Nachrichten englisch (`reword_many_2026-10-27.sh`, Baum je Commit unverändert);
  check-layering `csf32_expand.py` 0→1 behoben durch Wrapper `ops.csf32_expand` in `_custom_ops.py` (eigener Commit 9cf945ff8,
  Kern ist reines CUDA ohne Tensorkern-Befehle). Im Triton-Zweig `skinny_moe_triton.py` 0→1 (Kommentar „V100“) per Fixup
  umformuliert. **Diese zwei Änderungen auch in fork-main-v4 nachziehen** (nach der Messreihe, v4-dev wird gerade gemessen).
- check-layering `moe_runner.py` 37→39 ist auf 1Cat main 6a143197e selbst rot (gleiche Datei, gleiche Baseline) — nicht von uns.
- Stapel muss neu aufgesetzt werden (Turing- und Triton-Zweig geändert): `restack_2026-10-27.sh` (läuft nach dem T#3-Boot).
- Duplikatsuche: alle offenen PRs per Dateiliste verglichen. wo_a: #903 (areslp) repariert `test_sm70_warmup.py` ebenfalls auf
  `native_policy` → im Text genannt. MLA-Workspace, Loader: keine Überschneidung. Lookahead: #1240/#1265 bekannt.
- **Vor dem Senden jedes PRs: Commit-Hashes im Text gegen den Zweig abgleichen** (Turing-Text nennt noch 025bd2d63, 603e17908,
  c8dd9bba5 = alte PR-Hashes; jetzt 77006c020, 268d30f7d, 72e4eb0df; MLA-Workspace-Text nennt e87817494 usw.).
- Tests nach Ende der Messreihe: Turing + wo_a `test_sm70_native_policy.py` (Operatorliste geändert), GLM `test_csf32_scales.py`
  und `test_skinny_sm70_moe.py` (CSF über den neuen Wrapper), dann pre-commit je Zweig.

## Freigegeben von Peuqui (27.10. ~07:30)

- Rebase mit Force-Push der vier konfliktbehafteten offenen PRs #1216, #1217, #1218, #1235 (8 hängt an #1216).
- Triton-Paket (28fc41221, 43ecd2ac8, 5a11342ea, 7458c6257, 7b8fb5387, Triton-Skinny-MoE 64009b8dd, 00d06e31c, 7ea6a683a,
  1f9de3af9, 9712e0edf) als 1Cat-PR (Nr. 15); dafür den Triton-Patch als Fork auf Peuquis GitHub veröffentlichen und bei
  Triton upstream einreichen. Vorher: Lizenzvermerke, keine internen Pfade/Adressen, Secrets-Scan.
- Jeder PR doppelt geprüft vor dem Senden: zweiter Durchgang über Diff (Revert-Prüfung gegen origin/main), Tests, Text und
  jede Zahl gegen ihre Messdatei.

## Stand 27.10. ~14:10 (nach dem Absturz durch volle Platte, neue Sitzung)

- Alle acht Zweitprüfungen lagen vor dem Absturz vor; Zusammenfassungen in `review/`, Volltexte im Scratchpad der Sitzung b66b4495.
- GESENDET (je pre-commit nur mit dem auf main roten check-layering-Fund moe_runner.py 37 -> 39, mypy-3.10 grün, Live-Text = Datei):
  #1273 wo_a/Freeze (Kopf ad925b180; neu: Schaltertest `_freeze_tuning_if_requested`, mypy-Fix im Freeze-Test, 14 statt 13 main-Fehler,
  #828 im Duplikat-Absatz), #1274 Loader (4998496f3), #1275 MLA-Workspace (e41ef0048), #1276 Lookahead auf #1240 (8a1a11bc6; PP-Tests
  brauchen zwei sichtbare Karten, auf main ebenso).
- Fork-Korrekturzweig `review-fixes-2026-10-27` (gepusht, Kopf de9382483): TP-Fixes, Turing-Seitenprädikat, CSF-Wrapper, MLA-DCP,
  Freeze-Doku, Freeze-Testfix, Freeze-Schalterfunktion. Noch nicht in fork-main-v4 übernommen.
- TP-PR-Zweig: die drei TP-Korrekturen als Fixups eingefaltet, Referenzdoku neu (Kopf 3abccc05b), noch nicht getestet; wird nach den
  Turing-Korrekturen neu gestapelt.
- Force-Push #1263 (Signoff): Archiv-Tag `archive-pr1263-2026-10-27` gepusht, nachsignierter Commit lokal `pr1263-signoff` (7621e13a2);
  der Push selbst wurde von der Sitzungs-Berechtigungsprüfung abgelehnt → Peuqui führt ihn aus.
- Triton-Fork: sechs private Verweise umgeschrieben auf `sm7x-tensor-cores-en-v2` (9063e502c, 23 Commits auf v3.6.0, Bäume gleich),
  Archiv-Tag `archive-sm7x-tensor-cores-2026-10-27` gepusht; Force-Push durch Peuqui. Danach „Show and tell“-Discussion
  (Entwurf `triton-show-and-tell-draft.txt`, Peuqui 27.10.: erst Show and tell, dann abwarten, ob ein Wartungsangebot sinnvoll ist).
- Offen: Turing-Korrekturen (Split-8-Regel nur für M <= 8, __trap im #else, sm_75-Begründung, Texte), dann TP, DSv4-Prefill (Blocker:
  Commit-3-Pfad nie gebootet), Triton-Routen, GLM; Alt-PRs #1216/#1217/#1218/#1235 rebasen und force-pushen.
- 14:40: Peuqui hat beide Force-Pushes ausgeführt (#1263 = 7621e13a2, signiert; Triton-Fork sm7x-tensor-cores = 9063e502c, 23 Commits
  auf v3.6.0 ohne private Verweise). „Show and tell“ gepostet: https://github.com/triton-lang/triton/discussions/12517 (Live-Text =
  `triton-show-and-tell-draft.txt` ab Zeile 4). Jetzt abwarten, ob Rückmeldungen ein Wartungsangebot nahelegen (Peuqui).
- CI pre-commit von #1272–#1276 rot wie main selbst (Lauf 40009610580 auf 6a143197e: mypy gguf_turbomind_moe.py:609,
  kv_cache_interface.py:156–158, check-layering moe_runner.py); in #1273–#1276 im Text vermerkt.
- 15:35: Turing-Zweig überarbeitet (9 Commits, signiert; Split-Wahl im Op, Trap, can_implement je Gerät, ISO-Daten, TurboMind-Vermerk;
  Archiv lokal `archive-pr-turing-kernels-vor-review`), Text v2 vom Agenten (`body-turing-kernels.v2.md`, Quellen
  `review/turing-kernels-v2-sources.md`) noch an Op-Split, can_implement und sm_75-Messung anpassen. Stapel neu gesetzt
  (`restack_2026-10-27b.sh`; Fehlbedienung: zweiter Lauf auf TP abgebrochen, Ergebnis zeilengenau geprüft): TP dc10e4651, DSv4 8f2c40593,
  Triton 97f1152d7, GLM 058e3cd88. GLM-Stapel trägt noch die alten Loader-/MLA-Commits → an #1274/#1275 angleichen.
- sm_75-Familienbau: Variante `1Cat-vLLM-sm75trim` im Bau; danach 27B-A/B (Peuqui: „messen und fixen“). Ergebnis bestimmt auch den
  Build-Commit des Turing-PRs.
- 18:40: GESENDET #1278 Turing-Kernel (Zweig pr-turing-kernels, 7 Commits auf main 899e9ed33, Kopf 001b73e3e; Text `body-turing-kernels.v2.md`):
  Build-Commits entfallen, sm_75 nur für fp8_qpn8_mma_sm75.cu und skinny_moe_qpn_sm70.cu (A/B 27B: Familienbau bringt nichts, Peuqui „Passt“).
  Tests RTX/V100 auf neuem main, Fehlschläge = main (fp4_mainflow 168 nach #1277, native_policy 1, V100 TurboMind-Vergleich M=64 5/5 auch auf main).
- 1Cat main ist auf 899e9ed33 (#1277 DeepSeek-V4.1-Port, 197 Dateien). #1273 und #1276 haben dadurch nur einen Konflikt in der generierten
  runtime_reference.md (erst auf Nachfrage lösen; Force-Push nur durch Peuqui). #1272/#1274/#1275 konfliktfrei.
- Lokaler Zweig `pr-mla-kv-codec` (Testbasis) steht auf 899e9ed33 + #1272-Commit f4d5a4e6a — NICHT pushen (#1272 bleibt 4fbaef265).
- Nächstes: Stapel TP → DSv4-Prefill → Triton → GLM auf #1278 (001b73e3e) setzen; TP-PR-Text nach Prüfung, Tests, senden.

## Stand 27.10. ~19:30

- GESENDET #1279 TP-Prefill (Zweig pr-tp-prefill, 16 Commits auf main 899e9ed33 = #1278 7 + #1216 1 + #1227 2 + 6 eigene, Kopf 0d83b1b95;
  Text `body-tp-prefill.v3.md`, Live-Text = Datei). Eigene Commits: 40831e96a Attention, 9db824d54 Flash-V100, 4ce181c53 Überlappung,
  1c2d22880 Copy-Engine, e8f5c6e00 Split-KV, 0d83b1b95 NVFP4-Kachel; Nachrichten aus `review/tp-commit-messages/` (+ Granite, 3. host_reduce-Test).
- Vor dem Senden eingefaltet: Granite-4-Vision-Deepstack-Ablehnung (`_ds_layer_indices`) + Test, Config-Test copy_engine_max_bytes > max_bytes,
  Docstrings „about 23 … 61 TFLOPS“ / „about a third“, und neu: check-torch-cuda-call meldete `torch.cuda.set_device`/`current_device` in
  prefill_overlap.py und im Test → `torch.accelerator.set_device_index`/`set_stream`/`current_device_index`.
- Schlusslauf (`prtests_2026-10-27/tp-final2-*`, `tp-accel-*`): RTX/V100 grün bis auf runner_v2 2 (= main) und GQA-Verify-Skip (kein FA2 für
  SM75/SM70 im Bau); Paar host_reduce 3 passed. pre-commit (`tp-precommit-final2.log`): nur check-layering moe_runner.py 37→39 (= main), mypy-3.10 grün.
- Fork: beide Korrekturen in `review-fixes-2026-10-27` (087ec5105, 22bae4f3d, gepusht).
- #1278: doppelte Zeile „Unit tests on this branch (built in place):“ aus Datei und Live-Text entfernt.
- Nächstes: Stapel DSv4-Prefill → Triton → GLM auf den neuen TP-Kopf 0d83b1b95 setzen (alter TP-Kopf dc10e4651).

## Stand 27.10. ~19:45 (DSv4-Prefill)

- Stapel neu gesetzt (`restack_2026-10-27c.sh`, zeilengenau = TP-Nachtrag): DSv4 3265c8fa6, Triton 9d4207098, GLM e3e551c97.
- DSv4-Zweig (Baum `1Cat-vLLM-pr-dsv4-prefill2`), Prüfpunkte eingefaltet, Kopf bf311faf2 (Nachrichten noch alt):
  Blocker 1 Routen-Test ohne `sparse_tensor_cores`; Blocker 2 fp16-Referenz (`_reference_prefill_logits`, `_assert_close_to_max` aus
  e1e3855cb) — beim nächsten Restack des Triton-Zweigs Konflikt in test_deepseek_v4_sm70_indexer.py erwartet (Fassung von e1e3855cb nehmen);
  Kern: `q.stride(0) % 2 == 0` + Kopfkommentar; Routenlog je Prefill-Weg (Scope muss `process` werden, sonst nur Stufe 0 → nach dem Mini);
  Doku `indexer_prefill_cublas`; `del logits` auch im Kpool-Indexer; Gather-Kommentar mit 280/345 GB/s.
- Gebaut 19:26 (`_C` 73 × sm_70, 4 × sm_75). Tests `prtests_2026-10-27/dsv4-{rtx,v100}`: alle grün.
- Neue Nachrichten: `review/dsv4-commit-messages/*.txt` (nach dem Mini einspielen). Text-Entwurf `body-dsv4-prefill.v2.md` (Platzhalter).
- Blocker 3: Mini-A/B `dsv4-mini/mini_prdsv4_2026-10-27.log` (PR-Baum ohne sm7x-Triton, Varianten indexer-prdsv4 old/new = ohne/mit
  fusioniertem Turing-Zweig), danach Vollboot `validate_prdsv4_2026-10-27.sh` (Route, 8 Antworten, Langkontext).
- Duplikate: #828 nur QSA (GitHub-Dateiliste veraltet), #903/#1260 Konflikte bestehen schon ohne die sechs Commits; #1260 inhaltlich (FP32-Scores).

## Stand 27.10. ~20:45

- GESENDET #1280 DSv4-Prefill (Zweig pr-dsv4-prefill-stacked, 25 Commits auf main 899e9ed33 = #1279-Stapel 16 + #1222 3 + 6 eigene,
  Kopf 437ad0997; Text `body-dsv4-prefill.v2.md`, Live-Text = Datei). Eigene: 77fd5d11c Epilog, 459edf6a6 Softmax, 1ba5d920c Test,
  37febaf5a Turing-Scorer, 84777632d Logits, 437ad0997 Gather.
- Belege Blocker 3: Mini-A/B `dsv4-mini/mini_prdsv4_2026-10-27.log` (150k −8,9 %, 280k −15,2 %, Greedy gleich), Vollboot
  `validate_prdsv4_2026-10-27b.log` (Routen fused auf cuda:0/4, cuBLAS auf 1–3; 8 Antworten richtig; Langkontext bis 302 833 Token).
  Vollboot ohne DSpark: Model Runner V2 auf main lehnt 'dspark' ab. Decode-Schritt auf main wächst stark mit dem Kontext (115 → 269 ms,
  Fork 58 → 70 ms) — Ursache nicht gemessen, nicht von diesem PR.
- Neu: `info_once` loggt mit Standard-Scope nur auf dem ersten lokalen Rang (Stufe 0) → Routenzeilen mit `scope="process"`.
  Die Decode-Routenzeile auf main (624c83669) hat denselben Mangel (Folgepunkt).
- Schlusstests `prtests_2026-10-27/dsv4-final-*` grün, pre-commit `dsv4-precommit-final.log` nur check-layering (= main).
- Offen: Fork-Nachzug (v4-fix), Restack Triton/GLM auf 437ad0997 (Konflikt test_deepseek_v4_sm70_indexer.py und indexer.py durch e1e3855cb).
- 20:47: Stapel auf #1280 umgesetzt (von Hand, Backups `backup-pr-triton-vor-restack-d-2026-10-27`, `backup-pr-glm53-vor-restack-d-2026-10-27`):
  Triton-Kopf 27ebfc494-Nachfolger (pr-triton-sm7x), GLM (pr-glm53) exakt um das Triton-Delta verschoben. Konfliktlösungen:
  indexer.py Routenzeilen vereinheitlicht („Triton scorer on … (tensor cores: …)“, scope process), Indexer-Test = Fassung e1e3855cb,
  Routen-Test: `sparse_tensor_cores = False` + Tensorkern-Fall jetzt in 129410be0-Nachfolger (perf(dsv4): sparse prefill matmuls … tensor cores),
  9d4207098-Nachfolger passt die Durchgangsgröße an (Test im Zwischenstand grün).
  Für den Triton-PR noch: Docstring indexer_prefill_cublas um „mit triton_sm7x_tensor_cores nimmt der Triton-Scorer alle Chunks“ ergänzen.
- Fork `review-fixes-2026-10-27` 8429fb0a8 (DSv4-Korrekturen, Dateien gleich dem PR-Stand; `_C` dort nicht neu gebaut).
- 20:55: Triton-Zweig bereinigt (Kopf 43403996d): interne Messnamen/Daten aus zwei Docstrings, fremde Lambda-Umformatierung im
  Routen-Test zurück; GLM nachgezogen (30 Commits, Delta = Bereinigung). Offene Entscheidungen Peuqui für den Triton-PR:
  (a) Soll 2 Turing-Indexer mit Schalter: nativer SM75-Scorer summiert schneller (64 Köpfe Triton +1,0 %, 32 Köpfe +4,7 %) →
  Empfehlung: Turing behält den nativen Scorer; ändert die gemessene RTX-Route → neues DSv4-A/B nötig.
  (b) Soll 7 Durchgangsgröße nach Architektur (Turing 1024, Volta 128, begründet mit 48/32 GiB): ein Durchgang wäre auch auf V100
  13 % schneller, kostet 378 MiB Workspace je V100-Stufe → Empfehlung: ein Durchgang auf allen Karten, vorher Vollboot mit Pool 3500 prüfen.
  Danach: Texte (Volta-MoE-Endzahlen, mHC-Formulierung, Commit-3-Ausgabeänderung, A/B-Offenlegung, Abhängigkeit Commit 6 → 3),
  Signoff aller zehn, Schwelle 16 Zeilen/Experte auf dem Endkern nachmessen oder offen sagen.
- 22:09: A/B `ab_tfix_dsv4_2026-10-27a.log` (T = prod-v4b + beide Entscheidungen, Baum `1Cat-vLLM-v4b-tfix` c61972d77; P = Produktion;
  T P T P): Prefill 178k 41,40/41,34 → 41,18/41,17 s (−0,5 %), 287k 74,04/74,02 → 73,45/73,43 s (−0,8 %), 8,7k gleich; Decode kurz gleich,
  nach 14k 53,45/53,55 → 55,13/55,04 tok/s; GSM8K 29/30 überall, Annahme 4,118 überall; 8 Antworten T1 byte-gleich mit der Kontrolle;
  V100-Spitzen 32 048–32 108 → 32 488 MiB (KV verfügbar 1,74 → 1,39 GiB je V100-Stufe, Pool 317 261 Token gleich, ≈ 280 MiB Rest).
  Produktion NICHT umgestellt: indexer.py/sparse_prefill_bmm.py betreffen auch GLM (Turing-Scorer nimmt GLMs 32 Köpfe) → erst GLM-A/B.

## Stand 27.10. ~23:20

- GESENDET #1282 Triton-Routen (Zweig pr-triton-sm7x, 34 Commits = #1280-Stapel 25 + 9 eigene, Kopf 0356875b6; Text `body-triton-sm7x.v2.md`).
  Eigene: e9dbf58e5 Schalter (Refactor eingefaltet), e2ad1f3ad Scorer (Turing behält SM75-Scorer), a5597e08b Reduktion, 7a1c01d83 mHC,
  fd303bbcd Sparse-TC, ceef5239d/919403d60/c51b1d40b Volta-MoE (Schwelle 16 → 13, nachgemessen `triton-sm7x/moe_threshold_2026-10-27{,b}.log`),
  0356875b6 ein Durchgang auf allen Karten. Belege: Schlusstests `triton-final-*` (44 Dateiläufe grün), Vollboot `validate_prtriton_2026-10-27.log`
  (Routen + mma.sync in allen neuen Triton-Kernen, 8 Antworten richtig, Langkontext bis 302 833), Entscheidungs-A/B `ab_tfix_dsv4_2026-10-27a.log`.
- Fork `review-fixes-2026-10-27`: 1a6bdea6a (Entscheidungen), 569d11c56 (Schwelle 13), 89ce1e78b (Kommentar ohne Kartennamen).
- GLM-Stapel (pr-glm53, 30 Commits) sitzt auf 0356875b6. Nächstes: GLM-PR (review/glm53.md).

## Stand 28.10. ~06:20 — Merge-Welle von 1Cat (yangzhuxinyzx, 28.10. 03:23–03:31)

- main 899e9ed33 → c59b3c4cb (27 Commits). GEMERGT (Squash, je patch-gleich mit unserem PR-Kopf, nichts von 1Cat verändert):
  #715, #1183, #1213, #1214, #1219, #1222, #1226, #1227, #1229, #1234, #1237, #1240, #1263, #1264, #1272, #1274, #1275.
  Issue #739 als erledigt geschlossen. Keine Reviews, keine Kommentare von 1Cat. Fremd gemergt: #1265 (disable_eagle_block_drop,
  Bezug #1276), #1244/#1232/#1230/#1199/#1207/#1247 (Leon), #1267 (SM70 Top-k/Top-p auf dem Gerät), #1201/#1200 (tenotek).
- Offen von uns noch 14. Probe-Merge (`git merge-tree` gegen c59b3c4cb): sauber #1278, #1236, #1228, #1221, #1187; Konflikt
  #1279/#1280/#1282 (Stapel: runtime_reference.md + attention.py/flash_attn.py neben #1227), #1276 (runtime_reference.md,
  kv_cache_coordinator.py; erster Commit = #1240), #1273, #1235, #1217, #1218 (runtime_reference.md), #1216 (+ config/kernel.py).
  GitHub zeigt #1279/#1280 noch als mergeable — lokal widerlegt.
- Folgen: Stapel schrumpft beim Rebase (#1279 ohne #1227, #1280 ohne #1222, #1276 ohne #1240, GLM-Basis ohne #1274/#1275);
  Texte von #1276/#1279/#1280/#1282 nennen die gemergten PRs noch als Teil des Stapels. #674 ist seit 25.10. nicht nachgezogen
  (fehlen #1272–#1282 und die 17 Merges). Saubere PRs nach dem Sprung mit ihren Tests auf dem Probe-Merge prüfen (Lehre #1228).

## Stand 28.10. ~07:25 — alle Konflikt-PRs neu aufgesetzt, gepusht, Texte nachgezogen

- Rebase auf c59b3c4cb (`~/.cache/bench-scripts/rebase_c59_2026-10-28.sh`, Sicherungen `backup-<zweig>-vor-c59-2026-10-28`, Archiv-Tags
  `archive/<zweig>-vor-c59-2026-10-28` im Fork = öffentliche alte Köpfe). Weggefallen: #1227 (TP), #1222 (DSv4), #1240 (Lookahead),
  #1214 (PLE-Überlappen), #1274/#1275 (GLM-Basis). range-diff: nur Referenzanker (main selbst war veraltet: SpeculativeConfig-Anker von #1265).
  Revert-Prüfung `revert_check_c59_2026-10-28.py`: keine Upstream-Zeile zurückgesetzt.
- Neue Köpfe (gepusht 07:10–07:20): #1278 c15d1446f, #1216 01f1c7d2b, #1279 3f483154e (14), #1280 6dc35fdda (20), #1282 dbf0f3c3e (29),
  #1273 9a71c5c62, #1235 d1c15914b, #1217 27a394faf; #1276 bd3b78bcd und #1218 7308b33f9 noch einmal auf abec8ba85 (#1284 von 1Cat,
  06:43: pynccl-All-Reduce wieder out-of-place, Referenz regeneriert, speculative.py-Kopplung behoben). GLM pr-glm53 502d1b546 auf dbf0f3c3e.
- Tests `prtests_2026-10-28/` (Beleg `upstream-contrib/16-rebase-c59-2026-10-28/test-results-2026-10-28.txt`): jeder rote Testfall
  gleich main (Hash der roten IDs), auch die Probe-Merges #1236/#1228/#1221/#1187. pre-commit: nur check-layering = main.
- Texte: Update-Absatz + ersetzte Stapel-Sätze (`16-rebase-c59-2026-10-28/make_bodies.py`, `make_bodies_abec.py`), Live = Datei.
- 07:25: alle 14 offenen PRs sauber gegen main abec8ba85. Offen: #674 nachziehen, GLM-Vollboot + Text + Senden.
- 07:30 #674 nachgezogen (Text `16-rebase-c59-2026-10-28/body-674.v2.md`, Live = Datei): 17 nach „Merged“, #1273/#1276/#1278–#1282 neu offen, #739 geschlossen.
- GLM-PR vor dem Boot: Zweig pr-glm53 502d1b546 (27 Commits auf #1282 dbf0f3c3e). Übergabe-Fund `kda.py` 349/356/387 (`get_device_capability()`
  ohne Index) ist KEIN Fehler: CudaPlatform.resolve_device_id(None) gibt im Worker torch.cuda.current_device() zurück (unser #600); zudem nur TP4/TP8.
  Sperren-Scan nur über main-Zeilen neu seit 6a143197e (GLM-Commits laufen seit 25.10. in Produktion): drei Treffer, alle abseits von GLM PP5
  (GGUF Q5_K `runtime_dequant_blas.cu:370`, GGUF-MoE `gguf_moe.py:373`, DeepSeek-V4.1 `deepseek_v41/model.py:910`).
  Text: `make_glm_body.py` (Stapel-Sätze, Hashes per Betreff auf den neuen Zweig, Fork-Messarme 0b8b98c4f/c213a336e/31f961300/5fef3ad60 bleiben) → body-glm53.v3.md.

## Stand 28.10. ~08:45 — GLM-PR gesendet (#1286), CI-Funde behoben

- CI nach dem Push 07:10: auf allen Zweigen nur check-layering (= main abec8ba85). ZUSÄTZLICH mypy rot auf #1280/#1282:
  `deepseek_v41/attention_ops.py` (#1277) rief `attend_gathered_keys` noch mit `logits` (unser Fused-Softmax-Commit hat die
  Signatur geändert, und die Funktion nullt unbenutzte Schlüssel nicht mehr). Bestand seit dem Senden am 27.10. (lokal geprüft wurde
  nur `--from-ref`, main-mypy war damals selbst rot). Behoben im verursachenden Commit (Aufrufer nullen selbst, probs [T,H,W] fp16),
  Gleichwertigkeit gegen main-Funktion V100+RTX 99.69–100 % bitgleich (`scratchpad/v41check/check_v41_attend.py`), V4.1-Tests = main.
  Neu gepusht: #1280 32e4372db, #1282 bff037870 (Archiv-Tags `archive/<zweig>-vor-v41fix-2026-10-28`), Texte ergänzt.
- Lehre: vor jedem PR-Push zusätzlich `pre-commit run mypy-3.10 --hook-stage manual --all-files` (`~/.cache/bench-scripts/mypy_all_2026-10-28.sh`);
  lokal meldet es immer 3 Fehler in `flash_v100/runtime.py` (Umgebung, auch auf main, nicht in der CI).
- Derselbe Volllauf fand im GLM-Zweig: `glm5next/nvidia/mtp.py` rief `_try_load_fp8_indexer_wk`/`_try_load_fp8_attn_proj` ohne das neue
  `model`-Argument (Commit „load block scales next to their weights“) → eingefaltet; GLM-Modelltests danach grün.
- GLM-Vollboot aus dem PR-Baum (502d1b546): 1. Versuch scheiterte an fehlendem `_vllm_fa2_C_sm75.abi3.so` (PR-Bau nur 7.0), aus v4b kopiert;
  2. Versuch: bereit 752 s, Decode 38,86 tok/s, nach 8,7k 53,10 tok/s, Langkontext bis 516 196 Token, acht Antworten byte-gleich prod-v4.
- GESENDET #1286 „[Model][SM70][SM75] GLM-5.3-Flash-FP4-Local: compressed expert scales, 8-bit QPN8 projections“ (Kopf 814c88909,
  27 eigene Commits auf #1282), Live = `body-glm53.v3.md`; #674 um #1286 ergänzt (15 offen).
- Client-Etikett „nach 14k“ war falsch (lange Prompts 8 490–8 921 Token): `glm_ab_client.py` korrigiert; Texte #1282, #1272, #1264 berichtigt.
- Für den nächsten Fork-Abgleich mit 1Cat main: V4.1-Aufrufer-Korrektur aus 32e4372db mitnehmen (Fork hat #1277 noch nicht).

## Stand 28.10. ~14:15 — Produktion auf prod-v5 umgestellt

- `review-fixes-2026-10-27` (89ce1e78b) = Produktion `1Cat-vLLM-v5` seit 13:06 nach A/B aller drei Modelle (`ab_v5c_*_2026-10-28a`) und
  Kontrolle (`validate_prod_v5_2026-10-28.log`); Tag `verified-2026-10-28-prod-v5`; `fork-main-v4` per Fast-Forward = 89ce1e78b. STAND.md 28.10. 14:10.
- CI 08:00: alle Zweige nur check-layering (= main). #1280-Lauf 40650151656 hängt im Runner (pre-commit seit 07:27 UTC, kein
  Workflow-Zeitlimit → bis 6 h); #1282 enthält den #1280-Kopf und lief 07:27–07:36 grün bis auf check-layering.
- Weiter: Log-PR (KV-Speicher je Worker 5bbe3fa84 + Decode-Routenzeilen `scope="process"`), PR PP-Senden (c80e83f8a),
  Upstream-Abgleich Fork ← 1Cat main abec8ba85 (V4.1-Aufrufer-Korrektur mitnehmen), Optimierung, Aufräumen, Cache-Neubau.

## Stand 28.10. ~16:40 — Upstream-Abgleich geprüft, Decode-Wachstum auf main geklärt

- Gesendet seit 14:15: #1293 (Logs), #1294 (PP-Senden). `fork-main-v5` 40959fd24 = prod-v5 + main 6aea01fc6 geprüft: keine neuen
  Testfehler, Boot-Checks aller drei Modelle gleich der Produktion (STAND.md 28.10. 16:40).
- Das Decode-Wachstum vom 27.10. (main 115 → 269 ms) gibt es auf main 6aea01fc6 nicht mehr: ohne DSpark 46,0 → 46,8 ms je Schritt
  von 33k bis 303k (`dsv4_decode_growth_M_2026-10-28.log`); prod-v5 gleich eingestellt 34,6 → 38,5 ms (`…_F_…`). Der Vergleich
  vom 27.10. mischte Schritte ohne DSpark (main) mit DSpark-Runden (Fork). Behoben durch die gemergten #1263 und #1222.
- 1Cat-Kommentare 28.10.: #1227 (justxiami: Schalter wirkt nur auf FA2, nicht auf FLASH_ATTN_V100) stimmt mit unserem PR-Text
  überein, keine Antwort nötig. #1277 (justxiami): `ExpertCache.resolve()` bekommt seit #1277 ein Argument zu viel (GGUF-Expertencache
  bootet nicht) — steckt auch in fork-main-v5, betrifft keinen unserer Einträge (kein GGUF über vLLM).
- ZURÜCKGESTELLT (Peuqui 28.10. 16:50: 1Cat-eigene Fehler haben keine Priorität, erst wenn unsere Arbeit fertig ist): kleiner Test-PR für 1Cat (`top_k` im Test-Layer von `test_sm70_fp4_mainflow.py` seit #1277,
  eingefrorene Fixture mit `_DFLASH_NVFP4_TRACE_ARMED`). Aufräumen hat Peuqui 16:44 ausgeführt (151 G frei); Cache-Neubau läuft (`rebuild_prod_v5_2026-10-28.log`).

## Stand 01.11. ~22:00 — nächste PR-Runde (Bestandsaufnahme, nichts gesendet)

- 1Cat main 4d57473db vom 29.10., seitdem kein Merge. Offen 17 PRs, Konflikt: #1279, #1280, #1282, #1286. Force-Push dafür braucht eine neue Freigabe (die vom 28.10. galt jener Runde).
- Neue Commits seit prod-v5, Probe-Merge gegen main:
  - sauber, eigenständig: f1b54cf46 Stack-Fix (größter Hebel, V100 −1 496 MiB), 671babc33 V100-Verify (259k 160 → 86 ms), c9abb8d25 FA2-Split-Heuristik.
  - hängen an offenen PRs: ea2ec07f6 DFlash2-Überlappung (an #1279), 9bc45c663 DSv4-Scorer und 772cbe803 Pass 512 (an #1282; `_TC_TOKENS_PER_PASS` gibt es auf main nicht).
  - offen je nach A/B: Zwischenbucket 262 144 in `compressed_context_buckets` (Faktor 128 in der Liste 1, 2, 8, 32, 64).
- Vor dem Senden: `check-layering` beheben oder als main-Zustand belegen (`moe_runner.py` 37 → 39, `ubatching.py` 0 → 1), mypy-3.10 über alle Dateien, Zahlen aus den Messdateien vom 01.11., Revert-Prüfung, Duplikatsuche.
- Triton: Fork Peuqui/triton 9063e502c ist Stand aller Patches (letzter Kern-Commit 23.10.), Discussion #12517 ohne Kommentar, 1 Stimme. Nichts Neues einzureichen.

## Stand 02.11. ~12:15 — Stapel gepusht, drei neue PRs

- Force-Push (Peuquis Freigabe 02.11.): #1279 7344bc20e (+ ea2ec07f6 gefaltet), #1280 2aa808dab, #1282 af91cddde, #1286 53a6f0973; alle MERGEABLE, Basis main 4d57473db, sauber gegen 6db9c816c. Update-Absätze in `16-1cat-prs-2026-11-02/update-*.final.txt`, live = Entwurf. Archiv-Tags `archive-<zweig>-2026-11-01` im Fork.
- Neu: #1344 Stack-Fix (f1b54cf46), #1345 V100-Verify (671babc33), #1346 FA2-Pin (c9abb8d25); Texte `16-1cat-prs-2026-11-02/body-*.md`.
- Offen: P4 Scorer 9bc45c663 + Pass 512 772cbe803 auf #1282; P5 KDA-Warmup 3f6bdcc1f + MXFP8-LM-Kopf 0181fdc6c auf #1286 (fehlten in der Liste vom 01.11.); Triton-Wheel (Release Peuqui/triton) + PR auf #1282. PLE-Plattenstufe nicht eingereicht (Duplikat bzw. ohne Nutzennachweis auf main), Zweig `pr-ple-disk-tier` bereit. Nicht Produktionspfad, kleine Bugfix-Kandidaten: Sync-PP mit Spec Decode (fc51a3be1, 7096dc040), FLA Shared Memory je Gerät (a6d6d3e82).
