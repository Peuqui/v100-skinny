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

## Probe-Merge-Ergebnis (git merge-file, 3-Wege je Datei, 2026-09-03 abends)

Der Kollisionsgrad ist maschinell gemessen (Basis v1.3.0, ours =
fork_patch, theirs = v1.5.0; benchmarks/rebase-150-mergecheck-2026-09-03
.txt): **36 der 58 upstream-geaenderten Dateien mergen KONFLIKTFREI**
(Upstream aenderte andere Stellen als wir — rein mechanische Arbeit).
**22 Dateien haben echte Konflikte**, stark konzentriert:
worker_mamba_utils (26 Hunks), gpu_model_runner (16), worker_utils (7),
mhc_tilelang (6), utils (5), modelopt (4) — der Rest 1-3 Hunks, und
mehrere davon sind Obsoleszenz-Gewinne statt Arbeit (mhc_triton = unser
HEAD-Backport vs. das Original; speculative = Idempotenz-Fix evtl.
upstream; qwen3_5/gdn-Komplex = deren eigene SM70-Weiterentwicklung).
Gesamtbild: 18 unveraendert + 36 sauber + 22 Konflikt (davon ~6 ernst)
+ 2 geloescht (qwen4_exp durch Upstream-Paket ersetzt) + 5 fork-eigene.

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

## Phase-1-Smoke-Protokoll (2026-09-03 abends, RadixArk-FN auf Stock-1.5.0, 2x2)

Drei Boot-Gates in Serie (Skript scratchpad/smoke-flashnext-150.sh):
1. `--language-model-only` ist PFLICHT (deren SM70-Route wirft sonst
   NotImplementedError fuer den Vision-Tower — wie unser 1.3.0-Port).
2. `VLLM_PLE_CPU_OFFLOAD` unterstuetzt KEIN PP ("Unsupported settings:
   PP=2") — deren PLE-Offload ist TP-only (Referenz TP4). Auf unserer
   Kiste traegt daher die RTX-TP2-Stufe die PLE selbst
   (VLLM_PP_LAYER_PARTITION=20,28 statt 24,24).
3. **UPSTREAM-PP-BUG gefunden:** `hyper_connection_mixer` wird nur auf dem
   letzten PP-Rank instanziert (nvidia/model.py ~489), aber load_weights
   skippt seine Checkpoint-Tensoren auf den anderen Raengen nicht →
   ValueError auf PP0. Stock-1.5.0 hat Qwen4Exp offenbar nie unter PP
   gefahren. Minimal-Fix in der -150-venv deployt (skip_substrs +=
   "hyper_connection_mixer." wenn nicht last_rank) — als Smoke-Fix
   dokumentiert, gehoert in fork_patches_150 UND als Issue/PR an 1Cat
   (Freigabe Peuqui fuer Outward noetig).

Messplan-Erweiterung (Peuqui): Vergleich DREISEITIG — Stock-1.5.0 vs
1.5.0+Patches vs FRISCHER 1.3.0+Patches-Kontrollboot desselben Modells
(alte 32,2-Baseline vom 28.08. ist nach moe_qpn/mHC/QPN8-sm75 vermutlich
ueberholt; FN lief damals auf MARLIN-MoE). Sprachzerfall-Test in DE UND
EN (1Cats Testsprache — erklaert ggf., warum deren Audits nichts sehen).

## Phase-1-ERGEBNIS: Stock-1.5.0 ist auf unserer Kiste NICHT bootfaehig (PP-Sperren)

Drittes und hartes Gate: "N-gram PLE embedding currently requires
pipeline_parallel_size=1" — Stock verbietet PLE unter PP (non-first
ranks erhalten keine input_ids). Zusammen mit Gate 2 (PLE-Offload
TP-only) und Gate 3 (hyper_connection_mixer-Loader-Bug) ist Qwen4Exp
in 1.5.0 strikt auf homogene TP-Setups ausgelegt. FAZIT: unsere
Patches sind fuer heterogene/PP-Kisten NICHT obsolet, sondern
Voraussetzung. MATRIX-KORREKTUR: qwen4_exp.py/spec_decode_qwen4_exp.py
sind NICHT "ersetzt durch Upstream" — beim Rebase gilt: Upstream-Paket
als Basis + unsere PP/PLE-Kaskade-Deltas (Kern von PR dnv2003#7) darauf
portieren. Der dreiseitige Vergleich wird zweiseitig (Stock disqualifiziert
sich fuer PP); naechster Messpunkt: FRISCHER 1.3.0+Patches-Kontrollboot
(RadixArk roh, k=0) als aktuelle Baseline statt der 32,2 vom 28.08.

## Rebase-Nachlese: toter Align-Pfad in gpu_model_runner (2026-09-04, GEFIXT)

Beim Rebase blieb `_get_mamba_state_copy_funcs` als Upstream-Code
stehen, obwohl der Fork die Vertraege darunter geaendert hat:

| | Upstream 1.5.0 | dieser Fork |
|---|---|---|
| `mamba_utils.get_mamba_groups` | `dict[MambaSpec, list[int]]` | `tuple[list[int], MambaSpec]` |
| `validate_mamba_state_copy_funcs` | vorhanden | entfernt, Pruefung wanderte in `_get_copy_funcs_for_group` (pro Gruppe, zur Nutzungszeit) |
| `get_mamba_types` | fehlt | vorhanden, genau fuer diesen Aufruf |

Der stehengebliebene Aufrufer iterierte das Tupel, als waere es eine
Spec-Liste — `AttributeError: 'list' object has no attribute
'mamba_type'`, EngineCore tot beim ersten Request. Dahinter haette
sofort der zweite Defekt gewartet: der Aufruf des im Fork nicht mehr
existierenden Validators.

Warum es niemandem auffiel: Der Zweig ist nur ueber
`mamba_cache_mode == "align"` erreichbar, und den nimmt vLLM nur bei
eingeschaltetem Prefix-Caching. 1.5.0 schaltet Prefix-Caching fuer
Hybrid-Modelle per Default AB (`config/model.py`,
`is_prefix_caching_supported`: Hybride seien "still experimental") —
1.3.0 fuhr es an. Der gesamte Align-Pfad war damit seit dem Rebase
unerreichbar und ungetestet.

FIX: `get_mamba_types(self.kv_cache_config)` statt der Handarbeit auf
dem Tupel; der Validator-Aufruf entfaellt ersatzlos, weil
`_get_copy_funcs_for_group` dieselbe Zusicherung pro Gruppe prueft.

VERIFIZIERT 2026-09-04, 27B NVFP4 TP1 auf RTX 8000, greedy (T=0, fester
Seed), 1425-Token-Prompt: Cache trifft (784 Token), und die Antworten
sind kalt, mit vollem Cache-Treffer und bei Teiltreffer zeichengleich.
Null EngineCore-Fehler.

LEHRE FUER DEN NAECHSTEN REBASE: Geaenderte Helfer-Signaturen in
`mamba_utils` ziehen Aufrufer in `gpu_model_runner` nach sich, die der
3-Wege-Merge NICHT anfasst, weil ihr Text auf beiden Seiten identisch
ist. Pfade hinter Default-off-Schaltern (hier Prefix-Caching) fallen
durch jedes Boot-Gate — sie brauchen einen eigenen Smoke-Lauf mit
eingeschaltetem Schalter.

### Nachtrag: Align-Pfad MIT spekulativem Dekodieren verifiziert

Der erste Korrektheitstest lief ohne Spekulation, liess also den Zweig
`with_postprocess_align = (speculative_config is not None and
is_hybrid)` ungeprueft. Nachgeholt 2026-09-04 mit der von der
Kalibration selbst gebauten Kommandozeile (`VllmSpec.build_cmd`,
k=4, `--speculative-config method=mtp`), 27B NVFP4 TP1, greedy:

| Prompt | Bloecke a 816 | cached | Antwort |
|---|---|---|---|
| 1.345 Tok | 1 | 0 | identisch |
| 6.925 Tok | 8 | 5.712 (= 7 x 816) | identisch |

Kein Absturz, keine Zustandskorruption, null EngineCore-Fehler.
Wiederholter Aufruf 21,1 s -> 4,5 s.

WICHTIG FUER TESTS: Spekulation verschiebt die Blockgeometrie (816
statt 784 Token), und gecacht werden nur ABGESCHLOSSENE Bloecke vor dem
laufenden. Ein Prompt mit nur einem vollen Block liefert deshalb null
Treffer und sieht wie ein kaputter Cache aus — er ist bloss zu kurz.
Messungen brauchen mehrere Bloecke.
