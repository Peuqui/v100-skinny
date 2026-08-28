# Übergabe: 1Cat-vLLM 1.3.0 + Qwen3.8-Flash-Next (Qwen4Exp) Portierung

Stand: 2026-08-27 · Vorgeschichte: [DEEPSEEK-VLLM-HANDOVER.md](DEEPSEEK-VLLM-HANDOVER.md)
(DeepSeek-Schnitt, Begründung für den Schwenk) ·
Referenzkonfiguration und 85-tok/s-Beleg: [MERGE-PROJECT-HANDOVER.md](MERGE-PROJECT-HANDOVER.md)

> ## ✅ ZUERST LESEN — MTP ist GELÖST (28.08. abends)
>
> **Der MTP-Block des Checkpoints war unquantisiert (4,86 GiB BF16) und las
> damit fast so viel wie das gesamte 125B-Modell. Ersetzt durch eine
> NVFP4-Fassung (1,49 GiB) — fertig.**
>
> | | schwer | vorhersagbar |
> |---|---:|---:|
> | k=0 | 32,2 | — |
> | k=4, BF16-Block (vorher) | 14,0 | 19,4 |
> | **k=4, NVFP4-Block (jetzt)** | **49,2** | **67,2** |
>
> Akzeptanz 73,1 %, Länge 3,92, Kohärenz einwandfrei, null Fehler.
> Modell: `/home/mp/models/Qwen3.8-Flash-Next-180B-A4B-NVFP4-MTPQ`
> (Symlinks auf den RadixArk-Bestand + 1,49 GiB aus provsalt, 35 MB echt).
>
> **Pflicht-Betriebsparameter: `VLLM_SM70_E5_CACHE=0`, vor dem Skriptaufruf
> gesetzt** (Modul-Konstante) — sonst crasht `_e5_apply_ints` am QSA-Ring.
>
> Details im letzten Abschnitt („Abend II"), die Diagnose davor
> („Abend" / „Nachmittag"). Verworfene Thesen, nicht erneut verfolgen:
> Graph-Capture, XQA-Verifier, „Drafter produziert Müll", fehlendes
> `--async-scheduling`, SM70-Baseline, Speichermangel.

## Auftrag

Nach dem DeepSeek-Schnitt ist Qwen3.8-Flash-Next das Zielmodell: MTP im
Standardformat, 135 GB statt 176, GDN-Hybrid-Familie. Zwei Schritte wurden
freigegeben: (1) 1Cat-vLLM von 1.2.2 auf 1.3.0 heben, (2) upstream-PR #53896
(Qwen4Exp-Modellsupport) einziehen.

## Schritt 1: Upgrade auf 1Cat-vLLM 1.3.0 — ABGESCHLOSSEN und verifiziert

`.venv-sm70-130` ist eine **parallele** venv; die produktive `.venv-sm70`
(1.2.2) ist unangetastet. Aufgebaut per Klon (42 s) statt Neuinstallation —
alle kritischen Abhängigkeiten sind identisch (torch 2.10.0, tilelang 0.1.10).

**Wheel:** `1cat_vllm-1.3.0-cp312-cp312-linux_x86_64.whl`, SHA256
`2bdb14a9c44f83ee6a766d88ed0d85b11390d6f5d65747e8dbe80a8e2d5d63e0`,
gesichert unter `backups/2026-08-26-pre-1.3.0/`. Byte-identisch mit dem
GitHub-Snapshot (stichprobenartig über vier Dateien geprüft).

### Der Patch-Satz musste rebasiert werden

**Kritischer Punkt:** `fork_patches/` enthält **vollständige Dateien, keine
Diffs**. Ein naives Upgrade hätte alle 1.3.0-Verbesserungen in diesen 29
Dateien überschrieben — bei `fp8.py` etwa 220 neue Zeilen von 1Cat gegen 21
Zeilen von uns. Die Umstellung auf Diff-basiertes Rebasen ist damit
Voraussetzung jedes künftigen Upgrades, keine Kosmetik.

Rebase-Ergebnis (Diff gegen 1.2.2-Original, angewendet auf 1.3.0):

| | Anzahl |
|---|---:|
| Zieldatei in 1.3.0 unverändert → 1:1 übertragbar | 12 |
| Rebase sauber | 8 |
| Rebase mit Kontext-Toleranz | 5 |
| Handarbeit nötig | 2 |

Die rebasierten Patches liegen in `backups/2026-08-26-pre-1.3.0/fork_patches_130/`,
die 1.2.2-Originale und die Rebase-Diffs daneben.

**Falle, die Zeit gekostet hat:** `patch -F3` meldete bei `mhc_tilelang.py`
Erfolg, hatte den fp16-Einschub aber **mitten in eine Funktionssignatur**
gesetzt. Nur der Syntax-Check fand das. **Nach jedem Fuzz-Patch `py_compile`
laufen lassen** — und bei Einschüben lieber strukturell platzieren
(Zielfunktion suchen) als auf Kontext-Matching vertrauen.

**Zweite Falle:** `git apply --check` in `site-packages` meldet Erfolg mit
Exit 0, wendet aber **nichts** an — `.venv-sm70` steht in der `.gitignore`
des Repos, und git überspringt ignorierte Pfade stillschweigend. Ein
Patch-Test dort ist wertlos; in einen Baum außerhalb des Repos kopieren.

**Dritte Falle:** Die Deploy-Liste hat drei Mechanismen, nicht einen:
`deploy` (29 Ersetzungen), `deploy_new` (3 Dateien, die es im Wheel nicht
gibt: `qpn8_blk.py`, `gdn_attn_sm75.py`, `qwen_gdn_linear_attn_sm75.py`) und
einen `cp -r` des `flash_linear_attention`-Baums (19 Dateien). Wer nur
`^deploy ` greppt, übersieht ein Viertel.

**1Cat hat die FP8-Umgehung inzwischen selbst gebaut:** neue Datei
`deepseek_v4/common/ops/fp8_software.py` mit Software-E4M3-Konvertierung,
gegated auf `is_device_capability((7, 0))` — laut Doku „**exactly**", greift
also nur auf V100, nicht auf RTX 8000. Unsere e4b15-Patches in
`dsv4_cache_utils.py` und `dsv4_fused_compress_quant_cache.py` wurden
deshalb fallengelassen (1.3.0-Original deployt). **Offener Punkt:** Für
DeepSeek-KV auf sm75 müsste 1Cats Gate erweitert werden — betrifft nur den
geschnittenen DeepSeek-Port.

### Verifikation

- **Elf Boot-Gates PASS** (XQA-Decode aktiv, QPN8-Census 256/0 ineligible,
  MTP-Tiefe 7, FP16-KV aufgelöst) — heterogenes 2×2, GPU 0,2 (RTX) + 1,4 (V100).
- **Durchsatz** (bench.py, ninfer, max 2048, Seeds 1001/2002/3003):

| Stand | math | code |
|---|---:|---:|
| 1cat-vLLM 1.2.2 (Referenz) | 85,0 ± 1,1 | 56,3 ± 1,1 |
| 1cat-vLLM 1.3.0 | 83,9 ± 0,4 | 56,0 ± 1,2 |
| 1.3.0 + Qwen4Exp-Teilportierung | 83,7 ± 0,6 | 56,4 ± 4,8 |

  **Parität** — die Differenzen liegen innerhalb der Fehlerbalken. Die
  Portierung kostet keine Leistung.

### Was 1.3.0 bringt — und was nicht

Kern des Releases ist ein neuer **D=256-Long-Prefill-Attention-Pfad für V100**
(17,92 → 46,6 TFLOP/s, ~2,6×). Qwen3.8-Flash-Next hat `head_dim = 256`, trifft
die Bedingung also exakt. Aber die Einschränkungen sind hart und stehen im
Design-Dokument `docs/design/sm70_fa2_d256_prefill_pipeline.md`:

- **nur SM70** — die RTX 8000 (sm75) bekommt davon nichts, sie bleibt auf TRITON_ATTN;
- **nur Prefill** — „Decode TPOT is unchanged within 0.04 %";
- **nicht mit FP8-KV** — verlangt FP16-Aktivierungen und -KV.

Für Qwen3.8-Flash-Next ist FP16-KV leistbar (24 KiB/Token), der Pfad greift
also. Für DeepSeek (FP8-KV Pflicht) nicht.

## Schritt 2: Qwen4Exp-Portierung — Verdrahtung steht, Boot noch offen

### Was steht

- **33 neue PR-Dateien deployt**: `vllm/models/qwen4_exp/` (31, mit `nvidia/`-
  und `amd/`-Zweig), `v1/spec_decode/qwen4_exp.py` (Proposer),
  `transformers_utils/configs/qwen4_exp.py`.
- **29 Kern-Dateien tragen PR-Hunks** (18 automatisch angewendet, 12 Stellen
  von Hand portiert).
- **Modell ist registriert** (`registry.py`), Config-Klassen
  (`Qwen4ExpForCausalLMConfig` etc.) samt `MODELS_CONFIG_MAP`-Einträgen sind da.
- **Plattform-Weiche umgebogen**: `qwen4_exp/__init__.py` wählt bei
  Capability < 80 den `amd/`-Triton-Zweig. Verifiziert — auf dieser Maschine
  lädt `vllm.models.qwen4_exp.amd.model`.

### Der Machbarkeits-Check war positiv

Von 55 importierten vLLM-Modulen existieren **52** in 1.3.0; von 123
importierten Symbolen **114** direkt. Vier der fünf echten Lücken liefert die
PR selbst mit. Das einzige fehlende Modul, `cute_dsl.skinny_gemm`, wird nur
vom `nvidia/`-Zweig gebraucht — der `amd/`-Zweig hat dafür einen No-Op.

### Von Hand geschlossene Lücken

| Lücke | Lösung |
|---|---|
| `WeightsMapper` ohne `orig_to_new_stacked` | upstream-Klasse übernommen (rückwärtskompatibel: `_map_name` ist Wrapper um `_map_name_with_shard`); `ShardId`-Alias ergänzt |
| `maybe_fuse_shared_experts` fehlt | per AST aus upstream übernommen (auf CUDA ein No-Op) |
| `Qwen4ExpMTPModelArchConfigConvertor` | Klasse vor `MODEL_ARCH_CONFIG_CONVERTORS` eingefügt |
| Qwen4Exp-Config-Klassen | Block eingefügt, Registry-Einträge ins Dict verschoben |
| `KVCacheSpec.prefix_cacheable` fehlte auf `MambaSpec` | auf der Basisklasse mit Default `True` definiert (= Verhalten vor der Portierung; nur `CircularBufferSpec` opt-out) |
| `VocabParallelEmbedding(quant_method=…)` | Signatur-Parameter + FP8-all-reduce-Zweig (für die FP8-PLE-Tabelle) |
| `SlotMappingMode`, `KVCacheLayout` | existieren in 1Cat nicht; Annotationen zu Strings, Funktionen sind ungenutzt (ihre Aufrufer stecken in offenen Hunks) |

### Die Mamba-Gruppen-Semantik ist vermittelt (27.08., ERLEDIGT)

Der frühere Blocker: 1Cats `get_mamba_groups` gab ein **Tupel**
`(list[int], MambaSpec)` zurück, upstreams gleichnamige Funktion ein **Dict**.
Ein stumpfes Anwenden der PR-Hunks hatte `MambaCopyBuffers.create` und
`_get_mamba_state_copy_funcs` gebrochen, weshalb beide Dateien
zurückgesetzt worden waren.

**Gelöst durch minimal-invasiven Umbau von `v1/worker/mamba_utils.py`** auf die
upstream-Semantik, ohne die Fork-Erweiterungen zu opfern (DDTree-Trace,
`copies_per_req`, `batch_memcpy(max_size=…)`, `preprocess_mamba_all_specdec`,
`stage_*_to_gpu` sind alle erhalten). Die PR-Datei 1:1 zu übernehmen hätte
sie gekillt.

Warum die Verallgemeinerung unvermeidbar war: Qwen4Exp mischt **zwei**
Mamba-Typen — `GDN_ATTN` mit zwei State-Tensoren pro Layer und `SHORT_CONV`
(die PLE-Faltung) mit einem. Die Annahme `num_states = num_layers ×
num_state_types` ist damit falsch; sie ist überall durch ein gezähltes
`num_states` ersetzt (`count_mamba_states`).

Neu bzw. geändert in `v1/worker/mamba_utils.py`:

| Symbol | Rolle |
|---|---|
| `get_mamba_groups` | liefert `dict[MambaSpec, list[int]]` statt Tupel |
| `get_mamba_group_ids` | sortierte, duplikatfreie Gruppen-IDs aus dem Dict |
| `_get_mamba_spec_for_layer` | löst `UniformTypeKVCacheSpecs` je Layer auf |
| `assert_uniform_cache_scheduling` | prüft gemeinsame `block_size` / `num_speculative_blocks` / `mamba_cache_mode` |
| `count_mamba_states` | zählt physische State-Tensoren über alle Gruppen |
| `validate_mamba_state_copy_funcs` | prüft Copy-Funcs gegen `mamba_spec.shapes` |

Im Runner ist `self.model.get_mamba_state_copy_func()` an allen sechs
Aufrufstellen durch `self._get_mamba_state_copy_funcs()` ersetzt — ein
memoisierter Helfer, der über `get_mamba_state_copy_funcs(mamba_types)` das
Dict holt, validiert und in `initialize_kv_cache` zurückgesetzt wird.

### Weitere Verdrahtung, die dazugehörte (27.08.)

- **PLE/N-Gram-Kontext im Runner**: `uses_ngram_embedding`,
  `_prepare_ngram_context`, `_maybe_add_ngram_kwargs`, `ngram_context`-Puffer,
  erweiterte `_preprocess`-Signatur (`num_reqs`, `num_reqs_padded`) und die
  Dummy-Run-Variante für CUDA-Graph-Adressstabilität.
- **`Qwen4ExpMTPProposer`** registriert (Union, `use_qwen4_exp_mtp()`-Zweig)
  und mit Per-Gruppen-Blocktabellen versorgt (`set_per_group_block_table`).
- **`PleShortConvAttentionMetadataBuilder`** in die Spec-Decode-Weiche von
  `_build_attn_group_metadata` aufgenommen — sein `build()` verlangt
  `num_accepted_tokens` und `num_decode_draft_tokens_cpu`.
- **`CircularBufferSpec: CircularBufferManager`** in `spec_manager_map`
  (fehlte; `spec_manager_map[CircularBufferSpec]` hätte `KeyError` geworfen).
- **`prefix_cacheable`-Filter** in `kv_cache_coordinator` (Teilbarkeits-Assert
  und Mindestzahl cacheable Gruppen) und in `resolve_kv_cache_block_sizes`
  (`hashing_sizes`) — der QSA-Ring hat `prefix_cacheable = False`, seine
  Ring-Kapazität darf die Hash-Granularität nicht bestimmen.
- **Triton-Pfad für `mamba_get_block_table_tensor`** verdrahtet; der Kernel
  `mamba_get_block_table_tensor_triton` war eingezogen, wurde aber nie gerufen.

### Drei Fehler im eingezogenen Stand, die beim Aufräumen auffielen

1. **`_record_new_block_ids` existierte nicht.** Der aus der PR übernommene
   `CircularBufferManager` liest das Attribut in `_claim_ring_block`, aber es
   kam aus einem der zurückgewiesenen Hunks — AttributeError beim ersten
   Ring-Block. Jetzt als SSOT in `SingleTypeKVCacheManager.__init__` gesetzt,
   mit 1Cats Kriterium (`type(spec) in (FullAttentionSpec, TQFullAttentionSpec,
   MLAAttentionSpec)`), und die beiden dort duplizierten `type()`-Prüfungen
   nutzen es. Verhalten unverändert, `CircularBufferSpec` fällt korrekt raus.
2. **`import functools` doppelt** in `v1/attention/backends/utils.py`.
3. **`Fp8MoeBackend` doppelt** im Import-Block von `quantization/fp8.py`.

Beide Doppelungen stammen aus zweimal angewendeten Import-Hunks.

### Offene Hunks

Die 12 `.rej`-Dateien sind am 27.08. einzeln geprüft und abgearbeitet
(gesichert in `backups/2026-08-26-qwen4exp-partial/offene-hunks/`). Stand:

| Datei | Befund |
|---|---|
| `v1/core/single_type_kv_cache_manager.py` | erledigt (Manager-Map + `_record_new_block_ids`) |
| `v1/core/kv_cache_coordinator.py` | erledigt (`prefix_cacheable`-Filter) |
| `v1/core/kv_cache_utils.py` | Hash-Teil erledigt; **CSA-Allokation offen** (s. u.) |
| `v1/attention/backends/utils.py` | erledigt (Triton-Pfad verdrahtet) |
| `model_executor/layers/quantization/fp8.py` | erledigt (Doppelimport) |
| `v1/core/sched/scheduler.py` | **gegenstandslos** — 1Cat behandelt `pad_spec_decode` weiter oben und kennt die Zielstelle nicht |
| `config/vllm.py` | **gegenstandslos** — V2-Runner-Liste; `_is_default_v2_model_runner_model` endet auf `not is_moe and not is_quantized`, Qwen4Exp ist beides |
| `model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` | **gegenstandslos** — verifiziert: null Treffer für `_fused_gdn_decode_unsupported_reason` / `FUSED_GDN_STATE_DTYPES` |
| `model_executor/layers/fused_moe/oracle/unquantized.py` | **gegenstandslos** — FlashInfer-TRTLLM braucht sm80+, und der Pfad gilt nur unquantisierten MoE-Gewichten |
| `v1/worker/gpu/block_table.py`, `…/model_states/mamba_hybrid.py`, `…/warmup.py` | **gegenstandslos** — V2-Runner-Pfad |

Auch `SlotMappingMode` bleibt gegenstandslos: 1Cat kennt den Typ nicht, der
`_get_slot_mapping_mode`-Hunk und der `may_reinitialize_input_batch`-Hunk
haben in dieser Codebasis keine Zielstelle.

**Warum der V2-Pfad gegenstandslos ist:** `VllmConfig._is_default_v2_model_runner_model`
endet mit `return not model_config.is_moe and not model_config.is_quantized`.
Qwen3.8-Flash-Next ist MoE **und** quantisiert — es läuft garantiert über den
V1-Runner. Alle Rejects unter `v1/worker/gpu/` sind damit irrelevant.

**Warum die GDN-Hunks gegenstandslos sind:** Sie ändern
`_fused_gdn_decode_unsupported_reason` und den zugehörigen Kernel-Aufruf.
1Cat hat diesen Pfad gar nicht (null Treffer für
`_fused_gdn_decode_unsupported_reason`, `fused_gdn_decode_post_conv_mtp`,
`FUSED_GDN_STATE_DTYPES`), und der Ersatzpfad `RMSNormGated` akzeptiert
sigmoid bereits (`assert activation in ["silu", "sigmoid", "swish"]`).
Damit ist auch die `csrc/`-Änderung der PR für uns gegenstandslos — **kein
vLLM-Source-Build nötig**, was gut ist, da wir nur das Wheel haben.

### Achtung: nur ein Zweig darf laden

`nvidia/` und `amd/` registrieren denselben Custom-Op
(`vllm::qwen4_exp_grouped_gemma_rmsnorm`). Beide zu importieren wirft
`Tried to register an operator …`. Die Weiche in `__init__.py` sorgt dafür,
dass nur einer geladen wird — beim Testen nicht beide anfassen.

### Der verbliebene Blocker: CSA-Allokation (QSA)

`text_config` trägt `indexer_budget/compress_ratio/head_dim/kv_heads/n_heads`
— die 12 Full-Attention-Layer laufen über **Qwen Sparse Attention**. Es gibt
dafür keinen Schalter: `QSARawKeyCache.get_kv_cache_spec` liefert
bedingungslos einen `CircularBufferSpec`, `QSACompressedKeyCache` einen
`MLAAttentionSpec`. Damit erkennt `_classify_csa_linear_specs` die
CSA+linear-Konstellation **garantiert**, und
`get_kv_cache_groups` verzweigt in `_get_kv_cache_groups_csa_linear`
(bereits verdrahtet, Zeile ~2068).

Die zugehörige Allokation `_get_kv_cache_config_csa_linear` ist eingezogen,
aber in 1Cat **nicht lauffähig** — und sie ist auch nicht angeschlossen
(`get_kv_cache_config_from_groups` ruft sie nicht). Sie verlangt drei Dinge,
die 1Cat nicht hat:

- `compute_layout_strides` — existiert nicht,
- `KVCacheLayout` mit `is_block_compact` / `is_block_outermost` — existiert nicht,
- `KVCacheTensor(layers=…, layer_stride=…, block_stride=…, offset=…)` —
  **1Cats `KVCacheTensor` hat nur `size` und `shared_by`.**

Der Unterschied ist grundsätzlich. 1Cat legt in `_allocate_kv_cache_tensors`
einen `torch.zeros`-Puffer je `KVCacheTensor` an und gibt ihn **jedem** Layer
aus `shared_by` **ab Byte 0**. CSA-linear braucht das Gegenteil: einen Puffer,
in dem die Owner an **verschiedenen Offsets** aliasen —

    bytes_per_block = len(main_kv) * (main_kv_page_size + compressed_page_size)
    main_kv[i]           -> offset  i * main_kv_page_size
    compressed[i]        -> offset  compressed_offset + i * compressed_page_size
    compressor_state[i]  -> offset  compressed_offset + i * compressed_page_size
    mamba-Layer[i]       -> offset  i * main_kv_page_size      (aliast main_kv)

Das ist keine Hunk-Frage mehr, sondern eine Erweiterung von 1Cats
KV-Cache-Allokationsmodell um Byte-Offsets — samt dem Reshape-Schritt
`_reshape_kv_cache_tensors`, der die Blockstruktur baut und heute von einem
Puffer ab 0 ausgeht. **Das ist der nächste und vermutlich letzte große
Brocken vor dem ersten Boot-Versuch.**

## Boot-Versuche 27.08. — wie weit es kommt

Drei Läufe mit `scripts/serve-qwen38-flash-next.sh` (neu; **kein** Gate-Skript,
es startet und meldet, wo der Server stirbt). TP=2/PP=2, `k=0`,
`VLLM_PP_LAYER_PARTITION=18,30`, MML 4096, GPUs 0,2 (RTX) + 1,4 (V100).

Der Boot kommt jedes Mal bis in den **Modellaufbau** — Config, Quantisierung
(`modelopt_fp4`), Registry, Plattform-Weiche und Worker-Start tragen. Er stirbt
jeweils an einer konkreten, benannten Stelle:

| # | Fehler | Status |
|---|---|---|
| 1 | `'Qwen4ExpForConditionalGeneration' object has no attribute '_init_video_pruning'` | **gelöst** |
| 2 | `NotImplementedError: Qwen4Exp QSA currently requires BF16` (`amd/qsa.py`) | **geöffnet** |
| 3 | dieselbe Meldung aus `amd/indexer_qsa.py:95` | offen |

**Zu 1:** Der Aufruf steht in der PR-Modelldatei und erwartet eine Methode aus
einer upstream-Basisklasse, die 1Cat nicht hat. Er liegt im `else`-Zweig von
`language_model_only`. Mit **`--language-model-only`** entfällt er — das
Vision-Tower-Gewicht (1,1 GB) wird nicht geladen und alle Modalitätslimits
gehen auf 0. Für Text-Betrieb ist das die vorgesehene Konfiguration, kein
Workaround; das Flag steht fest im Boot-Skript.

**Zu 2:** Acht Stellen in `amd/qsa.py`, `common/qsa_cache.py` und
`amd/ops/qsa.py` banden QSA an BF16. Der Rechenkern rechtfertigt das nicht:
`_qsa_mqa_paged_kernel` ist Triton und konvertiert **jeden** Load sofort mit
`.to(tl.float32)`, alle Akkumulatoren sind `tl.float32`. Es gibt keine
BF16-spezifische Numerik. Die Gates sind deshalb auf "eine gemeinsame
Halbpräzision" geöffnet (fp16 **oder** bf16, aber konsistent), statt fp16
einfach durchzuwinken. Verifiziert: der Boot läuft danach eine Datei weiter.

### Warum BF16 auf dieser Hardware keine Option ist

`torch.cuda.is_bf16_supported()` meldet auf der V100 **True** — das ist
irreführend, es gibt keine BF16-Tensor-Cores auf Volta oder Turing, torch
rechnet über Konvertierung. Gemessen (4096³ Matmul, 20 Durchläufe, V100):

| dtype | TFLOP/s |
|---|---:|
| float16 | 38,4 |
| bfloat16 | 9,2 |

**Faktor 4,2.** `--dtype bfloat16` würde das Modell also auf ein Viertel der
Rechenleistung drosseln. Die fp16-Öffnung ist damit nicht eine von zwei
Möglichkeiten, sondern der einzige gangbare Weg.

### Zu 3: die verbliebenen BF16-Bindungen — und warum sie eine Entscheidung sind

Der Indexer prüft dasselbe (`indexer_qsa.py:95`) und legt zwei Puffer fest in
BF16 an (Zeilen 130, 139). Dazu kommen **vier `params_dtype=torch.bfloat16`**,
also echte **Gewichte** in BF16:

| Datei | Zeile | Betrifft |
|---|---:|---|
| `amd/model.py` | 257 | im Decoder-Layer |
| `amd/model.py` | 428 | im Modell |
| `amd/mtp.py` | 220 | MTP-Block |
| `common/hyperconnection.py` | 48 | Hyper-Connections (5,2 GB Gewichte) |

Diese vier sind qualitativ etwas anderes als die QSA-Gates. Dort war BF16
Konvention über einem fp32-Kern; hier ist es die gewählte Speicherpräzision
für Gewichte, vermutlich wegen des größeren Exponentenbereichs (die
Hyper-Connections summieren viele Streams). Sie auf fp16 umzustellen ist
möglich, aber jede Stelle ist eine eigene numerische Entscheidung mit
Verifikationsbedarf — und sie stehen zu lassen heißt, genau diese Layer in
den 9-TFLOP/s-Pfad zu zwingen. **Das ist zu entscheiden, bevor weiter
geöffnet wird.**

## Modell-Bestand

`RadixArk/Qwen3.8-Flash-Next-NVFP4` liegt vollständig im HF-Cache
(135,2 GB, 418 Dateien, keine `.incomplete`). Aufschlüsselung:

| Bauteil | Größe | Anteil |
|---|---:|---:|
| MoE-Experten (geroutet, NVFP4) | 68,0 GB | 50,3 % |
| **PLE** (N-Gram-Embedding) | 51,2 GB | 37,9 % |
| Hyper-Connections | 5,2 GB | 3,9 % |
| Linear-Attention (GDN) | 4,3 GB | 3,2 % |
| Shared Experts | 2,6 GB | 2,0 % |
| Full-Attention | 1,5 GB | 1,1 % |
| Vision-Tower | 1,1 GB | 0,8 % |
| MTP-Block | 0,4 GB | 0,3 % |

**PLE ist der bestimmende Sonderfall.** Es ist eine Hash-basierte
N-Gram-Tabelle: 8 Köpfe × 20 Mio Einträge × 320 Byte fp8 = exakt 51,2 GB,
im Checkpoint in 128 Shards. Sie hängt **vollständig an Layer 1**
(52,73 GB); alle anderen Layer wiegen 1,7–1,9 GB.

Konsequenzen:
- **Layer 1 passt auf keine einzelne Karte** (max. 48 GB) ⇒ TP=1/PP=5 wie bei
  DeepSeek ist unmöglich. Es braucht **TP ≥ 2** auf der Stufe, die Layer 1 trägt.
  Die Tabelle ist eine `VocabParallelEmbedding` und wird beim Laden auf den
  TP-Vokabularbereich geschnitten (`compute_ple_shard_overlap`) — bei TP=2
  also 26,4 GB pro Karte.
- **Die 128 Shards sind Dateiformat, keine Laufzeitstruktur** — zur Laufzeit
  ein Tensor pro TP-Rang. Freie Verteilung auf Karten gibt es nicht, und sie
  brächte nichts: die Lookup-IDs sind Hashes (`remainder(mixed, sizes) + offsets`),
  gleichverteilt über die ganze Tabelle, ohne heiße Bereiche.
- **PLE kostet fast keine Rechenzeit**: 2,56 KB pro Token gegen 14,93 GB fürs
  ganze Modell — Faktor 5,8 Millionen. Es ist VRAM-Ballast, kein Rechenposten.
  Es gehört dorthin, wo Platz ist (RTX 8000, 48 GB), nicht auf die schnelle Karte.
- **CPU-Offload scheidet aus**: `VLLM_PLE_CPU_OFFLOAD` (PR #53899) hält die
  Tabelle im pinned Host-RAM — der Mini hat 30 GB. Aux-GPU-Offload ist nur
  Issue #53908, kein Code.

## Empfohlene Topologie

**TP=2, PP=2** über vier Karten; die fünfte V100 bleibt frei (Vigilantia/TTS):

- Stufe 0 = beide RTX 8000 (96 GB): Layer 0–17 **inkl. PLE**
- Stufe 1 = beide V100 (64 GB): Layer 18–47

Rechnung (Gewichte, GMU 0,90): Stufe 0 83,6 GB / 2,8 GB frei, Stufe 1
51,6 GB / 6,0 GB frei. Das Fenster ist schmal — nur Split 15…19 passt
überhaupt. Der Split ist fürs Tempo fast belanglos (unter 2 % zwischen
16 und 20), also **nach VRAM-Komfort wählen**, Startwert 17 oder 18.

**PP > 1 ist erlaubt, entgegen dem PR-Wortlaut.** Upstream verbietet in
`GPUModelRunner.__init__` jedes `pipeline_parallel_size > 1` für PLE-Modelle,
weil `ngram_context` und `query_start_loc` nur auf dem ersten Rang entstehen
und nicht-erste Ränge gar keine `input_ids` bekommen. Der Checkpoint hat aber
`ple_layer_ids: [2]` — **genau einen** PLE-Layer, und nur Layer mit
`self.ple is not None` lesen diese drei Eingaben (alle anderen ignorieren die
kwargs). Der Guard steht deshalb hier auf der tatsächlichen Bedingung: **alle
PLE-Layer müssen auf PP-Rang 0 liegen**, geprüft über `get_pp_indices`, das
`VLLM_PP_LAYER_PARTITION` respektiert. Bei Split 17/18 ist das erfüllt. Die
Fehlermeldung nennt den zulässigen Layer-Bereich und den Ausweg. Ohne diese
Präzisierung wäre die unten empfohlene Topologie nicht bootbar.

**Kontext ist kein Engpass:** Nur 12 der 48 Layer haben KV-Cache (die
36 GDN-Layer halten einen konstanten State), dazu `num_key_value_heads = 2`.
Macht 24 KiB/Token in bf16, 12 KiB in fp8 — 24 GB KV tragen rund eine
Million Token.

**Warum PLE auf die RTX gehört:** Beim Decode zählt Bandbreite, nicht
Rechenleistung. V100 (HBM2) liest mit 900 GB/s, RTX 8000 (GDDR6) mit
672 GB/s. Die V100 sind also die *schnelleren* Karten und sollen die echten
Rechen-Layer tragen; die RTX haben den VRAM für den PLE-Ballast, der ohnehin
nur 2,5 KB pro Token anfasst.

**Der eigentliche Kostenfaktor ist TP:** Rund 96 All-Reduce-Aufrufe pro Token,
und auf dieser Kiste ist `NCCL_P2P_DISABLE=1` Pflicht (Ryzen-Root-Port-P2P
defekt), sie laufen also über den Host. Bei 30–50 µs Latenz sind das 3–5 ms
gegen 9,3 ms reine Bandbreitenzeit. Das ist der Preis dafür, dass PLE sonst
nicht unterzubringen ist — und die lohnendste Optimierungsstelle, sobald das
Modell läuft.

## Nächste Schritte

0. **Entscheiden: die vier `params_dtype=torch.bfloat16` auf fp16 ziehen
   oder stehen lassen?** (s. Boot-Versuche). Stehen lassen kostet Tempo
   (Faktor 4,2 auf den betroffenen Layern), Umstellen kostet
   Verifikationsaufwand. Diese Entscheidung geht allem Weiteren voran, weil
   sie bestimmt, wie die restlichen BF16-Stellen behandelt werden.
1. Danach die Boot-Kette weitertreiben: `indexer_qsa.py` und was dahinter
   kommt. Erwartet werden mindestens noch `is_flash_attn_varlen_func_available()`
   (der QSA-Backend-`__init__` verlangt FlashAttention — auf dieser Kiste
   fraglich, der Fork hat sein eigenes `FLASH_ATTN_V100`) und die
   CSA-Allokation.
2. **CSA-Allokation portieren** (bekannter Blocker, s. o.): `KVCacheTensor` um
   einen Byte-Offset erweitern, `_allocate_kv_cache_tensors` auf gemeinsame
   Puffer mit Offset-Slices umbauen, `_reshape_kv_cache_tensors` entsprechend,
   und `_get_kv_cache_config_csa_linear` auf dieses Modell umschreiben und in
   `get_kv_cache_config_from_groups` anschließen. Der Boot erreicht diese
   Stelle noch nicht — sie liegt hinter den BF16-Fragen.
3. Danach die llama.cpp-Latte: upstream kennt `qwen4_exp` noch nicht
   (`llama-arch.h` hat QWEN3NEXT/QWEN35/QWEN35MOE/DFLASH), unsloths GGUFs
   sind also vorerst nicht ladbar. Für den Tempo-Vergleich neu prüfen.

## Betriebsnotizen

- **Regressionstest ist Pflicht nach jedem Eingriff.** Der Import-Test hätte
  keinen der drei Laufzeitfehler gefunden — `quant_method`, `prefix_cacheable`
  und die Mamba-Semantik schlagen erst beim Modellaufbau bzw. im ersten
  Forward zu. Kommando siehe unten; elf Gates müssen grün sein.
- **Das 0.6B-Testvehikel taugt hier nicht als Vorfilter**: dicht, FP8, ohne
  MTP — es durchläuft die MoE-, Mamba- und Spekulationspfade gar nicht, in
  denen alle bisherigen Fehler saßen.
- **`pkill -f` mit einem Muster, das auf die eigene Kommandozeile passt,
  killt die eigene Shell.** (Steht schon im DeepSeek-Handover; ist wieder
  passiert.) Nur über explizite PIDs aufräumen.

### Repro Regressionstest

```bash
cd /home/mp/Projekte/v100-skinny
SNAP=$(ls -d /home/mp/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/*/)
TP=2 PP=2 PORT=8025 K=7 ATTN_BACKEND=AUTO \
ENV_PREFIX=/home/mp/Projekte/v100-skinny/.venv-sm70-130 \
VLLM_QWEN35_MTP_SHARE_IO_WEIGHTS=0 DISABLE_CAR=1 NCCL_P2P_DISABLE=1 ASYNC_SCHED=1 \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2,1,4 \
CUDA_HOME=$PWD/.cuda-nvcc-deb/usr/local/cuda-12.8 \
LOG=$PWD/serve-130-regress.log bash scripts/serve-qwen38-mini.sh "$SNAP"
```

### Regressionsstand 27.08.

Nach **jedem** der drei Pakete (Mamba-Dict, PLE/Proposer, KV-Cache) wurde der
Regressionstest gefahren: elf Gates grün, Durchsatz auf Parität, τ und
Ausgabelänge Zeichen für Zeichen identisch.

**Achtung, die Zahlen der 26.08.-Tabelle sind mit diesen nicht vergleichbar:**
das dort genannte `bench.py` existiert im Repo nicht. Gemessen wird hier mit
`benchmarks/v11_suite.py`, Zellen `math,code`. Deshalb wurde die Referenz am
27.08. neu erhoben — derselbe Harness, derselbe Boot-Befehl:

| Stand | math | code | τ (math/code) |
|---|---:|---:|---|
| Baseline (Teilportierung, 26.08.-Stand) | 90,0 | 92,2 | 3,75 / 3,93 |
| + Mamba-Dict | 89,7 | 91,8 | 3,75 / 3,93 |
| + PLE/Proposer | 89,7 | 92,3 | 3,75 / 3,93 |
| + KV-Cache-Verdrahtung | 89,7 | 92,2 | 3,75 / 3,93 |

Kommando:

```bash
PROBE_BASE=http://127.0.0.1:8025 PROBE_MODEL=qwen3.8-27b PROBE_CELLS=math,code \
ARM_TAG=<name> AC_OUTDIR=$PWD/ac_suite \
./.venv-sm70-130/bin/python benchmarks/v11_suite.py
```

Ergebnisse liegen als JSON in `ac_suite/`. Der Modellname am Server ist
`qwen3.8-27b`, **nicht** der Snapshot-Pfad — `PROBE_MODEL` mit dem Pfad
zu füttern gibt 404.

### Sicherungen

- `backups/2026-08-26-pre-1.3.0/` — fork_patches (1.2.2), 1.2.2-Originale,
  Rebase-Diffs, das 1.3.0-Wheel mit SHA256, **`fork_patches_130/`** (der
  rebasierte Satz, 32 Dateien)
- `backups/2026-08-26-qwen4exp-partial/` — `geaendert/` (29 Dateien mit
  PR-Hunks), `neu/` (37 Dateien), `offene-hunks/` (die verbliebenen `.rej`),
  `pr53896.diff` (der PR-Diff selbst — lag vorher nur im flüchtigen
  Session-Scratchpad)
- `backups/2026-08-27-qwen4exp-mamba-dict/` — `geaendert/` (die am 27.08.
  angefassten Dateien) und `diffs/` (Unified Diffs gegen den 26.08.-Stand)

### Boot-Repro Qwen3.8-Flash-Next

```bash
cd /home/mp/Projekte/v100-skinny
SNAP=$(ls -d /home/mp/.cache/huggingface/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/*/)
CUDA_HOME=$PWD/.cuda-nvcc-deb/usr/local/cuda-12.8 \
bash scripts/serve-qwen38-flash-next.sh "$SNAP"
# Log: serve-flash-next.log ; PID: .flash-next.pid
```

---

# Sitzung 2026-08-27: Qwen3.8-Flash-Next LÄUFT (k=0, kohärent)

## Ergebnis in einem Satz

`RadixArk/Qwen3.8-Flash-Next-NVFP4` bootet unter 1Cat-vLLM 1.3.0 auf dem
heterogenen 2×2-Gitter (2× RTX 8000 + 2× V100) und generiert kohärent —
Kohärenztest **8/8**, **28–29 tok/s** im Decode.

**MTP (k=4) ist ebenfalls kohärent (8/8), aber nur mit `--enforce-eager`.**
Mit CUDA-Graphen liefert der erste Spekulationsschritt NaN; die Ursache ist
damit das Graph-Capture, nicht die Numerik. Details unten.

## Repro

```bash
cd /home/mp/Projekte/v100-skinny
SNAP=$(ls -d /home/mp/.cache/huggingface/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/*/)
CUDA_HOME=$PWD/.cuda-nvcc-deb/usr/local/cuda-12.8 \
bash scripts/serve-qwen38-flash-next.sh "$SNAP"
# TP=2 PP=2 k=0 GMU=0.90 MML=4096 partition=18,30 block=16, Port 8026
./.venv-sm70-130/bin/python scripts/deepseek_coherence.py \
  --url http://127.0.0.1:8026 --model qwen3.8-flash-next \
  --label k0 --out results-coherence-flash-next.json --max-tokens 64
```

Kohärenz 8/8: `Paris` · `1591` (37×43) · `10` · Folgemuster korrekt ·
`Mercury` · sinnvolle Prosa · `s[::-1]` · die Achterliste exakt und in
Reihenfolge (das geht nur mit funktionierender Attention über den KV-Cache).

**Boot-Skript-Neuerungen:** `EXTRA_ARGS`-Durchreicher; `--block-size` wird aus
der QSA-Ringkapazität berechnet (s. u.); `--language-model-only` bleibt fest.

## Die BF16-Frage war keine Abwägung

Die Übergabe stellte die vier `params_dtype=torch.bfloat16` als Wahl zwischen
Tempo und Verifikationsaufwand dar. Das trifft nicht zu: die
Hyper-Connections rechnen `F.linear(hyper_input_normed, weight)`, und
`hyper_input_normed` trägt die **Aktivierungs**-dtype. Unter `--dtype float16`
ist das fp16 gegen ein bf16-Gewicht:

```
RuntimeError: expected m1 and m2 to have the same dtype, but got:
c10::Half != c10::BFloat16
```

Stehenlassen kostet nicht Tempo, es läuft nicht durch den ersten Forward.
Alle vier Stellen ziehen jetzt `model_config.dtype` (nicht hart fp16 —
hardware-agnostisch). Ebenso die drei BF16-Bindungen in `amd/indexer_qsa.py`.

**Korrektur der Bestandstabelle:** die Hyper-Connections wiegen **1,32 GB**,
nicht 5,2 GB (398 Tensoren, `hc_count=4`, `hc_lowrank=320`).

## Der eigentliche Blocker stand nicht in der Übergabe: die PLE-Tabelle

Der Boot starb an **zwei** Stellen; die Übergabe nannte nur die V100-Seite.
Die RTX-Seite starb vorher an `torch.OutOfMemoryError: Tried to allocate
47.69 GiB`. Ursache: die PLE-Tabelle (128 Shards × 2.500.012 × 160,
**F8_E4M3**, 51,2 GB) wurde in Halbpräzision materialisiert — 102,4 GB, bei
TP=2 also 47,69 GiB pro Rang. Genau der Betrag im Log.

Der **nvidia**-Zweig des PR hat dafür `Qwen4ExpPLEFp8EmbeddingMethod`, der
**amd**-Zweig nicht — und selbst das nvidia-Gate greift hier nicht: es
verlangt `isinstance(quant_config, Fp8Config)`, unser Checkpoint fährt
`modelopt_fp4` und listet `*.ple.*` in `ignore`. Upstream fällt das nicht auf
(auf 8 H-Karten sind 102 GB bei TP=8 rund 13 GB pro Karte).

**Gelöst über die Deklaration des Checkpoints selbst:**
`text_config.ple_embedding_dtype = "float8_e4m3fn"`. Weder der PR noch 1Cat
lesen dieses Feld; es ist aber das einzige verlässliche Signal, weil die
PLE-Zeilen von keiner Quant-Config beschrieben werden. Neu in
`amd/ple_layer.py`: die FP8-Embedding-Methode (aus dem nvidia-Zweig
übernommen), `_get_ple_embedding_quant_method(config)` mit diesem Gate,
`params_dtype`/`quant_method` durchgereicht, und der Dequant
(`_dequantize_embeddings`, eine `weight_scale` pro Tabelle) **im Custom-Op** —
dort, wo FP8 Inductor nie erreicht (dessen Triton kennt auf sm70/sm75 kein
`fp8e4nv`). Vorab auf beiden Architekturen verifiziert: FP8-Lookup, int8-View
für den All-Reduce und Dequant rechnen in Eager exakt.

## Wiederkehrendes Muster: upstream-Vokabular gegen Fork-Vokabular

Ein großer Teil der Arbeit war Übersetzung, kein Neubau:

| PR/upstream | 1Cat 1.3.0 |
|---|---|
| `MLAAttentionSpec.tokens_per_state` | `compress_ratio` |
| `spec.unpadded_page_size_bytes` | `real_page_size_bytes` |
| `cache_config.prefix_match_unit` | `hash_block_size` |
| `KVCacheTensor(layers=, layer_stride=, block_stride=, offset=)` | `KVCacheTensor(size=, shared_by=)` |
| `AttentionLayerBase.bind_kv_cache()` | direkte Zuweisung `layer.kv_cache = …` |
| `triton_mrope(..., is_neox_style)` (9 Args) | 8 Args, NeoX fest verdrahtet |
| `async_tensor_h2d(tensor, device=…)` | `(list, dtype, device)` |
| upstream-4D-KV-Cache `[blk, bs, kvh, 2*hd]` | 5D `[blk, 2, bs, kvh, hd]` |

Jede Übersetzung ist im Code kommentiert. Bei `triton_mrope` steht die
NeoX-Annahme jetzt als lauter `NotImplementedError` da, statt unterstellt zu
werden (`get_rope` wird ohne `is_neox_style` gerufen, Default `True`).

## Die CSA-Allokation war kleiner als befürchtet

Die Übergabe erwartete „eine Erweiterung von 1Cats KV-Cache-Allokationsmodell
um Byte-Offsets". Das war nicht nötig. Upstream drückt Aliasing als *gleicher
Offset in einem großen Puffer* aus; 1Cat drückt dasselbe als `shared_by` auf
einem gemeinsamen `KVCacheTensor` aus (`_allocate_kv_cache_tensors` gibt den
Puffer jedem Layer ab Byte 0). Die CSA-Anordnung ist damit eine reine
Übersetzung — gleiche Partition, identische Gesamtgröße:

```
main_kv[i] + mamba[*][i]   |   compressed[i] + compressor_state[i]
```

Jede Spalte ein Tensor. `_get_kv_cache_config_csa_linear` ist entsprechend neu
geschrieben (ohne `compute_layout_strides`/`KVCacheLayout`) und in
`get_kv_cache_config_from_groups` als erster Zweig angeschlossen; die
Speicherabschätzung `_max_memory_usage_bytes_from_groups` hat den passenden
Zweig bekommen. Nur der Allokator liest das Layout — kein Kernel hängt an den
Byte-Offsets, deshalb ist die Übersetzung zulässig.

## Zwei Fehler in 1Cat, die erst diese Kombination sichtbar macht

1. **Komprimierter Cache × Kernel-Block-Teilung.** `_reshape_kv_cache_tensors`
   nahm `shape_block_size = storage_block_size` (des Spec-Blocks), teilte die
   Blöcke aber zugleich in Kernel-Blöcke. Bei `block_size=1568`,
   `storage=392`, `kernel_bs=32` ergab das `[392, 392, 1, 128]` für einen
   Tensor mit 401.408 Elementen. Richtig ist die pro **Kernel**-Block
   gespeicherte Zeilenzahl: `kernel_block_size // compress_ratio` = 8.
   Wo Kernel- und Spec-Block gleich sind (DeepSeek), liefert die neue Formel
   exakt den alten Wert — der Fehler war bisher unsichtbar.
2. **`SlidingWindowMLASpec.__post_init__` rief `super()` nicht auf**, weshalb
   dort `head_size_v` bis heute `None` blieb. Behoben im Zuge des
   Hochziehens von `head_size_v` auf `AttentionSpec` (Details unten).

Dazu aufgeräumt: der doppelte FP8-Zweig in `VocabParallelEmbedding.forward`
(der zweite unerreichbar) und die zweimal definierte
`UniformTypeKVCacheSpecs.get_page_sizes` — beide vom selben Muster wie die in
der Vorsitzung gefundenen Doppel-Hunks.

## Was sonst noch neu verdrahtet wurde

- **`head_size_v` von `FullAttentionSpec` auf `AttentionSpec` hochgezogen**,
  damit `CircularBufferSpec(head_size_v=0)` es erbt; `real_page_size_bytes`
  zählt jetzt `head_size + head_size_v` statt `2 * head_size`. Für alle
  bestehenden Specs identisch (nachgemessen), für den Nur-Keys-Ring korrekt.
  Die dadurch doppelten Overrides in `FullAttentionSpec`/`SlidingWindowSpec`
  sind entfernt — eine Implementierung statt drei.
- **`WeightsMapper`-Shard-IDs werden endlich konsumiert.** Der Mapper hängt
  `data.shard_id` an, `AutoWeightsLoader._load_param` reichte sie nicht an den
  Weight-Loader weiter — gestapelte Parameter bekamen so einen einzelnen Shard
  als ganzen Tensor. Dazu `_STACKED_WEIGHTS_MAPPER` (qkv, in_proj_qkvz,
  in_proj_ba, gate_up, shared_expert.gate_up) in Modell **und** MTP; gegen die
  Tabelle aus `tests/models/qwen4_exp/test_weight_loading.py` des PR geprüft,
  20/20.
- **`expert_mapping` am FusedMoE** (`fused_moe_make_expert_params_mapping`) —
  1Cats `Qwen3NextSparseMoeBlock` setzt es nicht, `AutoWeightsLoader` braucht es.
- **`PPMissingLayer` statt `None`** für den `hyper_connection_mixer` auf
  Nicht-Letzt-Rängen (`_load_module` überspringt ihn dann sauber).
- **`short_conv_state_shape(..., num_spec=0)`** — die Nachbarn in derselben
  Datei haben den Parameter längst; nur diese eine nicht.
- **`is_uniform_type`-Zweig für `CircularBufferSpec`** (dessen eigener Hook
  `is_uniform_with_collection` war da, die Dispatch-Seite fehlte).
- **`CircularBufferManager`-Signaturen** an die Basisklasse angeglichen (alle
  vier Rümpfe sind No-Ops; `find_longest_cache_hit` gab sogar ein 2-Tupel
  statt eines Tupels zurück).
- **`KVBlockZeroer` auf mehrere Seitengrößen verallgemeinert** — ein
  Kernel-Start je Seitengrößen-Klasse statt eines Uniformitäts-Asserts. Das
  Nullen ist korrektheitsrelevant (frische GDN-Zustände müssen null sein),
  konnte also nicht abgeschaltet werden. Bei einheitlicher Seitengröße
  verhaltensgleich.
- **`get_top_tokens` / `get_topk_tokens_and_logits`** am inneren Modell und am
  MTP-Draft (der Greedy-Fastpath des Forks ruft sie).

## Zwei Turing-Grenzen, beide als Fähigkeits- statt Plattformfrage gelöst

1. **GDN:** die FlashQLA-SM70-Kernel wollen 86.016 B Shared Memory, Turing
   kann 65.536 B. `Qwen4ExpDecoderLayer` baut auf sm75 jetzt die
   upstream-GDN-Schicht (`qwen_gdn_linear_attn_sm75`) — dieselbe Weiche, die
   `qwen3_5.py` schon hat. Deren `forward` gibt zurück; der Fork-Kernel
   schreibt in einen Puffer, dafür gibt es neu
   `QwenGatedDeltaNetAttentionUpstreamCall` (Spiegelbild des vorhandenen
   `…ForkCall`).
2. **QSA-Sparse-Attention:** `_qsa_sparse_paged_gqa_splitk_kernel` brauchte
   81.920 B. Die Kachel hält K und V über die volle Kopfdimension plus den
   fp32-Akkumulator:
   `2·BLOCK_N·HEAD_DIM·itemsize + BLOCK_M·HEAD_DIM·4` — bei BLOCK_N 64 und
   HEAD_DIM **256** exakt 81.920 B. Der Code löste das für AMDs 64-KiB-LDS
   bereits, fragte aber `current_platform.is_rocm()`. Jetzt fragt er das
   Gerätebudget (`shared_memory_per_block_optin`) und halbiert die Kachel, bis
   sie passt: RTX 8000 → BLOCK_N 32, V100 (96 KiB) → unverändert 64.
   Gegenprobe mit identischen Eingaben auf beiden Karten:
   max. absolute Abweichung **2,4e-4** (fp16-Rauschen).
   Repro: `scratchpad/qsa_probe.py` (Vorlage im Sitzungsverlauf).

**Merksatz, der sich erneut bewährt hat:** die Bedingung fragte nach der
Plattform, gemeint war die Fähigkeit.

## Eine Falle, die viel Zeit gekostet hat

Wenn ein Worker in `execute_model` eine Ausnahme wirft, **fängt die Busy-Loop
sie und macht mit dem nächsten Scheduler-Output weiter** — der Send an die
nächste PP-Stufe unterbleibt, und die meldet dann einen völlig anderen Fehler
(hier: „size of tensor a (32) must match tensor b (2)", weil sie die Nutzlast
des Folgeschritts empfing). Die gemeldete Stelle war die Folge, nicht die
Ursache. **Bei PP-Fehlern immer zuerst prüfen, ob ein früherer Rang eine
Ausnahme geworfen hat** (`grep "WorkerProc hit an exception"`).

## MTP (k>0): bootet, rechnet aber falsch — nächster Arbeitspunkt

### Blockgröße hängt an der Spekulationstiefe

`QSAKeyStateCache.get_kv_cache_spec` verlangt, dass die Ringkapazität die
Attention-Blockgröße teilt:
`capacity = compress_ratio · ceil((compress_ratio + k) / compress_ratio)`.
Bei `compress_ratio = 4`:

| k | Kapazität | kleinste passende Blockgröße |
|---:|---:|---:|
| 0 | 4 | 16 |
| 1–4 | 8 | 16 |
| 5–8 | 12 | **48** |
| 9–12 | 16 | 16 |

Das Boot-Skript rechnet das jetzt selbst aus (`BLOCK_SIZE` überschreibbar).
**Blockgröße 48 sprengt den Speicher** — jeder CSA-Block wird dreimal so
groß, ein Rang bleibt bei ≤ 0 Bytes KV. Praktikabel sind also k ≤ 4 und
k = 9…12. Zusätzlich muss `GMU` auf **0,95** (bei 0,90 reicht es mit
Drafter-Gewichten nicht).

### Stand k=4: Prefill korrekt, erster Spekulationsschritt liefert NaN

**Zwei getrennte Befunde, sauber isoliert.**

**(a) Der E5-Metadaten-Cache verdirbt schon den Prefill.** Mit dem Default
`VLLM_SM70_E5_CACHE=1` liefert das Zielmodell als erstes Token `<|im_end|>`
(logprob -0,0097), bei jedem Prompt. Der Cache (`_e5_md_hit` /
`_e5_apply_ints`, `gpu_model_runner.py` ~9328/1292) setzt eine
Blocktabellen-Form voraus, die der QSA-Ring -- ein Block pro Request -- nicht
hat; bei k=1 fliegt er sogar hart mit
`output with shape [] doesn't match the broadcast shape [1]`.
**Mit `VLLM_SM70_E5_CACHE=0` ist der Prefill korrekt.** Fuer dieses Modell muss
der Schalter also aus; ob der Cache reparabel ist oder fuer CSA-Modelle
grundsaetzlich ausscheidet, ist offen.

**(b) Danach bleibt: der erste Spekulationsschritt rechnet NaN.**
Gemessen mit `VLLM_SM70_E5_CACHE=0`, k=4, greedy:

| max_tokens | Ausgabe |
|---:|---|
| 1 | ` Paris` OK |
| 2 | ` Paris!` |
| 3 | `!!!` |
| 12 | `!!!!!!!!!!!!` |

`!` ist Token-ID 0 -- die klassische Signatur von NaN-Logits, deren Argmax auf
Index 0 faellt. Bestaetigt durch die Logprob-Serialisierung, die mit
`ValueError: Out of range float values are not JSON compliant: nan` abbricht.

**Damit ist die Lage praezise eingegrenzt:** Prefill (kein Spekulationspfad
aktiv) rechnet korrekt -- der gesamte portierte Stapel ist also gesund. Erst
der erste Draft/Verify-Schritt produziert NaN. Verdaechtige, in dieser
Reihenfolge:

1. **Puffer, die in Runde 0 uninitialisiert gelesen werden** --
   `num_accepted` / `num_decode_draft_tokens`, die der Runner den
   Metadata-Buildern reicht. Das Merge-Handover nennt genau diesen Verdacht
   fuer seine "Restbaustelle 2" (qwen3_5, k>0 heterogen, Wortsalat ab Token 1).
2. **Der QSA-Ring bei Kapazitaet 8** (statt 4 bei k=0): er haelt jetzt zwei
   Kompressionsgruppen, die Slot-Arithmetik (`circular_qsa_slot_mapping`)
   muss das tragen.
3. **Der PLE-Conv-State** mit `conv_state_len + num_spec` = 13 statt 9.

### Die Ursache ist gefunden: das CUDA-Graph-Capture

`--enforce-eager` (plus `VLLM_SM70_E5_CACHE=0`, `GMU=0.97`) macht MTP
**vollstaendig kohaerent** -- Kohaerenztest **8/8**, wie bei k=0:

```
[capital ] Paris          [recall  ] Mercury
[arith   ] 1591           [prose   ] sinnvolle Erklaerung
[count   ] 10             [code    ] s[::-1]
[seq     ] Muster korrekt [longctx ] apple, brick, candle, dune, ember,
                                     forge, granite, harbor.
```

Damit ist die Diagnose eindeutig: **die NaN kommen aus dem Graph-Replay, nicht
aus der Numerik.** Der portierte Stapel rechnet mit Spekulation richtig; das
Capture friert Adressen von Metadaten-Tensoren ein, die pro Schritt neu
allokiert werden. Genau das Muster, das das Merge-Handover als
"Restbaustelle 1" fuer den upstream-GDN-Builder beschreibt -- hier fuer
Qwen4Exp bestaetigt.

Die 14,5 tok/s des Eager-Laufs sind **nicht** aussagekraeftig: Eager kostet
mehr, als die Spekulation einbringt (k=0 mit Graphen liegt bei 28-29 tok/s).
Ein Tempo-Vergleich ist erst sinnvoll, wenn MTP mit Graphen laeuft.

**Speicher:** Eager braucht `GMU=0.97`; bei 0,95 bleibt ein Rang bei
<= 0 Bytes KV. Die Profilierung misst ohne Graph-Pools andere Spitzen.

### PIECEWISE ist keine Abkuerzung -- drei verschiedene Fehlerbilder

Gepruefte Kombinationen bei k=4, `VLLM_SM70_E5_CACHE=0`:

| Graph-Modus | Verhalten |
|---|---|
| `FULL_AND_PIECEWISE` (Default) | NaN ab dem ersten Spekulationsschritt (`!!!`) |
| `PIECEWISE` | **PP-Deadlock** -- 0 % GPU-Last, `shm_broadcast: No available shared memory broadcast block found in 60 seconds`, Engine steht |
| `--enforce-eager` | **kohaerent, 8/8** |

Der Deadlock ist ein eigener Befund, kein abgeschwaechtes NaN: er passt auf
das Muster von Patch 7 des Merge-Projekts (asymmetrische Send/Recv-Zaehler
zwischen den PP-Stufen, NCCL-Streams verklemmen). Es sind also womoeglich
**zwei** Probleme -- eingefrorene Capture-Adressen *und* eine
PP-Spekulations-Asymmetrie, die erst ohne FULL-Capture sichtbar wird.

### Empfohlene Reihenfolge fuer die Fortsetzung

1. Die eigentliche Reparatur: die pro Schritt neu allokierten
   Metadaten-Tensoren der betroffenen Builder in persistente Puffer legen
   (upstream nutzt dafuer `build_for_cudagraph_capture`, der Fork-Runner
   ruft nur `build()`). Kandidaten sind der sm75-GDN-Builder und der
   `PleShortConvAttentionMetadataBuilder`.
2. Getrennt davon den PIECEWISE-Deadlock verfolgen -- Send/Recv-Zaehler je
   Stufe mitschreiben (das Merge-Handover beschreibt genau dieses Vorgehen
   fuer Patch 7). Er ist der einfachere der beiden Faelle, weil er ohne
   Capture auftritt.
3. Der E5-Metadaten-Cache bleibt fuer dieses Modell aus
   (`VLLM_SM70_E5_CACHE=0`); ob er fuer CSA-Modelle reparabel ist, ist eine
   eigene Frage.
4. Erst wenn MTP mit Graphen laeuft: Tempo gegen k=0 messen.

**Betriebsparameter fuer MTP heute:** `VLLM_SM70_E5_CACHE=0`,
`EXTRA_ARGS=--enforce-eager`, `GMU=0.97`, k <= 4 oder k = 9...12.

## ACHTUNG: der Code lebt nur in der venv

Die 19 geaenderten Dateien liegen **ausschliesslich** in
`.venv-sm70-130/lib/python3.12/site-packages/vllm/` und als Kopie unter
`backups/2026-08-27-qwen4exp-bringup/geaendert/`. Sie sind **nicht** in
`fork_patches/` und **nicht** in der Deploy-Liste von
`scripts/bootstrap-sm70.sh`.

**Ein Re-Bootstrap oder eine pip-Neuinstallation loescht die gesamte Arbeit.**

Dasselbe gilt fuer die 33 neuen PR-Dateien unter `vllm/models/qwen4_exp/`, die
schon die Vorsitzung nur deployt und in
`backups/2026-08-26-qwen4exp-partial/neu/` gesichert hat.

Die Integration in `fork_patches/` + Bootstrap ist damit ein offener Punkt und
eine bewusste Entscheidung: sie braucht einen Mechanismus fuer den
`models/qwen4_exp/`-Baum (33 Dateien in Unterverzeichnissen), den die heutige
flache `deploy <datei> <ziel>`-Liste nicht abbildet -- analog zum `cp -r` des
`flash_linear_attention`-Baums. Vor dem naechsten Bootstrap erledigen.

## Sicherungen

`backups/2026-08-27-qwen4exp-bringup/` — `geaendert/` (19 Dateien, flach
benannt nach ihrem Pfad), das Boot-Skript und das Kohärenz-JSON.
`backups/2026-08-27-qwen4exp-dtype/vorher/` — die Ausgangsfassungen.

---

# Sitzung 2026-08-27 (Abend): MTP ist langsamer als kein MTP — Messungen und Irrwege

> Diese Sitzung hat **keinen** Fortschritt beim ursprünglich beauftragten
> NaN-Problem gebracht, sondern die Problemlage neu vermessen. Mehrere
> Aussagen der Vorsitzung stimmen so nicht. Wer hier weitermacht, sollte
> zuerst "Was widerlegt wurde" lesen.

## Kernbefund: MTP kostet Tempo statt es zu bringen

Sauber gemessen (`scripts/bench.py`-Methodik: fester Prompt, 200 Token,
`ignore_eos`, ein verworfener Aufwärmlauf, n=3):

| Konfiguration | Durchsatz | Streuung |
|---|---:|---|
| k=0 (ohne Spekulation) | **31,4 tok/s** | 31,4 / 31,4 / 31,3 |
| k=4 MTP (gesunde Variante) | **14,4 tok/s** | 14,3 / 14,4 / 14,4 |

Akzeptanzlänge 1,92 bei 23 % Draft-Annahme — der Drafter arbeitet also, kostet
aber mehr, als er einbringt. Rückrechnung: 133 ms pro Iteration, davon ~35 ms
Verify, also **~25 ms je Drafter-Schritt** — fast so viel wie die kompletten
48 Schichten des Zielmodells (31,9 ms).

## Die Hardware hat Reserve — sie wird nur nicht abgerufen

Gleicher k=0-Server, mehrere gleichzeitige Anfragen:

| parallel | gesamt | pro Strom |
|---:|---:|---:|
| 1 | 31,7 tok/s | 31,7 |
| 2 | 41,6 tok/s | 20,8 |
| 4 | **87,5 tok/s** | 21,9 |

2,8× mehr Durchsatz aus derselben Hardware. Genau diese Reserve in einen
einzelnen Strom zu übersetzen ist die Aufgabe von MTP.

## Der wichtigste Verweis: MERGE-PROJECT-HANDOVER.md

Auf **identischer Topologie** (TP=2 RTX-Stufe + TP=2 V100-Stufe, PP über die
Generationsgrenze, `CUDA_VISIBLE_DEVICES=0,2,1,4`) erreicht das Merge-Projekt
mit Qwen3.8-27B-NVFP4 und MTP k=7 **85,0 ± 1,1 tok/s** — gegen 32,2 tok/s
ohne MTP, also **2,5×**. Es liegt also nicht an TP2/PP2 und nicht an der
gesperrten P2P-Situation.

**Der entscheidende Unterschied steht in `scripts/serve-qwen38-mini.sh:145`:**

```bash
K1=$((K + 1)); K2=$((K1 * 2))
--compilation-config "{\"cudagraph_capture_sizes\":[$K1,$K2]}"
```

Die Referenz setzt die Capture-Größen **explizit** auf `[k+1, 2(k+1)]` und
umgeht damit den 1Cat-MTP-Default-Block in `engine/arg_utils.py:1885-1895`
(der greift nur, wenn `compilation_config.cudagraph_capture_sizes is None`).
Unser `serve-qwen38-flash-next.sh` überlässt die Liste der Automatik und
bekommt `[1,2,4,5,8,9,10,15,20]` — und genau die ist kaputt (s.u.).

Ausserdem fehlen unserem Skript diese Schalter, die die Referenz setzt:
`VLLM_SKINNY_LMHEAD=1`, `VLLM_SKINNY_LMHEAD_NATIVE=1`,
`VLLM_SM70_GDN_CHAIN_SPEC_FAST_BUILD=1`,
`VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_DEFAULT=0`, `VLLM_SKINNY_DROP_CT=1`,
`VLLM_SM70_QPN8_MT2=1`, `VLLM_FLASH_V100_DECODE_PARTITION_SIZE`,
`--async-scheduling`, sowie `attention_backend` und
`use_local_argmax_reduction` explizit in der Spec-Config.
Der LM-Kopf-Schalter ist der aussichtsreichste: der Drafter ruft den Kopf
über ~152.000 Vokabeleinträge bei **jedem** der k Schritte.

## WICHTIGSTER BEFUND: der skinny-NVFP4-Pfad ist für dieses Modell inaktiv

Qwen3.8-Flash-Next läuft **komplett am skinny-Kernelpfad des Forks vorbei** —
an genau den qpn/qpn2-Tensor-Core-Kerneln, für die v100-skinny existiert.
Belegt über zwei unabhängige Signale:

| Signal | Flash-Next (unser) | 27B (Referenz, `matrix-nvfp4-k0.log`) |
|---|---|---|
| `Skinny route map`-Zeilen | **0** | **30–32**, inkl. lm_head `N=124160 K=5120 -> qpn2` |
| Quelle der FP4-Warnung | `marlin_utils_fp4.py:305` | `marlin.py:282` |

Beide Dateien geben denselben Warntext aus — die Zeilennummer verrät den Pfad.
`marlin_utils_fp4.py` ist vLLMs **generischer** Weight-only-Marlin;
`model_executor/kernels/linear/nvfp4/marlin.py` ist der skinny-Kernel.

`VLLM_SKINNY_NVFP4=1` ist gesetzt und `_SKINNY_ENABLED` damit true
(`nvfp4/marlin.py:32` — ebenfalls eine **Modul-Konstante zur Importzeit**).
Der Pfad ist also aktiviert, wird für dieses Modell aber nie ausgewählt.

**Das ordnet alle Messungen dieser Sitzung neu ein.** Auch die 31,4 tok/s bei
k=0 stammen vom langsamen Pfad. Die Frage ist damit nicht mehr in erster Linie
"warum ist MTP langsam", sondern "warum greift für Qwen4Exp die
Kernel-Auswahl nicht". Der MTP-Aufschlag kommt obendrauf, weil der Drafter den
lm_head (~152.000 Vokabeleinträge) k-mal pro Iteration über den generischen
Pfad ruft.

**Nächster Schritt:** Auswahl-Logik zwischen
`model_executor/kernels/linear/nvfp4/{marlin,cutlass,emulation,flashinfer}.py`
und `layers/quantization/utils/marlin_utils_fp4.py` vergleichen — warum
resolviert der Qwen4Exp-Baum auf den generischen Kernel? Verdacht: die
PR-Modelldateien unter `models/qwen4_exp/` bauen ihre Linear-Layer an der
gepatchten Quant-Methode vorbei.

## Capture-Größen: die Messreihe

Alle mit k=4, sonst identischer Konfiguration. "Prefill-Gesundheit" ist die
mittlere Logprob über die letzten 12 Prompt-Tokens eines stark vorhersagbaren
Musters (`scripts/health_probe.py`): gesund ≈ −0,1 bis −0,3, zerstört ≈ −16,7.

| Capture-Größen | Ergebnis |
|---|---|
| `[1,2,4,8]` | gesund, −0,11 |
| `[1,2,4,5,8]` | gesund, −0,11 |
| `[1,2,4,5,8,9]` | **Hänger schon beim Graph-Capture** |
| `[1,2,4,5,8,9,10]` | **Hänger beim ersten Request** |
| `[1,2,4,5,8,10,15,20]` | **Hänger beim ersten Request** |
| `[1,2,4,5,8,9,10,15,20]` (Automatik) | Prefill zerstört, −16,75 (5× reproduziert) |
| `[5,10]` (Referenz-Schema `[k+1,2(k+1)]`) | **Hänger beim ersten Request** |

Jede Größe über 8 führt zu Hänger oder Zerstörung. `MNS=1` (wie in der
Referenz) ist **keine** Option: der Boot stirbt an
`ConstraintViolationError: Constraints violated (L['query_start_loc'].size()[0])`
— mit einer einzigen Sequenz ist `num_reqs+1` konstant 2 und verletzt die
dynamische Shape-Bedingung von torch.compile.

## Was widerlegt wurde (Vorsicht, steht teils noch falsch im Dokument)

1. **`VLLM_SM70_E5_CACHE=0` ist in dieser Konfiguration wirkungslos.**
   Läufe mit `=0` und `=1` liefern byte-gleiche Ergebnisse (−16,75, identische
   Tokens). Die Variable erreicht die Worker nachweislich (Weitergabe durch
   `setsid` + fork/spawn/forkserver nachgestellt). Sie ist trotzdem eine Falle:
   `_E5_CACHE` ist eine **Modul-Konstante zur Importzeit**
   (`v1/worker/gpu_model_runner.py:545`).
   Der Betriebsparameter im MTP-Abschnitt ist damit nicht belegt.
2. **Torch.compile ist unschuldig.** `cudagraph_mode:NONE` bei aktivem
   torch.compile ist gesund (−0,11). Nur die Graphen zählen.
3. **Compile-Cache und ModelInfo-Cache sind unschuldig.**
   `VLLM_DISABLE_COMPILE_CACHE=1` (frische Übersetzung) ändert nichts.
4. **Die "zwei sterbenden Server" waren kein vLLM-Bug**, sondern mit dem
   startenden Tool-Aufruf abgeräumte Prozesse. Seit Boot und Tests verkettet
   als verwalteter Hintergrund-Task laufen, ist kein Server mehr gestorben.
5. **`set_skip_topk` / `compact_topk_indices` sind toter Code im PR selbst**,
   keine verlorene Verdrahtung des Forks — der Proposer ist byte-identisch
   mit der PR-Fassung in `backups/2026-08-26-qwen4exp-partial/`.

## Ungeklärt

Der Lauf `serve-flash-next-mtp-k4c.log` (oben 15:58:56–16:00:35) war
**vollständig kohärent** — mit Graphen, mit der vollständigen Automatik-Liste
`[1,2,4,5,8,9,10,15,20]`, identischer Konfiguration und identischem Code
(19 portierte Dateien byte-identisch zur 16:23-Sicherung, Bytecode aktuell,
KV-Cache-Größen identisch: 21.969/34.523). Fünf spätere Läufe derselben
Konfiguration sind deterministisch zerstört. Dafür gibt es bislang **keine
Erklärung**.

Ebenso ungeklärt: Das Schadensbild dieser Sitzung ist **nicht** das im
MTP-Abschnitt beschriebene. Dort ist der Prefill korrekt und erst der
Spekulationsschritt liefert NaN; hier ist der Prefill selbst zerstört, sodass
nie ein zweites Token entsteht — kein Lauf dieser Sitzung hat je eine
SpecDecoding-Metrik erzeugt, ausser den gesunden Varianten.

## Werkzeuge, die diese Sitzung hervorgebracht hat

- `health_probe.py` — Prefill-Gesundheit als **Zahl** statt Textstichprobe.
  Hat mehr geleistet als jede Kohärenzsuite; gehört nach `scripts/`.
- `bench.py` — stationärer Decode-Durchsatz mit `ignore_eos`, n=3.
- `conc.py` — aggregierter Durchsatz bei 1/2/4 parallelen Strömen.
- `run_probe.sh` / `run_bench.sh` — Boot + Test **verkettet**, damit kein
  Leerlauf-Fenster entsteht, mit Fail-Fast nach 60 s bei Hängern.

## Betriebsnotiz

`pkill -f` killt die eigene Shell (im Dokument bereits gewarnt — ich bin
trotzdem hineingelaufen, Exit 144). Aufräumen ausschliesslich über
`nvidia-smi --query-compute-apps=pid` und explizite PIDs.

## llama.cpp-Gegenvergleich: Stand und Modellbestand

**Warum überhaupt:** vLLM liefert auf diesem Modell 31,4 tok/s (k=0) bzw.
14,4 (k=4). Ob das an vLLM oder an der Hardware liegt, entscheidet nur ein
zweiter Stack. `llama-swap` ist dieser zweite Stack.

**Blocker war der Build.** `llama-swap-autoscan` verwarf das Modell mit
`✗ unsupported architecture 'qwen4exp' — skipping`. Unser Build stand auf
`353b32d8b` (21.08.2026); die Unterstützung ist erst danach gelandet — als
buchstäblich neuester Commit auf master:
`6c84c7d5d model: add Qwen3.8-Flash-Next (qwen4exp) (#27742)`, 98 Commits
Abstand.

**Eigene Patches:** keine im produktiven Baum. `/home/mp/llama.cpp` war
sauberer master ohne lokale Commits. Der cuBLAS-Patch aus dem Worktree
`llama.cpp-pr26574` ist upstream gemerged
(`d9b6be07d ggml-cuda: provide static workspace for cuBLAS handles (#26574)`).
Ein Update verliert also nichts, was im Einsatz war.

**Vorgehen beim Rebuild** (Stand beim Schreiben: läuft):
Nicht über den laufenden Build, weil llama-swap bei jedem Modellwechsel
`llama-server` neu startet und ein halb geschriebenes Binary fatal wäre.
Stattdessen `build-new/` mit identischer Konfiguration
(`Release`, `CUDA_ARCHITECTURES=70;75`, `GGML_CUDA=ON`, `GGML_NATIVE=ON`,
`GGML_BLAS=OFF`, `GGML_CUDA_FA_ALL_QUANTS=OFF`).
Alt-Binaries gesichert unter `/home/mp/llama.cpp-build-backup/bin-353b32d8b`.
Tausch erst nach Verifikation, per `mv` (Rename stört laufende Prozesse nicht).

### Modellbestand und Namenskonvention

```
/home/mp/models/Qwen3.8-Flash-Next-180B-A4B/
  UD-Q6_K_XL/Qwen3.8-Flash-Next-180B-A4B-UD-Q6_K_XL-0000{1..6}-of-00006.gguf   158 GiB
  mtp/mtp-Qwen3.8-Flash-Next-Q8_0.gguf                                           3,9 GiB
```

Layout und Benennung folgen dem DeepSeek-Muster
(`<Modell>/<QUANT>/…` plus `<Modell>/<sidecar>/<präfix>-…`).
Der Anzeigename in AIfred wird aus dem **Dateinamen** abgeleitet, nicht aus
dem Verzeichnis — deshalb trägt der Dateiname die volle Deskription.

**Basis-GGUF von unsloth enthält KEINEN MTP-Kopf** (`blk.0`–`blk.47`, 1224
Tensoren). Der Kopf kommt separat von `quimmedes/Qwen3.8-Flash-Next-MTP-GGUF`
und ist auf Tensor-Ebene verifiziert: Architektur `qwen4exp`, **`blk.48`**,
34 Tensoren inkl. `output.weight` / `output_hc_down|up` / `token_embd`.
Basis und Kopf müssen in Checkpoint und Vokabular übereinstimmen, **nicht** in
der Quantisierung — wie bei DeepSeek (Q4-Basis, Q8_0-Draft).

llama.cpp erwartet Sidecars mit Präfix `mtp-` (`find_best_mtp`,
`common/download.cpp:641`, analog `dspark-`). Für lokale Modelle wird der Kopf
über `--model-draft` plus `--spec-type draft-mtp` eingebunden.

## Offenes Arbeitspaket für die nächste Instanz

**Frage:** Warum wählt der Qwen4Exp-Baum den skinny-NVFP4-Kernel nicht?

Einstiegspunkte:
- `model_executor/kernels/linear/nvfp4/marlin.py` — skinny-Kernel,
  `_SKINNY_ENABLED` (Zeile 32, Modul-Konstante zur Importzeit),
  `can_implement` (Zeile 278) gibt bedingungslos `True` zurück.
- `model_executor/layers/quantization/modelopt.py:1609` —
  `self.kernel = MarlinNvFp4LinearKernel(NvFp4LinearLayerConfig())`.
  Prüfen, ob der Qwen4Exp-Pfad hier überhaupt vorbeikommt.
- `model_executor/layers/quantization/utils/marlin_utils_fp4.py:305` — der
  generische Pfad, der unsere Läufe bedient (Zeilennummer in der Warnung
  verrät den Pfad!).
- Verdacht: die PR-Modelldateien unter `models/qwen4_exp/` bauen ihre
  Linear-Layer an der gepatchten Quant-Methode vorbei.

**Verifikation:** `Skinny route map`-Zeilen im Log zählen. Referenz
`matrix-nvfp4-k0.log`: 30–32 Zeilen inkl. lm_head `N=124160 -> qpn2`.
Unsere Flash-Next-Läufe: 0.

**Danach erst** MTP erneut messen — die Grundlage verschiebt sich.
Reihenfolge: Kernelpfad, dann fehlende Schalter des Referenzskripts
(`VLLM_SKINNY_LMHEAD`, `_NATIVE`, `VLLM_SM70_GDN_CHAIN_SPEC_FAST_BUILD`,
`--async-scheduling`), dann `VLLM_SM70_MTP_PROFILE=1` für die Phasenaufteilung.

**Vorschlag zur Dokumentstruktur** (nicht umgesetzt, Entscheidung offen):
Die drei Handovers getrennt lassen, aber eine kurze Einstiegsebene darüber —
Maschinen-Invarianten (P2P gesperrt, Aufräumen nur über explizite PIDs, Code
lebt nur in der venv), eine „was läuft"-Tabelle mit Repro-Befehlen und
gemessenem Durchsatz, und eine Landkarte der Dokumente mit Status. Heute
kostete das Fehlen dieser Ebene mehrere Boot-Zyklen: der funktionierende
Startbefehl und die Gate-Prüfungen lagen fertig in `serve-qwen38-mini.sh`.

## llama.cpp-Gegenmessung — und was sie NICHT bedeutet

Gleiches Werkzeug wie bei vLLM (`scripts/bench.py`, 200 Token, `ignore_eos`, n=3):

| Stack | Durchsatz |
|---|---:|
| vLLM k=0 (TP2×PP2) | 31,4 tok/s |
| llama.cpp, ohne Spekulation, 5 GPUs | **33,2 tok/s** (best 34,8) |
| vLLM k=4 MTP (gesunde Variante) | 14,4 tok/s |

**Fehlschluss, den man hier NICHT ziehen darf** (in der Sitzung zunächst
gezogen und vom User korrigiert): „llama.cpp landet auch bei ~33, also ist das
die Hardware-Grenze." Das folgt nicht — llama.cpp nutzt auf Volta ebenfalls
keinen spezialisierten Tensor-Core-Pfad. Zwei unoptimierte Stacks bei
derselben Zahl belegen keine Obergrenze.

**Was dagegen belegt ist:**
- 4 parallele Anfragen liefern **87,5 tok/s** aggregiert (gegen 31,7 bei einer).
  Die Reserve ist physisch vorhanden, Faktor 2,8.
- Das 27B erreicht auf **identischer** Topologie 32,2 ohne MTP und **85,0 mit**
  (MERGE-PROJECT-HANDOVER.md). Die Basiswerte 32,2 und 31,4 liegen praktisch
  gleichauf — der gesamte Unterschied entsteht im MTP-Pfad.

**Zielmarke für Qwen3.8-Flash-Next ist damit 60–80 tok/s**, nicht ~35.

### llama.cpp: Stand und offener Punkt

Build aktualisiert auf `6c84c7d5d` (build 10660) in `build-new/`; Alt-Binaries
unter `/home/mp/llama.cpp-build-backup/bin-353b32d8b`. **Tausch noch nicht
erfolgt** — `build/bin` ist unverändert der alte Stand.

Das Hauptmodell lädt sauber (4,5 min, ~110 GB über 5 GPUs) und liefert
33,2 tok/s. **MTP läuft noch nicht:** beide Drittanbieter-Sidecars scheitern.

| Sidecar | Fehler |
|---|---|
| `quimmedes/…-MTP-Q8_0` | `output_hc_norm.weight` fehlt in der Datei |
| `dzannotti/…-MTP-Q4_K_M` | hat `output_hc_norm`, aber `blk.0.hc_attn_norm` fehlt |

Ursache: Die Sidecars deklarieren `qwen4exp.block_count = 49`, enthalten aber
nur `blk.48`. llama.cpp allokiert daraufhin 49 Schichten und sucht `blk.0`.
Diese Dateien sind zum **Verschmelzen** mit der Basis gedacht, nicht als
eigenständiges Draft-Modell — anders als beim 27B, wo MTP im Haupt-GGUF liegt.

**Nächster Ansatz:** `dzannotti/Qwen3.8-Flash-Next-MTP-GGUF` enthält zusätzlich
einen Ordner `unsloth-UD-Q4_K_XL-mtp-shards/` — offenbar die unsloth-Basis
**mit** eingebautem MTP. Das wäre die Struktur, die llama.cpp erwartet.
Alternativ selbst konvertieren, jetzt wo `convert_hf_to_gguf.py` qwen4exp kennt.

### llama.cpp-Update: DURCHGEFÜHRT und verifiziert

`build/bin` trägt jetzt **build 10660 / `6c84c7d5d`**. Vorgehen: in `build-new/`
gebaut, dann mit `-DCMAKE_BUILD_RPATH=/home/mp/llama.cpp/build/bin` neu
gelinkt (RUNPATH war absolut und hätte das Verschieben sonst zerstört),
danach `build` → `build-old-353b32d8b`, `build-new` → `build`.

Rückfallebenen: `/home/mp/llama.cpp/build-old-353b32d8b/` (kompletter Baum)
und `/home/mp/llama.cpp-build-backup/bin-353b32d8b/` (nur Binaries).

**Regressionstest 27B** (gleiche Serve-Parameter, `scripts/bench.py`, n=3):

| Binary | Durchsatz |
|---|---:|
| alt `353b32d8b` | 26,9 tok/s (best 27,8) |
| neu `6c84c7d5d` | 27,6 tok/s (best 28,0) |

Keine Regression. Antwortqualität geprüft (Reasoning + „Paris"), MTP-Draft-
Kontext wird geladen. Achtung bei eigenen Tests: das 27B denkt zuerst — mit
`max_tokens=16` kommt `content` leer zurück, das ist kein Fehler.

### Volta-Kernel: was llama.cpp auf sm70 tut

Relevant für die Frage, wieviel der skinny-Pfad bringen kann:

- `VOLTA_MMA_AVAILABLE` (`common.cuh:275`) gilt **exklusiv** für
  `__CUDA_ARCH__ == 700` und wird **nur** in `fattn-mma-f16.cuh` benutzt —
  also ausschliesslich für Flash Attention.
- Der MMQ-Tensor-Core-Pfad hängt an `TURING_MMA_AVAILABLE` (sm75+). Die V100
  fällt auf **DP4A** zurück (`mmq-config-pascal-dp4a.cuh`); die RTX 8000
  bekommt den MMA-Pfad.
- Bei Batch 1 läuft ohnehin `mul_mat_vec_q` (bandbreitenlimitiert) — deshalb
  landen beide Stacks bei ~31–33 tok/s.

**Daraus folgt:** Der Vorteil der skinny-Kernel entsteht erst bei M > 1, also
genau wenn MTP k+1 Token gleichzeitig durchs Zielmodell schickt. Der Fork
zielt sichtbar darauf (`modelopt.py`: `if m <= 8: return ext.gemm_qpn8(...)`,
Volta-`m8n8k4`). Das erklärt, warum das 27B **mit** MTP von 32 auf 85 sprang
und ohne MTP nicht. MTP und skinny-Kernel sind kein Entweder-oder, sondern
bedingen einander.

---

# Sitzung 2026-08-28: die llama.cpp-Seite ist vermessen — und sie verschiebt die vLLM-Latte

> Diese Sitzung hat **keine** vLLM-Arbeit gemacht. Sie hat die
> Vergleichsseite sauber vermessen und dabei drei Dinge geklärt, die für
> die Fortsetzung des Ports unmittelbar relevant sind.

## Kernbefund für den vLLM-Port: die Latte liegt höher, als sie aussah

Alle llama.cpp-Werte: 5 GPUs, split 14:14:4:8:8, ctx 256512, PLE **auf der
Platte**, 400 Token je Lauf, Page-Cache gesättigt (19 GB).

| Konfiguration | wechselnde Prompts | identischer Prompt |
|---|---:|---:|
| ohne Spekulation | **33,0 ± 0,6** | 35,7 |
| mit ngram-Spekulation | **32,3** | 49,7 |
| erster Lauf nach dem Laden (kalter Cache) | — | 22–31 |
| *vLLM k=0 (Vorsitzung), PLE **im VRAM*** | | *31,4* |

**Methodik-Warnung, teuer gelernt:** Mit stets identischem Prompt bei
`temperature 0` lernt der ngram-Drafter die eigene Ausgabe auswendig —
das ergab scheinbare 49,7 tok/s und einen scheinbaren Spekulationsgewinn
von +39 %. Mit wechselnden Prompts bleibt davon **nichts** übrig; die
Spekulation kostet dann sogar rund 2 %. Wer hier misst, variiert die
Prompts. Der PLE-Cache-Effekt ist davon unberührt (gleichverteilte
Hash-Lookups), der Drafter-Effekt vollständig.

**Das ist die eigentliche Nachricht.** vLLM hält die PLE-Tabelle über
`VocabParallelEmbedding` im VRAM (26,4 GB je Rang bei TP=2) und liegt
trotzdem gleichauf mit llama.cpp, das dieselbe Tabelle bei jedem Token
von einer USB-NVMe nachliest. Der Vorsprung, den vLLM strukturell haben
müsste, kommt nicht an. Das stützt den Befund der Vorsitzung, dass
Qwen4Exp am skinny-NVFP4-Kernelpfad vorbeiläuft (0 statt 30–32
`Skinny route map`-Zeilen) — der Verlust dort ist offenbar groß genug,
um einen kompletten Plattenzugriff pro Token zu kompensieren.

**Zielmarke bleibt 60–80 tok/s**, aber die Untergrenze ist jetzt härter:
Alles unter 33 wäre schlechter als llama.cpp ohne jede Spekulation.

## Was llama.cpp kann und vLLM nicht — und umgekehrt

| | llama.cpp | vLLM |
|---|---|---|
| PLE (50,7 GiB, EIN Tensor) | nur lazy von Platte | **im VRAM**, über TP-Ränge geteilt |
| MTP für `qwen4exp` | **existiert nicht** (s.u.) | portiert, k=4 kohärent (eager) |
| skinny-Kernel | kein Äquivalent nötig (mmvq/MMQ decken M≤8) | vorhanden, aber für dieses Modell **inaktiv** |

Beides zusammen heißt: Für Qwen3.8-Flash-Next hat der vLLM-Pfad **zwei**
unabhängige strukturelle Vorteile (PLE im VRAM, MTP möglich) — er löst
sie derzeit nur nicht ein.

## MTP für qwen4exp gibt es in llama.cpp nicht (endgültig geklärt)

Drei unabhängige Belege, alle im Quelltext von build 10660:

- `conversion/qwen4exp.py`: `supports_mtp_export = False` mit dem Kommentar
  „the MTP block is a separate draft head; vLLM drops it too" — deshalb hat
  unsloths Basis nur `blk.0`–`blk.47`;
- `src/models/qwen4exp.cpp`: **0** Treffer für nextn/mtp
  (qwen35: 72, qwen35moe: 81, qwen3next: 69);
- `mtp_on_hybrid_qwen` (`llama-model.cpp:2429`) listet QWEN3NEXT/QWEN35/
  QWEN35MOE/BAILINGMOE3, **nicht** QWEN4EXP.

Die GGUF-Metadaten bestätigen es: Der produktive 27B trägt `arch=qwen35`
mit 4 MTP-Tensoren, Flash-Next `arch=qwen4exp` mit 0.

**Konsequenz:** Kein Anbieter-GGUF kann das ändern, und selbst ein korrekt
gemergter MTP-Kopf wäre nutzlos — es fehlt der Lesecode. Die beiden
Drittanbieter-Sidecars (quimmedes, dzannotti) sind gelöscht. Nicht erneut
suchen, nicht konvertieren. **Watch-Item für llama.cpp-Updates.**

## Die PLE-Mechanik, vollständig

`per_layer_token_embd.weight`: **50,7 GiB, Q8_0, EIN Tensor**,
shape `[160, 320001536]`, liegt komplett in Shard 3 von 6.

- llama.cpp legt ihn mit `TENSOR_READ_LAZY` an (`qwen4exp.cpp:139`) und
  liest die Zeilen bei Bedarf — Voraussetzung ist **mmap**.
- Er passt auf keine einzelne Karte (größte: 48 GB), und `-sm layer` kann
  einen Tensor nicht teilen. `--tensor-read-lazy off` legt ihn deshalb
  nicht ins VRAM, sondern in den **Host-RAM** — bei 30 GB physisch ein
  sicherer OOM (verifiziert, Abbruch bei RSS 17,8 GB).
- `-sm row` ist kein Ausweg: `ggml_backend_split_buffer_type` existiert nur
  im SYCL-Backend, nicht in CUDA. Zusätzlich `fit.cpp:477`:
  „changing weight allocation for LLAMA_SPLIT_MODE_ROW not implemented".
- Readahead ist bereits sauber behandelt: llama.cpp setzt
  `POSIX_MADV_RANDOM` gezielt auf die Lazy-Bereiche (`llama-mmap.cpp:504`),
  obwohl die Platte auf 4 MB Readahead steht.

**Der Page-Cache ist der bestimmende Faktor.** 19 GB Cache decken 37 % der
Tabelle; der Durchsatz steigt vom ersten Lauf (22–31) auf den
Sättigungswert. Mehr RAM ginge im Mini nicht (nicht erweiterbar).
Praktische Folge: **Nach jedem Modellstart ist das Modell spürbar langsam
und wird über die ersten Minuten schneller.**

## Betriebsparameter, die sich geändert haben

**`--direct-io` ist für dieses Modell tödlich.** Es schließt mmap aus, damit
fällt Lazy Read aus, und die 50,7 GiB gehen in den Hauptspeicher → OOM-Kill,
der über die systemd-Unit auch AIfred mitreißt. Beide Flags sind in
build 10660 ohnehin deprecated und schreiben dasselbe Feld (`load_mode`,
nur der letzte gewinnt) — `--mlock --direct-io` war ein stiller Widerspruch.
Richtig ist `--load-mode auto`. Für Modelle ohne Tensor > 4 GiB bleibt
`--load-mode dio` die schnellere Wahl.

**ngram-Spekulation bringt bei Prosa nichts** (32,3 gegen 33,0 ohne) — sie
ist der einzige für qwen4exp verfügbare Mechanismus, rechtfertigt sich bei
freiem Text aber nicht. Ungeprüft: strukturierter Output (Code, Listen),
wo n-Gramme erfahrungsgemäss besser treffen. Vor einem Entfernen aus der
Config dort messen.

## AIfred-Seite (committed, nicht in diesem Repo)

Die Kalibration nahm die **Dateigröße** als VRAM-Bedarf und verteilte die
PLE zusätzlich über `mb_per_layer` auf alle Layer — 157,5 statt 106,9 GB.
Folge: „nur 35/48 Layer passen auf 4 GPUs" und ein um über 100k Token zu
niedrig angesetzter Kontext. Gefixt über `get_gguf_lazy_tensor_bytes()`
mit llama.cpps eigener 4-GiB-Schwelle; `Model.size_mb` ist jetzt die
VRAM-relevante Größe. Ergebnis: 106,9 GB berechnet gegen 107,5 GB real
gemessen, und die Kalibration findet den **vollen nativen Kontext von
262.144** (split 13:14:7:7:7 über 5 GPUs, KV=f16).

Commits `f8be6a1e` (Fix) und `2dbbc3c3` (Messwerte) im AIfred-Repo,
Messtabelle in `docs/de/benchmarks/performance-history.md`.

## Beobachtung für die Kalibrierung (offen, nicht umgesetzt)

Der Kalibrationslauf brauchte **85 Minuten für 17 Probeläufe**, davon 14
allein für die Bisektion der Speed-Variante — **96 % der Zeit sind
Modell-Ladevorgänge** (je 5,0 min). `llama-fit-params` beantwortet
dieselbe Frage in ~3 s und lag in der Gegenprobe 2 % neben der Realität,
dabei auf der sicheren Seite (sagt eher mehr Bedarf voraus). Das interne
Kostenmodell musste sich dagegen über einen Bias von −3.946 MB nachführen.
Ein fit-params-Vorfilter vor jedem Probelauf wäre der größte Hebel;
der Umbau der Bisektion selbst ist der zweite Schritt.

## Nächster Schritt: zurück zu vLLM

Unverändert Punkt 1 der Vorsitzung: **Warum wählt der Qwen4Exp-Baum den
skinny-NVFP4-Kernel nicht?** Einstiegspunkte stehen im Abschnitt
„Offenes Arbeitspaket" oben. Die Messungen dieser Sitzung machen die Frage
dringender, nicht weniger: vLLM hat die PLE im VRAM und ist trotzdem
langsamer als llama.cpp mit derselben Tabelle auf der Platte.

---

# Sitzung 2026-08-28 (Vormittag): die PLE-Tabelle kaskadiert jetzt VRAM → Host-RAM

> Diese Sitzung hat den PLE-Ballast beweglich gemacht und dabei einen
> strukturellen Befund erzwungen, der jede weitere Speicherarbeit an diesem
> Modell betrifft: **bei PP bestimmt die schwächste Stufe die Kapazität aller.**
> Wer nur eine Stufe entlastet, gewinnt nichts.

## Was gebaut wurde

Die PLE-Tabelle (51,2 GB, 16 Hash-Köpfe × 20 Mio Zeilen × 160 Byte FP8) liegt
nicht mehr zwingend vollständig im VRAM. Sie wird **zeilenweise geteilt**: der
vordere Teil bleibt im Gerätespeicher, der Rest liegt in pinned Host-RAM und
wird über einen UVA-View gelesen — ohne expliziten Transfer, direkt im Gather.

Drei Stellen, alle auf vorhandenen Bausteinen:

| Datei | Was |
|---|---|
| `models/qwen4_exp/common/ple.py` | Platzierungsrechnung, geteilter Ladepfad, geteilter Gather, Automatik-Formel |
| `models/qwen4_exp/amd/ple_layer.py` | `Qwen4ExpPLEFp8EmbeddingMethod` legt beide Tabellen **lazy** an, löst das Budget auf, prüft den Host-Speicher |
| `scripts/serve-qwen38-flash-next.sh` | `PLE_HOST_GIB` (Default `auto`) |

**Der Host-Weg ist nicht neu gebaut, sondern der vorhandene:** `should_pin_memory()`
und `get_accelerator_view_from_cpu_tensor()` aus `model_executor/offloader/`,
also derselbe UVA-Mechanismus wie beim generischen `UVAOffloader`. Warum der
generische nicht reicht: er offloadet **ganze Parameter**. `ngram_embedding.weight`
ist 25,6 GB je Rang, bei TP=2 also 51,2 GB pinned — der Mini hat 30 GB. Es fehlte
allein die Granularität, nicht der Mechanismus.

**Der geteilte Ladepfad baut `copy_ple_embedding_shard_` nicht nach**, sondern ruft
sie zweimal mit den beiden Teilbereichen der TP-lokalen Zeilenachse. Die
Überlappungsrechnung bleibt eine SSOT.

### Warum lazy, und nicht nachträglich umschichten

Die Aufteilung muss **vor** dem Laden feststehen. Nachträglich wäre der volle
Tensor plus die verkleinerte Kopie gleichzeitig am Leben — Peak 48,6 GiB auf
einer 48-GiB-Karte. Ausgelöst wird die Materialisierung deshalb vom **ersten
Checkpoint-Shard**: dann ist der Modellaufbau fertig, aller übrige Gewichts-VRAM
allokiert und jeder Layer im `static_forward_context` registriert — genau die
Information, die die Automatik braucht. `create_weights` registriert bis dahin
einen Parameter mit **null Zeilen**.

## Der Zugriff kostet praktisch nichts (gemessen)

`scratchpad/ple_uva_probe.py`, echte Geometrie, RTX 8000:

| Last | VRAM | Host über UVA |
|---|---:|---:|
| 1 Token (16 Lookups) | 0,005 ms | **0,005 ms** |
| 4096 Token Prefill | 0,071 ms | 3,404 ms |

Der Link sättigt bei **3,08 GB/s** — exakt PCIe 3.0 x4, also der USB4-Tunnel.
Im Decode ist der Host-Zugriff **nicht messbar teurer** als VRAM; ein voller
262k-Prefill verlängert sich um rund 0,2 s.

Der geteilte Gather selbst (zwei Lookups plus `torch.where`, branchless) kostet
**+0,035 ms je Decode-Schritt** — 0,1 % bei 32 ms/Token. Branchless ist Absicht:
ein `.any()` auf die Indexmaske wäre ein Geräte-Sync pro Token.

**Bitgleichheit ist verifiziert**, nicht angenommen: `scratchpad/ple_split_unit.py`
prüft Laden und Lookup gegen den ungeteilten Pfad für Host-Anteile von 0 % bis
100 % und beide TP-Ränge — 44 Prüfungen, alle grün. Der Offload kann die Numerik
also nicht verändern; das war später wichtig.

## Die Kaskade rechnet selbst (Default `auto`)

`VLLM_QWEN4EXP_PLE_HOST_GIB` ist standardmäßig `auto`: die Tabelle bleibt im
VRAM, und ausgelagert wird **nur, was der Zielkontext beansprucht**. Der
KV-Bedarf kommt aus der Engine-eigenen SSOT — Summe über
`spec.max_memory_usage_bytes(vllm_config)` aller KV-Layer dieser Stufe, dieselben
Specs, die der Allokator später konsultiert. Der freie Speicher wird gemessen
(`mem_get_info`), nicht geschätzt.

Beispiel aus dem Lauf (Split 20/28, MML 262144):

```
PLE auto placement: 42.54 GiB usable at gmu=0.90, 16.65 GiB already allocated,
1.37 GiB needed for 262144 tokens of context, 2.84 GiB held in reserve
-> 2.16 GiB of the table go to host memory
```

**Der volle Kontext kostet auf Stufe 0 nur 1,37 GiB.** Kontext war nie der
Engpass dieses Modells.

Einziger nicht messbarer Term ist die Aktivierungsspitze, die die Engine erst
nach dem Platzieren profiliert. Sie steckt in `VLLM_QWEN4EXP_PLE_VRAM_RESERVE`
(Default 6 % der Karte). Gemessen lag die Lücke zwischen (Gewichte + KV) und dem
Utilization-Budget bei **1,41 / 2,24 / 2,26 / 2,28 GiB** einer 48-GiB-Karte, der
Default lässt also etwas Luft. Zu viel Reserve kostet Host-RAM, zu wenig lässt
die Engine den Kontext mit klarer Meldung ablehnen.

## WICHTIGSTER BEFUND: die schwächste PP-Stufe deckelt alle

`v1/core/kv_cache_utils.py`, im Klartext:

```python
# Change the num_blocks of each rank to the smallest among all ranks.
min_num_blocks = min(kv_cache_config.num_blocks for kv_cache_config in kv_cache_configs)
```

Deshalb ist der PLE-Offload **allein wirkungslos**: er entlastet Stufe 0 (die
RTX-Karten tragen die Tabelle), aber das Minimum sitzt auf Stufe 1 (V100, ohne
PLE). Belegt durch zwei Läufe, die sich nur im Offload unterscheiden:

| | ohne Offload | mit 2 GiB Host |
|---|---:|---:|
| Gewichte Stufe 0 | 38,65 GiB | 36,65 GiB (exakt −2,00) |
| KV-Speicher Stufe 0 | 3,14 GiB | 4,31 GiB (+1,17) |
| **KV-Kapazität** | **489.028 Tok** | **489.028 Tok** |

Der Speicher wird frei — und verpufft. **Der Ertrag entsteht erst, wenn der
freigeräumte Platz in zusätzliche Layer umgesetzt wird**, die Stufe 1 entlasten.
Offload und Layer-Split greifen nur zusammen.

Nebenwirkung als Diagnosehilfe: sobald beide Stufen dieselbe Blockzahl melden,
gibt der Log die Größe nur noch **einmal** aus statt zweimal — die Stufen sind
dann ausbalanciert.

## Messreihe (MML 262144, k=0, GMU 0,90)

**Lesehinweis zur Spalte „Pool":** das ist **keine** Kontextlänge. vLLM meldet
`num_tokens = max_concurrency * max_model_len` — die Gesamtzahl der Token-Slots
im KV-Pool. Die einzelne Sequenz bleibt bei **262.144** (`max_position_embeddings`).
Die Spalte sagt also, wie viele Anfragen **voller Länge** gleichzeitig Platz
haben. Der volle Kontext lief bereits vor jeder Änderung; gewonnen wurde
**Parallelität, nicht Länge**.

| Split | PLE-Host | Gewichte St. 0 | KV-Speicher St. 0 | Pool (Slots) | Concurrency | Durchsatz | Kohärenz |
|---|---:|---:|---:|---:|---:|---:|---|
| 18/30 | 0 | 38,65 GiB | 3,14 GiB | 489.028 | 1,87x | — | — |
| 18/30 | 2 GiB | 36,65 GiB | 4,31 GiB | 489.028 | 1,87x | 31,1 | 8/8 |
| **20/28** | **auto (2,16)** | 38,00 GiB | 3,76 GiB | **729.377** | **2,78x** | **31,5** | **8/8** |
| 20/28 | 4 GiB | 36,16 GiB | 4,78 GiB | 774.095 | 2,95x | 31,6 | 8/8 |
| 22/26 | 6 GiB | 35,65 GiB | 5,27 GiB | 982.268 | 3,75x | 31,7 | **7/8** |

Referenz zum Vergleich (MML 4096, Split 18/30, kein Offload): 217.770 / 261.324
Slots, 31,4 tok/s, Kohärenz 8/8 — vom Regressionslauf dieser Sitzung **exakt**
reproduziert, bevor irgendetwas verändert wurde.

**Der Durchsatz bleibt über alle Varianten bei 31,1–31,7 tok/s.** Die Kaskade
kostet kein Tempo, der verschobene Layer-Split auch nicht.

## Split 22/26 ist nicht einsetzbar — und der Offload ist unschuldig

Bei Split 22/26 antwortet das Modell auf den `count`-Prompt im Chat-Pfad mit
einem einzelnen leeren Token: erstes Token `<|im_end|>` mit logprob **−0,059**
(94 %), zweiter Kandidat −3,0. Das ist kein numerischer Grenzfall.

Isoliert über die gespeicherten Kohärenz-JSONs:

| Lauf | Split | PLE-Host | `count` |
|---|---|---:|---|
| Referenz 27.08. | 18/30 | 0 | `10` |
| Regressionslauf | 18/30 | 0 | `10` |
| voller Kontext | 18/30 | 2 GiB | `10` |
| Automatik | 20/28 | 2,16 GiB | `10` |
| manuell | 20/28 | 4 GiB | `10` |
| **ausgereizt** | **22/26** | 6 GiB | **leer** |

**Es ist der Layer-Split, nicht der Offload** — der liefert bitgleiche Bytes
(Unit-Test) und ist bei 18/30 und 20/28 unauffällig. Split 22/26 schiebt die
Layer 18–21 auf die Turing-Seite, wo der upstream-GDN-Ersatzpfad und die
halbierte QSA-Kachel greifen. Der zuerst verdächtigte Layer 19 (fünfter
full-attention-Layer, `full_attention_interval = 4`) ist entlastet: er wandert
bei Split 20/28 ebenfalls und stört dort nicht. **Verbleibender Verdacht: die
GDN-Layer 20/21 auf sm75.** Eigener Befund über den sm75-Pfad, unabhängig von
dieser Arbeit.

## Host-RAM ist die harte Untergrenze der Kaskade

Eine dritte Ebene „notfalls Platte" gibt es **nicht**, und sie lässt sich nicht
in dieselbe Abstraktion einhängen: der zero-copy-Gather braucht **page-locked**
Speicher. Normaler oder mmap-Speicher ist für die GPU nicht adressierbar; eine
Plattenebene bräuchte einen CPU-seitigen Gather mit Transfer, also einen
Host-Umweg pro Schritt.

Empirische Obergrenze auf dieser Maschine: **12 GiB gepinnt** (6 GiB × 2 Ränge)
ließ das System auf 2 GB Restspeicher laufen. 8 GiB gepinnt sind unauffällig.
Reicht der Host nicht, bricht `_check_host_memory` mit benannter Meldung ab
statt in den OOM-Killer zu laufen — die Prüfung ist bei TP absichtlich
optimistisch (jeder Rang pinnt gleichzeitig seinen Anteil).

## Betriebsempfehlung

```bash
MML=262144 PP_PARTITION=20,28 bash scripts/serve-qwen38-flash-next.sh "$SNAP"
# PLE_HOST_GIB bleibt auf auto
```

Voller nativer Kontext (262.144 je Sequenz), 2,78 gleichzeitige Anfragen dieser
Länge, 31,5 tok/s, Kohärenz 8/8.

## ACHTUNG: der Code lebt weiter nur in der venv

Beide geänderten Dateien gehören zum `models/qwen4_exp/`-Baum, für den die
flache `deploy`-Liste in `scripts/bootstrap-sm70.sh` **keinen Mechanismus hat**
(offener Punkt der Vorsitzung, unverändert). Gesichert unter
`backups/2026-08-28-ple-vram-cascade/` — `vorher/`, `geaendert/`, `ergebnisse/`
sowie Unit-Test und Mikro-Benchmark. Ein Re-Bootstrap löscht die Arbeit.

## MTP mit vollem Kontext: die Kaskade greift am falschen Ende

Erwartung war, dass die Kaskade gerade bei MTP hilft, wo der Speicher knapp ist.
**Sie hilft dort nicht.** Zwei Läufe, Split 20/28, `--enforce-eager`, MML 262144:

| GMU | Ergebnis |
|---|---|
| 0,90 | `ValueError: No available memory for the cache blocks` |
| 0,95 | `2.14 GiB KV cache is needed … available … 1.45 GiB`, geschätzte max. Länge 177.760 |

Stufe 0 hatte dabei 3,90 GiB frei, und die Automatik lagerte korrekt **nichts**
aus — ihr lokaler Bedarf war gedeckt. **Der Engpass liegt auf Stufe 1**, wo keine
PLE liegt und stattdessen Drafter-Gewichte und Draft-Blöcke dazukommen. Ein
PLE-Offload kann dort nichts freimachen.

Es fehlten nur **0,7 GiB** — weniger als ein Layer (0,823 GiB je Rang). Die
naheliegenden Hebel sind daher ein weiter verschobener Split (Splits ≥ 22 sind
wegen des Kohärenzbefunds gesperrt) oder die vertauschte Stufenzuordnung.

**Zur Einordnung:** der MTP-Stand der Vorsitzungen lief mit MML **4096**. Der
Test hier verlangt MTP *und* vollen Kontext gleichzeitig; das ist die deutlich
härtere Anforderung, die Läufe sind also nicht direkt vergleichbar.

## Die Stufen zu vertauschen scheitert an einem Gate, nicht an der Hardware

Naheliegend wäre, die PLE auf die V100 zu legen und die Layer samt KV-Cache auf
die 48-GB-Karten — dann läge der Engpass dort, wo Platz ist. Überschlagen (6
Layer auf der V100-Stufe): 6,8 GiB gepinnt, und die RTX-Stufe käme auf **8,63 GiB
KV je Karte** statt der 1,45 GiB, an denen MTP scheitert.

Das braucht keine Codeänderung, nur `CUDA_VISIBLE_DEVICES=1,4,0,2`. **Der Boot
stirbt aber sofort:**

```
ValueError: The quantization method modelopt_fp4 is not supported for the
current GPU. Minimum capability: 75. Current capability: 70.
```

`VllmConfig._get_quantization_config` (`config/vllm.py:646`) ruft
`current_platform.get_device_capability()` **ohne device_id**, fragt also
`device_id=0` — die erste sichtbare Karte — und nimmt sie stellvertretend für das
ganze System. Steht die V100 vorn, fällt der Boot durch.

**Das Gate prüft die falsche Sache:** im laufenden Betrieb rechnen die beiden
V100 der Stufe 1 heute schon NVFP4-Layer. Es ist bisher nur nie aufgefallen, weil
die RTX per Konvention vorn steht. Ein Vertausch-Experiment setzt voraus, dass
die Prüfung über alle sichtbaren Geräte statt über Gerät 0 geht — ein Eingriff in
`config/vllm.py`, die als `vllm_config.py` in `fork_patches/` liegt.

**Offen und unbewertet bleiben damit die beiden Gegenargumente**, die gegen den
Vertausch sprechen: die V100 lesen mit 900 GB/s gegen 672 GB/s der RTX (beim
Decode zählt genau das, und vertauscht lägen ~42 von 48 Layern auf den
langsameren Karten), und es lägen ~42 statt 20 Layer auf dem sm75-Pfad, dessen
Qualität bei Split 22/26 gerade fragwürdig geworden ist.

## Der Vertausch läuft — und widerlegt die Bandbreiten-Empfehlung

Das Gate war ein vergessener Wert, keine Hardware-Grenze. `ModelOptNvFp4Config.
get_min_capability()` gab hart `75` zurück, während die Geschwister-Configs
(`ModelOptFp8Config`, Z. 419) längst `_SM70_MIN_CAP if _SM70_MODELOPT else …`
liefern. NVFP4 wurde beim Volta-Umbau schlicht ausgelassen; es fiel nie auf, weil
`VllmConfig._get_quantization_config` **Gerät 0** abfragt und die Turing-Karte
konventionell vorn steht. **Gefixt in `modelopt.py` — und in `fork_patches/`
mitgezogen**, sonst hätte der nächste Bootstrap ihn entfernt.

Damit ist `CUDA_VISIBLE_DEVICES=1,4,0,2` bootbar: die V100 werden Stufe 0 und
tragen die PLE, die RTX werden Stufe 1 und tragen den Großteil der Layer.

**Gemessen (MML 262144, k=0, Split 6/42, Automatik):**

| | Standard (RTX trägt PLE, Split 20/28) | Vertauscht (V100 trägt PLE, Split 6/42) |
|---|---:|---:|
| Durchsatz | 31,5 tok/s | **34,0 tok/s** |
| Kohärenz | 8/8 | **8/8** |
| Gewichte Stufe 0 | 38,00 GiB | 25,87 GiB |
| PLE-Auslagerung | 2,16 GiB/Rang | 3,76 GiB/Rang |

**Der Vertausch ist 8 % schneller, nicht langsamer.** Die Empfehlung weiter oben
in diesem Dokument — „V100 (HBM2) liest mit 900 GB/s, RTX 8000 (GDDR6) mit
672 GB/s … die V100 sind die *schnelleren* Karten und sollen die echten
Rechen-Layer tragen" — **trifft für dieses Modell nicht zu.** Limitierend ist
nicht die Bandbreite, sondern der Kernelpfad: der generische Marlin läuft auf
Turing über die MMA-Tensor-Cores, auf Volta nicht. Mehr Layer auf den RTX heißt
mehr Layer auf dem besseren Pfad. Das ist die Kehrseite desselben Befunds, der
als Punkt 1 offen steht (skinny-NVFP4-Pfad inaktiv).

**Nebenbefund:** 42 Layer auf dem sm75-GDN-Ersatzpfad sind sauber kohärent. Der
Split-22/26-Ausfall liegt also **nicht** an der Zahl der Layer auf Turing — der
Verdacht muss anders gefasst werden.

**34,0 tok/s liegen erstmals über llama.cpp** (33,0 mit wechselnden Prompts).

### MTP auf der vertauschten Anordnung: Speicherblocker weg, Graph-Bug bleibt

`CUDA_VISIBLE_DEVICES=1,4,0,2`, Split 6/42, k=4, **GMU 0,90**, MML 262144,
`--compilation-config {"cudagraph_capture_sizes":[1,2,4,8]}`, **ohne**
`--enforce-eager`:

- **Der Boot gelingt.** 356.622 / 423.220 Slots, 1,36x Concurrency bei vollem
  Kontext. Zum Vergleich: auf der Standardanordnung scheiterte derselbe Versuch
  bei GMU 0,90 **und** 0,95 an `No available memory for the cache blocks`. Das
  „GMU auf 0,97"-Erfordernis der Vorsitzungen ist damit erledigt.
- Die Automatik lagert hier 3,79 GiB/Rang aus; der Kontextbedarf der V100-Stufe
  beträgt nur 0,31 GiB (sie trägt bloß die Layer 0–5).
- **Der erste Prompt antwortet korrekt** (`Paris`), der zweite reißt die Engine ab:
  `TimeoutError: RPC call to sample_tokens timed out` → `EngineDeadError`.

Das ist ein **Hänger, kein NaN** — dasselbe Muster, das die Abend-Sitzung für
mehrere Capture-Größen und für PIECEWISE beschreibt. **Der MTP-Blocker war nie
ein Speicherproblem**, er sitzt im Graph-Capture. Die Speicherarbeit hat eine
Voraussetzung geschaffen, mehr nicht.

Aufräumen nach dem Absturz: die vier Worker halten VRAM und reagieren nicht auf
`SIGTERM`; `kill -9` über die PIDs aus
`nvidia-smi --query-compute-apps=pid` (niemals `pkill -f`).

## Punkt 1 ist geklärt — die Frage war falsch gestellt

„Warum wählt der Qwen4Exp-Baum den skinny-NVFP4-Kernel nicht?" unterstellte einen
Defekt. Es ist keiner. Der Checkpoint quantisiert **ausschliesslich die
MoE-Experten**:

| quantisiert (trägt `weight_scale`) | Anzahl |
|---|---:|
| `layers.N.mlp.experts.N.{gate,up,down}_proj` | 3 × 24.576 |
| `layers.N.ple.ple_embedding.ngram_embedding` | 1 |

`exclude_modules` deckt alles andere ab: `*.self_attn.*`, `*.linear_attn.*`,
`*.mlp.gate*`, `*.mlp.shared_expert.*`, `*hyper_connection*`, `*.ple.*`,
`lm_head`, `embed_tokens`. **Es gibt keine quantisierten Linear-Layer**, also
kann der skinny-*Linear*-Kernel per Konstruktion nie laufen. Die „0 statt 30
`Skinny route map`-Zeilen" sind eine Struktureigenschaft, kein Defekt.

Das 27B kommt über `quant_algo = MIXED_PRECISION` mit **dense** MLP-Layern
(`mlp.gate_proj/up_proj/down_proj` als NVFP4) auf den Linear-Pfad; Flash-Next
deklariert global `NVFP4` und hat nur MoE. Gegenprobe: `VLLM_NVFP4_GEMM_BACKEND=marlin`
erzwingen ändert **nichts** (34,0 tok/s vorher wie nachher, weiter 0 Route-Zeilen) —
es gibt schlicht keine Linear-Layer für den Kernel.

### Der eigentliche Hebel ist der MoE-Pfad — und er ist zweifach blockiert

**(1) Der Backend-Name war unerreichbar.** `fused_moe/oracle/nvfp4.py` mappt
`"sm70_skinny"` auf `NvFp4MoeBackend.SM70_SKINNY`, aber das `MoEBackend`-Literal
in `config/kernel.py` listete ihn nicht — `--moe-backend sm70_skinny` wurde von
argparse abgewiesen. Auto-Select erreicht ihn ebenfalls nie: in
`AVAILABLE_BACKENDS` steht `MARLIN` **vor** `SM70_SKINNY` und ist hier
unterstützt, die Suche endet einen Eintrag zu früh.
**Gefixt**: Literal ergänzt, `fork_patches/kernel_config.py` angelegt und
`deploy kernel_config.py vllm/config/kernel.py` in `bootstrap-sm70.sh` eingetragen.
Danach meldet der Log `Using 'SM70_SKINNY' NvFp4 MoE backend` und
`Using Nvfp4SkinnySm70Experts`.

**(2) Die Expertengeometrie passt nicht.** `moe_intermediate_size = 640`:

| Konfiguration | K (down_proj) | Kernel-Check | Ergebnis |
|---|---:|---|---|
| TP=2 | 320 | `k % 128 == 0` | `K must be a multiple of 128` |
| TP=1 | 640 | `k % 256 == 0` (Z. 1089) | `K must be a multiple of 256` |

640 = 5 × 128, also durch 128 teilbar, durch 256 nicht.
**Wichtig: 128er-Kernel existieren** (`kernels/skinny_kernels.cu`, Z. 732, 764,
2069) und würden K=640 akzeptieren — nur der Dispatch wählt den 256er-Pfad
(Z. 1089). Der skinny-MoE ist für dieses Modell also **nicht prinzipiell
ausgeschlossen, sondern eine Frage der Kernel-Auswahl**. Das ist der konkrete
Einstiegspunkt; es bedeutet CUDA-Arbeit mit Neubau, kein Nebenbei-Fix.

### TP=1 ist jetzt möglich — die Kaskade hebt das Ausschlusskriterium auf

Das Dokument nennt weiter oben „Layer 1 passt auf keine einzelne Karte ⇒ TP=1/PP=5
ist unmöglich, es braucht TP ≥ 2". Diese Bedingung galt **nur wegen der
PLE-Tabelle**. Mit der Kaskade trägt eine einzelne RTX sie:

```
PLE table placement: 218622473 of 320001536 rows on device (32.58 GiB),
101379063 rows in host memory (15.11 GiB)
```

TP=1/PP=4 lädt damit vollständig durch. Es scheitert erst an der KV-Struktur:

```
ValueError: CSA+linear sharded mamba cache owners must use one spec.
```

Derselbe Fehler tritt bei TP=2 mit Split 3/45 auf. Gemeinsame Ursache: **eine
Stufe braucht mindestens einen Full-Attention-Layer.** `layer_types` beginnt mit
drei `linear_attention`, der erste `full_attention` ist **Layer 3**; eine Stufe 0
mit weniger als vier Layern hält nur GDN- und PLE-ShortConv-Zustände, und die
CSA+linear-Allokation kommt mit zwei Mamba-Typen ohne Attention-Owner nicht
zurecht. Split 6/42 (TP=2) läuft, weil Layer 3 enthalten ist.

### TP=1 ist gemessen — und deutlich schlechter

Mit Split 4/15/15/14 (Stufe 0 enthält Layer 3) bootet TP=1/PP=4 vollständig:
554.769 Slots, 2,12x, voller Kontext. Aber:

| Topologie | Durchsatz | Kohärenz |
|---|---:|---|
| TP=2 / PP=2, vertauscht, Split 6/42 | **34,0 tok/s** | 8/8 |
| TP=1 / PP=4, Split 4/15/15/14 | **24,9 tok/s** | 7/8 (`count` → „user") |

**−27 %.** Damit ist die Erwartung dieses Dokuments widerlegt, TP sei „der
eigentliche Kostenfaktor … und die lohnendste Optimierungsstelle". Die Rechnung
„96 All-Reduces × 30–50 µs = 3–5 ms von 32 ms" zählt nur die Kosten. Die
Gegenseite fehlt: **TP=2 halbiert die Rechenzeit jedes Layers**, weil zwei Karten
daran arbeiten. Dieser Gewinn ist größer als der Kommunikationsaufwand — auch bei
gesperrtem P2P. TP ist hier kein Ballast, sondern der Grund für das Tempo.

Der Weg bleibt trotzdem dokumentiert, weil er eine Voraussetzung geklärt hat: die
Kaskade macht TP=1 überhaupt erst bootbar. Für den Durchsatz lohnt er nicht.

### Der simt-Umweg ist kein Workaround

Versucht: in `_expert_gemm` (`nvfp4_skinny_moe.py`) bei `K % 256 != 0` statt
`gemm_wmma` gechunktes `gemm_simt` rufen — die Dispatch-Entscheidung liegt in
Python, der Eingriff ist fünf Zeilen. Der K-Check fällt damit weg, der Boot
kommt weiter und instanziiert die Experten, stirbt aber im Graph-Capture:

```
CUDA error: operation failed due to a previous error during capture
```

**Zurückgenommen** (venv wieder identisch zu `fork_patches/nvfp4_skinny_moe.py`):
der Umweg ersetzt eine klare Fehlermeldung durch einen undurchsichtigen
Capture-Abbruch. Der skinny-MoE braucht für dieses Modell eine **wmma-Variante
mit KC=128** — echte Kernel-Arbeit mit Neubau, kein Python-Fix.

Zu bedenken: bei TP=2 ist K=320, dann bräuchte es sogar KC=64. **Expert-Parallelism**
(`--enable-expert-parallel`) wäre der Ausweg, der K bei 640 belässt, weil ganze
Experten je Rang liegen statt geteilter — ungetestet, und für den skinny-MoE erst
nach dem KC=128-Kernel relevant.

### Expert-Parallelism: gemessen, kein Gewinn

`--enable-expert-parallel` auf dem besten Betriebspunkt (vertauscht, Split 6/42,
MML 262144, k=0): **31,9 tok/s gegen 34,0**, Kohärenz 8/8, KV 797.226 Slots
(3,04x). **−6 %.** Ganze Experten je Rang statt geteilter kosten hier mehr an
All-to-All, als sie an All-Reduce sparen. Für den skinny-MoE bliebe EP dennoch
die Voraussetzung, weil es K bei 640 belässt — aber erst nach einem
KC=128-wmma-Kernel.

## Topologie-Messreihe (MML 262144, k=0, Einzelstrom)

| Topologie | Durchsatz | Kohärenz |
|---|---:|---|
| TP=2/PP=2, Standard (RTX trägt PLE), Split 20/28 | 31,5 | 8/8 |
| TP=2/PP=2, vertauscht, Split 6/42, **+EP** | 31,9 | 8/8 |
| **TP=2/PP=2, vertauscht, Split 6/42** | **34,0** | **8/8** |
| TP=1/PP=4, Split 4/15/15/14 | 24,9 | 7/8 |

Ausgangslage der Vorsitzung: 31,4 tok/s. llama.cpp: 33,0.

### Entwarnung: der Split-22/26-Ausfall ist ein Grenzfall, kein Defekt

Nachgemessen mit gleicher Offload-Menge wie der gesunde Lauf (4 GiB, Split 22/26,
MML 4096, Standardanordnung):

| Messung | Wert |
|---|---|
| Prefill-Gesundheit (`health_probe.py`) | **−0,35** (kaputt wäre −12…−15) |
| Kohärenz | 7/8 — nur `count` |
| Durchsatz | 31,9 tok/s |
| `count` über `/v1/chat/completions` | dreimal deterministisch leer |
| derselbe Prompt über `/v1/completions` | korrekt |

**Der Forward rechnet sauber.** Und der Ausfall hängt an genau dieser
Formulierung:

| Prompt | Antwort |
|---|---|
| „…word 'strawberry'? Reply with just the number." | leer |
| „…does the word 'strawberry' have?" | normal |
| „Count the letters in 'strawberry'…" | normal |
| „…word **'blueberry'**? Reply with just the number." | `9` (korrekt) |

Jede Variation funktioniert, auch die wortgleiche mit anderem Wort. Es ist eine
Grenzfallentscheidung des Modells (denken vs. sofort EOS), die von der Rundung
abhängt, und die Layer-Verteilung verschiebt sie. **Kein Vorbehalt gegen den
sm75-Pfad und damit keiner gegen den empfohlenen Betriebspunkt** (42 Layer auf
sm75, Kohärenz 8/8).

Nebenbefund derselben Messreihe: Split 22/26 **ohne** Offload bootet gar nicht
(`No available memory for the cache blocks`) — die 6 GiB des Ursprungslaufs waren
keine Zutat, sondern Voraussetzung. Der Ausfall trat auch mit 4 GiB auf, die
Kaskade ist damit endgültig entlastet.

### MTP-Stand nach dieser Sitzung — zwei Verdachtsmomente korrigiert

**(a) `build_for_cudagraph_capture` IST verdrahtet.** Die Vorsitzung nannte als
Reparatur, die Metadaten-Builder in persistente Puffer zu legen, „upstream nutzt
dafür `build_for_cudagraph_capture`, der Fork-Runner ruft nur `build()`". Das
trifft nicht zu: `gpu_model_runner.py:6477` ruft es im `for_cudagraph_capture`-Zweig,
und sowohl `gdn_attn_sm75.py:535` als auch `short_conv_attn.py:534` implementieren
es. Dieser Ansatz ist damit erledigt, bevor er begonnen wurde.

**(b) Das Fehlerbild passt nicht zu eingefrorenen Adressen.** Mit Capture-Größen
`[1,2,4,8]` auf der vertauschten Anordnung antwortet der **erste** Prompt korrekt
(`Paris`), erst der zweite reisst die Engine ab
(`TimeoutError: RPC call to sample_tokens timed out` → `EngineDeadError`).
Eingefrorene Capture-Adressen würden sofort falsch rechnen. Ein Zustandsproblem,
das erst beim zweiten Request zuschlägt, passt eher auf das Patch-7-Muster des
Merge-Projekts (asymmetrische Send/Recv-Zähler zwischen den PP-Stufen).

**(c) Eager hilft auf der vertauschten Anordnung NICHT mehr.** Auf der
Standardanordnung war `--enforce-eager` die kohärente Variante (8/8, 14,4 tok/s).
Vertauscht hängt die Engine auch damit: `bench.py` läuft in den 180-s-Timeout,
danach ist der Server tot. Die Spekulationsmetriken bleiben bei **3 Drafts,
12 Draft-Token, 0 akzeptierten** stehen.

**Konsequenz für die Priorisierung:** MTP ist ein eigenes Arbeitspaket, und sein
Ertrag ist unsicher — die Vorsitzung mass Akzeptanzlänge 1,92 bei 23 % Annahme,
und dort kostete der Drafter mehr, als er einbrachte. Der Kartentausch (+8 %) und
MTP schliessen sich derzeit gegenseitig aus; wer MTP verfolgt, muss auf der
Standardanordnung arbeiten.

## Nächste Schritte

1. **Der Kernelpfad bleibt Punkt 1** (unverändert aus der Vorsitzung): warum
   wählt der Qwen4Exp-Baum den skinny-NVFP4-Kernel nicht? 0 statt 30–32
   `Skinny route map`-Zeilen. Daran hat diese Sitzung nichts geändert.
2. **GDN-Layer auf sm75 prüfen** — Split 22/26 als reproduzierbarer Einstieg in
   ein Qualitätsproblem des upstream-Ersatzpfads.
3. **Capability-Gate über alle Geräte** statt über Gerät 0, falls die vertauschte
   Stufenzuordnung gemessen werden soll.
4. **Bootstrap-Integration des `models/qwen4_exp/`-Baums**, vor dem nächsten
   Re-Bootstrap.

---

# Sitzung 2026-08-28 (Nachmittag): MTP scheitert an zwei Defekten — keiner davon ist das Graph-Capture

> Diese Sitzung hat die MTP-Diagnose neu aufgesetzt. Die bisherige Leitthese
> (eingefrorene Capture-Adressen) ist nicht die Ursache; sie hatte sich in der
> Vormittags-Sitzung bereits zweifach selbst entkräftet. Es sind **zwei
> unabhängige Defekte**, beide unten belegt. Drei weitere Verdachtsmomente
> wurden geprüft und ausgeschlossen — sie müssen nicht erneut untersucht werden.

## Defekt 1: der Verifier läuft nicht über FLASH_ATTN_V100 — der XQA-Pfad ist unbeteiligt

**Das ist der Grund, warum MTP hier nichts einbringt.** Ausgangsbeobachtung:
„XQA path active" erscheint in **jedem** 27B-MTP-Lauf zweimal
(`s4-2x2-k7.log`, `matrix-nvfp4-k7.log`) und in **keinem einzigen**
Flash-Next-Lauf — auch nicht in `serve-flash-next-mtp-bench_k4.log`, dem Lauf
mit den 14,4 tok/s, der **mit vollen CUDA-Graphen lief** (`enforce_eager=False`,
`cudagraph_mode=FULL_AND_PIECEWISE`) und 23,9 % Akzeptanz erreichte. Eager
scheidet als Erklärung damit aus.

**Die Ursache ist struktureller Art.** `models/qwen4_exp/amd/qsa.py:296` setzt
unbedingt, ohne Plattform- oder Capability-Zweig:

```python
self.attn_backend = Qwen4ExpQSAFlashAttentionBackend
```

Die full-attention-Layer von Qwen4Exp laufen also über ein **modelleigenes
Backend** (`Qwen4ExpQSAFlashAttentionBackend(FlashAttentionBackend)`, Zeile 66)
mit eigener Implementierung `Qwen4ExpQSAFlashAttentionImpl` — deren Docstring
sagt es klar: „Run paged sparse GQA with the QSA Triton kernel"
(`.ops.qsa.qsa_sparse_paged_attention`). Der gesamte `flash_attn_v100.py`-Pfad
einschliesslich seines XQA-Verifiers ist an diesem Modell **nicht beteiligt**.
Deshalb meldet auch kein Flash-Next-Lauf je „Using AttentionBackendEnum.
FLASH_ATTN_V100 backend", während die 27B-Referenz es auf ihrer V100-Stufe tut.

**Konsequenz für die Diagnose:** Fork-Bug (b) aus dem `MERGE-PROJECT-HANDOVER.md`
(„ohne XQA rechnet der MTP-Verifier auf SM70 bei q>1 still falsch") beschreibt
das Verhalten von FLASH_ATTN_V100 und ist hier **nicht anwendbar**. Die richtige
Frage lautet stattdessen: **rechnet der QSA-Triton-Kernel beim Verify mehrerer
Draft-Token (q>1) korrekt?** Bei k=0 ist das Modell kohärent 8/8, der Kernel ist
bei q=1 also gesund; die Akzeptanzraten von 0 % bis 28 % legen nahe, dass er es
bei q>1 nicht ist. Der Kernel kennt Spekulation grundsätzlich (`ops/qsa.py:1063`
behandelt „speculative rows" explizit) — das ist der Einstiegspunkt.

**Nebenbefund, für dieses Modell folgenlos, aber festhaltenswert:** Selbst wenn
man Qwen4Exp auf FLASH_ATTN_V100 umbiegen wollte, ginge XQA nicht. Die
Geometrie passt nicht — 24 Heads / 2 KV-Heads ergibt q_per_kv = 12, und die
Grenze steht nicht im Python-Dispatch (`flash_attn_v100.py:4169`), sondern im
vorkompilierten Kernel:

```
XQA decode supports q_per_kv in {4, 6, 8}, got
staged XQA supports q_per_kv=6 only
```

Die 27B-Referenz hat 24/4 = 6 und passt. `flash_attn_v100` ist ein Fremdpaket
und liegt nur als `.so` vor; eine Instanz für 12 müsste upstream entstehen.

**Sofort umsetzbar und überfällig:** Das Boot-Skript hat kein Gate auf den
tatsächlich gewählten Verifier-Pfad. Die Referenz `scripts/serve-qwen38-mini.sh:288`
prüft ihren (XQA) und hätte jeden bisherigen Flash-Next-MTP-Lauf als nicht
zitierbar abgewiesen. Für Qwen4Exp braucht es ein eigenes, auf den QSA-Pfad
zugeschnittenes Gate.

## Defekt 2: der PP-Spec-Stash bricht — der „Hänger" ist ein maskierter Crash

Reproduktion auf der vertauschten Anordnung (`CUDA_VISIBLE_DEVICES=1,4,0,2`,
Split 6/42, k=4, GMU 0,90, MML 262144, Capture `[1,2,4,8]`): der Server
bootet, der erste Request läuft in den Timeout, und im Log stehen auf **beiden**
PP0-Rängen:

```
RuntimeError: PP spec decode: missing stashed scheduler_output
              for the hybrid state update on a non-last rank.
              gpu_model_runner.py:10649
```

Das ist die Schutzabfrage aus Patch 6 des Merge-Projekts. Stufe 0 empfängt den
Spekulations-Zustand, aber der `scheduler_output`, den ihr eigenes
`execute_model` (Zeile 9774) hinterlegen müsste, fehlt — Empfang und Stash
laufen nicht mehr im Takt. Das ist eine Buchführungs-Asymmetrie zwischen den
Stufen, dieselbe Klasse, für die Patch 7 gebaut wurde.

Warum das bisher als Hänger erschien, steht im Merge-Handover selbst: der
Executor propagiert Worker-Exceptions bei PP nicht sauber, der EngineCore
bleibt in `shm_broadcast` stehen. Der beobachtete `TimeoutError: RPC call to
sample_tokens timed out` ist die **Folge**, nicht die Ursache.

**Der Fehler steht bereits in drei Logs der Vorsitzungen** —
`serve-flash-next-mtp-eager-swap.log`, `serve-flash-next-mtp-k1.log`,
`serve-flash-next-swap-mtp.log` — wurde dort aber nie als Ursache benannt.
Wer an MTP weiterarbeitet, greppt zuerst nach `missing stashed`, bevor er
einen Deadlock diagnostiziert.

Spekulationsmetriken desselben Laufs: Akzeptanz **0,0 %** über alle vier
Positionen (`Accepted: 0 tokens, Drafted: 8 tokens`), passend zu Defekt 1.

## Geprüft und ausgeschlossen — nicht erneut untersuchen

1. **Fehlendes `--async-scheduling` ist folgenlos.** Der Verdacht lag nahe,
   weil der gesamte PP-Spec-Transport daran hängt (Sender 9768 und 10399–10406,
   Empfänger 9939) und `serve-qwen38-flash-next.sh` den Schalter nicht setzt,
   die Referenz aber schon. Er ist trotzdem falsch: `"mtp"` und
   `"qwen4_exp_mtp"` stehen in `MTPModelTypes` ⊂ `EagleModelTypes`, und
   `config/vllm.py:1047` schaltet async scheduling dann von selbst ein.
   Gegenprobe über alle 170 Logs im Repo: ausnahmslos „Asynchronous scheduling
   is enabled", auch in jedem Flash-Next-MTP-Lauf.
2. **Die fehlenden Spec-Schalter setzt 1Cat selbst.** `attention_backend=
   FLASH_ATTN_V100` und `use_local_argmax_reduction=True` erscheinen im Log als
   „Applied 1Cat SM70 MTP defaults" (`arg_utils.py:1908`), weil das Skript bei
   K>0 `VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS=1` setzt. Aus derselben Quelle
   stammt übrigens die Capture-Liste `[1,2,4,5,8,9,10,15,20]` — sie ist ein
   1Cat-MTP-Default, nicht vLLMs Automatik.
3. **Die SM70-Baseline greift.** Der device-0-Verdacht aus Session 4 des
   Merge-Projekts trifft hier nicht zu: alle Läufe zeigen 8 Zeilen
   „Auto-setting VLLM_SM70_*", die Flash-Next-Läufe wie die 27B-Referenz. Die
   Warnung „not SM70 CUDA" betrifft nur `VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH`
   und erscheint in den funktionierenden Läufen genauso.

## Werkzeuge dieser Sitzung

`scratchpad/mtp_async_ab.sh` (Boot + vier Requests + Metrik-Auswertung,
verkettet) und `mtp_std_xqa.sh` (dasselbe auf der Standardanordnung, mit einem
Prompt über 4096 Token für die XQA-Sequenzlängenbedingung). Aufräumen nach
Abstürzen weiterhin ausschliesslich über `nvidia-smi --query-compute-apps=pid`
und `kill -9`, niemals `pkill -f`.

## Betriebsnotiz: der CUDA_HOME-Default des Boot-Skripts zeigt ins Leere

`scripts/serve-qwen38-flash-next.sh` setzt `CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"`.
**Dieses Verzeichnis existiert auf dieser Maschine nicht** — apt liefert nur
12.0, der nvcc 12.8 liegt entpackt unter
`$REPO/.cuda-nvcc-deb/usr/local/cuda-12.8` (so dokumentiert im
`MERGE-PROJECT-HANDOVER.md`, dessen Repro-Kommandos ihn explizit setzen).

Der Fehler ist latent: Solange die JIT-Artefakte unter
`~/.cache/flashinfer/0.6.11.post2/70_75/cached_ops/` liegen, bootet der Server.
Verlangt eine Konfiguration einen Op, der noch nicht gebaut ist, stirbt der
Boot nach ~10 Minuten mit `RuntimeError: Ninja build failed` — die Fehlermeldung
nennt nvcc nicht, der Zusammenhang ist also nicht offensichtlich.

Bis das Skript korrigiert ist: `CUDA_HOME` bei jedem Aufruf explizit
mitgeben. Der Skript-Default sollte auf den Repo-Pfad zeigen.

## Defekt 3 (und vermutlich der auslösende): der E5-Metadaten-Cache crasht

Verifikationslauf auf der **Standardanordnung** (`CUDA_VISIBLE_DEVICES=0,2,1,4`,
Split 24/24, GMU 0,95, MML 16384, k=4, Capture `[1,2,4,8]`, **ohne** eager,
`PLE_HOST_GIB=6`): Server bootet, erster Request stirbt, danach 500er, dann tot.
Drei Fehler im Log, in dieser Reihenfolge:

```
RuntimeError: output with shape [] doesn't match the broadcast shape [1]
              gpu_model_runner.py:1292, in _e5_apply_ints        <- zuerst
RuntimeError: PP spec decode: missing stashed scheduler_output ...   <- danach
TimeoutError: RPC call to sample_tokens timed out.                   <- Folge
```

Der erste Fehler trifft **beide** PP0-Ränge und steht in `_e5_apply_ints`, dem
E5-Metadaten-Cache. Die Absturzzeile ist `t[0].copy_(torch.tensor(row, ...))`
auf `spec_state_indices_tensor`: der Cache setzt eine zweidimensionale Form
voraus, der QSA-Ring hat eine eindimensionale (ein Block pro Request), also ist
`t[0]` ein Skalar und die Zuweisung von `row` (Länge 1) scheitert.

**Damit ist die ursprüngliche Diagnose der ersten MTP-Sitzung rehabilitiert**
(„Der E5-Metadaten-Cache verdirbt schon den Prefill … setzt eine
Blocktabellen-Form voraus, die der QSA-Ring nicht hat; bei k=1 fliegt er sogar
hart mit `output with shape [] doesn't match the broadcast shape [1]`"). Die
Abend-Sitzung hatte den Cache mit dem Befund „`VLLM_SM70_E5_CACHE=0` ist
wirkungslos" entlastet — dieser Schluss trägt nicht: Dass zwei Läufe
byte-gleich waren, zeigt nur, dass in **jenem** Schadensbild der Cache nicht
der Unterschied war, nicht dass er unschuldig ist. `_E5_CACHE` ist eine
Modul-Konstante mit Default **an** (`gpu_model_runner.py:545`), sie greift also,
wenn die Variable vor dem Prozessstart im Environment steht.

**Wahrscheinliche Kausalkette** — sie ordnet Defekt 2 als Folge ein:
der E5-Crash wirft auf PP0, der Executor propagiert Worker-Exceptions bei PP
nicht sauber, die Ränge laufen auseinander, in der Folgerunde fehlt der Stash
(`missing stashed scheduler_output`), und der Timeout ist das Endstadium. Wer
Defekt 2 isoliert untersucht, arbeitet möglicherweise an einer Folgeerscheinung.

**Nächster Schritt (läuft):** derselbe Betriebspunkt mit
`VLLM_SM70_E5_CACHE=0`, vor dem Skriptaufruf gesetzt, damit die Modul-Konstante
sie sieht. Fällt der Crash weg, wird zum ersten Mal sichtbar, was der
QSA-Verifier bei q>1 tatsächlich leistet.

**Weiteres aus diesem Lauf, unabhängig bestätigt:**
- „XQA path active": **0** — auch mit V100 als letzter Stufe, vollen Graphen
  und einem Seq-Hint über 4096. Deckt sich mit Defekt 1 (eigenes QSA-Backend).
- Akzeptanz **0,0 %** über alle vier Positionen.
- `missing stashed scheduler_output` tritt **auf beiden Kartenanordnungen** auf,
  nicht nur auf der vertauschten.

## DURCHBRUCH: mit `VLLM_SM70_E5_CACHE=0` läuft MTP stabil — es lohnt sich nur nicht

Erstmals ein MTP-Betrieb ohne Absturz an diesem Modell. Betriebspunkt:
`CUDA_VISIBLE_DEVICES=0,2,1,4`, TP=2 PP=2, Split 24/24, GMU 0,95, MML 16384,
`PLE_HOST_GIB=6`, Capture `[1,2,4,8]`, **ohne** `--enforce-eager`,
`VLLM_SM70_E5_CACHE=0` **vor** dem Skriptaufruf gesetzt.

Ergebnis: keine Exception im Log, Server nach allen Requests noch oben,
Spekulation arbeitet — und damit ist auch Defekt 2
(`missing stashed scheduler_output`) verschwunden. Er war eine Folge des
E5-Crashes, keine eigenständige Ursache.

**Der Schalter war die ganze Zeit da.** Er wurde in der Abend-Sitzung nur
wirkungslos gesetzt und daraufhin verworfen. Er muss im Environment stehen,
bevor der Python-Prozess startet, weil `_E5_CACHE` eine Modul-Konstante ist.
Das Boot-Skript reicht ihn nicht durch — voranstellen genügt (er wird vererbt).

### Messreihe am identischen Betriebspunkt

| | k=0 | k=4 |
|---|---:|---:|
| Durchsatz (bench.py, 200 tok, n=3) | **32,2 tok/s** | **13,3 tok/s** |
| Streuung | 32,2 / 32,2 / 32,2 | 13,3 / 13,3 / 13,3 |
| Prefill-Gesundheit | −0,52 | −0,11 |
| Akzeptanz | — | 15,0 % |
| Akzeptanzlänge | — | 1,60 |
| Per-Position-Akzeptanz | — | 0,40 / 0,10 / 0,10 / 0,00 |

**MTP kostet 59 % des Durchsatzes.** Das ist die erste gleichbedingte Messung —
die bisherigen 14,4 gegen 31,4 stammten aus verschiedenen Läufen mit
unterschiedlichen Splits und Graph-Modi. Die Größenordnung bestätigt sich damit,
aber jetzt belastbar. Die Ursache steht in der Per-Position-Zeile: nach der
ersten Position bricht die Akzeptanz auf 10 % und dann auf null ein. Der Drafter
läuft viermal und liefert im Mittel 0,6 Zusatz-Token.

**Damit ist die Frage klar umrissen und von allem Beiwerk befreit:** warum
akzeptiert der Verifier ab Position 2 praktisch nichts? Crash, Speicherblocker,
Graph-Capture und E5-Cache sind als Störgrößen ausgeräumt; übrig bleibt der
QSA-Triton-Kernel bei q>1 (Defekt 1).

### Nebenbefund: sehr kurze Prompts kippen — und das ist NICHT MTP

Bei beiden Läufen liefert ein 6-Token-Prompt („The capital of France is") Müll,
während längere korrekt sind:

| Prompt | k=4 | k=0 |
|---|---|---|
| „The capital of France is" (6 tok) | `'\n\n\| ::\|\n\|\n\n\n\|'` | `' ")\n    assert "'` |
| „…Paris. The capital of Germany is" (13 tok) | „ Berlin. The capital of Italy is Rome" | dito, korrekt |
| „1, 2, 3, 4, 5, 6, 7," | „ 8, 9, 10," | „ 8, 9, 1" |
| „def add(a, b):\n return" | „ a + b\n\ndef subtract(a, b):" | — |

Da es bei k=0 genauso auftritt, gehört es **nicht** zu MTP. Verdacht: bei
Prompts unterhalb einer QSA-Kompressionsgruppe (`compress_ratio = 4`,
Blockgröße 16) kippt der Prefill. Auffällig ist außerdem, dass eine Wiederholung
desselben kurzen Prompts korrekt antwortet — das riecht nach Prefix-Caching, das
den kaputten Erstdurchlauf überdeckt. Eigener Untersuchungsgegenstand; er
entwertet kurze Kohärenz-Stichproben als Messinstrument.

Zur Einordnung des Betriebspunkts: Split 24/24 mit 6 GiB PLE-Offload ist
qualitativ nicht optimal (bei k=0 setzt das Modell das Prompt-Muster fort,
statt „Madrid" zu antworten — bei k=4 antwortet es korrekt). Er wurde gewählt,
weil Split 18/30 an der V100-Stufe keinen KV-Cache mehr übrig lässt.

# Sitzung 2026-08-28 (Abend): der Drafter ist gesund — es ist der QSA-Indexer

> **Die Diagnose dreht sich um.** Weder Drafter noch Verifier sind defekt.
> Gemessen mit dem Step-Dump des Forks: der Drafter trifft bis zu **5 von 5**
> Token. MTP ist trotzdem langsamer, weil ein Draft-Schritt mehr kostet als ein
> kompletter Forward des Zielmodells. Die Ursache ist benannt und upstream
> bereits gelöst — in SGLang, nicht in vLLM.

## Der Drafter produziert keinen Garbage — Beleg aus dem Step-Dump

`VLLM_SM70_MTP_DUMP_STEP_DIR=<dir>` schreibt je Runde `draft_token_ids` und
`sampled_token_ids`. Dekodiert, an einem Prompt, der sich selbst fortsetzt
(Liste von Hauptstädten), k=4:

```
akzeptiert=5 von 5   Drafter: ['.', ' The', ' capital', ' of']
                     Verifier: ['.', ' The', ' capital', ' of', ' Belgium']
akzeptiert=2 von 5   Drafter: [' is', '?', ' The', ' capital']
                     Verifier: [' is', ' Brussels', ·, ·, ·]
```

Das Muster wiederholt sich über alle 40 Runden. Der Drafter beherrscht die
**Struktur** perfekt (Satzgerüst: 5/5) und scheitert nur am **Faktenwissen**
(der Hauptstadtname nach „ is": 2/5). Genau so soll ein MTP-Kopf sich
verhalten. Metrik desselben Laufs:

```
Mean acceptance length: 5.00 · Per-position 1.000/1.000/1.000/1.000 · 100.0 %
Mean acceptance length: 3.50 · Per-position 1.000/0.500/0.500/0.500 ·  62.5 %
```

**Akzeptanzlänge 3,5 — die offizielle B200-Referenz meldet 3,3.** Unser
Drafter ist auf Augenhöhe. Die 15 % aus der Vormittagsmessung waren kein
Defekt, sondern die Schwierigkeit des bench-Prompts (freier Fachtext).
Damit sind die früheren Deutungen „Verifier rechnet still falsch" und
„Drafter produziert Müll" **beide widerlegt**.

## Der Beweis, dass es an den Kosten liegt

Gleicher Server, nur der Prompt variiert (200 Token, `ignore_eos`, Dump aus):

| Konfiguration | Akzeptanz | Durchsatz |
|---|---:|---:|
| k=0 (Referenz, schwerer Prompt) | — | **32,2 tok/s** |
| k=4, vorhersagbarer Prompt | ~3,5 von 5 | **20,3 tok/s** |
| k=4, schwerer Prompt | ~15 % | **14,0 tok/s** |

**Selbst bei nahezu perfekter Akzeptanz bleibt MTP 37 % langsamer als gar
keine Spekulation.** Rückrechnung: bei Akzeptanzlänge 3,5 und 20,3 tok/s
dauert eine MTP-Iteration 172 ms, ein reiner Target-Forward 31 ms. Die vier
Draft-Schritte kosten also zusammen ~141 ms, **je Schritt ~35 ms — mehr als
das komplette 125B-Zielmodell**. Der MTP-Block hat 4B Parameter (MoE, wenig
aktiv); er dürfte einen Bruchteil kosten. Kein Akzeptanzgewinn kann das
aufholen: Der Ertrag ist strukturell gedeckelt.

## Die Ursache: der QSA-Indexer läuft bei JEDEM Draft-Schritt

LMSYS beschreibt im Day-0-Blog zu diesem Modell genau diesen Punkt und die
Lösung, die SGLang dafür gebaut hat — **IndexShare**: die QSA-Top-k-Auswahl
aus dem Draft-Extend-Pass wird für die ganze MTP-Iteration festgehalten,
sodass jeder Draft-Decode-Schritt den Indexer überspringt. „The draft's
indexer work per MTP iteration drops from N invocations to one."

Der MTP-Block trägt den Indexer tatsächlich
(`mtp.layers.0.self_attn.indexer.index_qk_proj.weight` im Checkpoint), und
**der vLLM-PR enthält die Haken dafür — aber tot**:

- `models/qwen4_exp/amd/mtp.py:246` `set_skip_topk()` und `:252`
  `compact_topk_indices()` sind definiert;
- **beide werden nirgends im gesamten Baum aufgerufen** (verifiziert per grep);
- `skip_topk` kommt im QSA-Indexer des `amd/`-Zweigs **gar nicht vor** — die
  Zuweisung `attention.indexer.skip_topk = skip` würde ein Attribut setzen,
  das niemand liest.

Die Abend-Sitzung vom 27.08. hatte diese beiden Methoden als „toter Code im PR
selbst" abgehakt. Das stimmt — nur ist es kein harmloser Befund, sondern
**genau die fehlende Optimierung**, die den Draft-Schritt teuer macht.

## Was die Recherche sonst noch geklärt hat

- **PLE ist im Drafter bereits aus.** `mtp.py:7` „forces PLE off", und der
  Forward reicht `ngram_context=None`. Die offizielle Anforderung („the draft
  model disables PLE") ist erfüllt — dieser Hebel ist schon gezogen.
- **Der MTP-Block ist unquantisiert.** `exclude_modules` enthält `mtp.*` und
  `model.mtp.*`, im Checkpoint gibt es 0 Skalen-Tensoren unter `mtp.`. Die
  Bug-Klasse aus vLLM-Issue #43304 (Drafter erbt das Quantisierungsschema und
  scheitert am Laden) trifft uns nicht — unser Drafter lädt und rechnet.
- **Issue #36331** (0 % Akzeptanz, Qwen3.5-122B NVFP4) hat eine andere
  Signatur (`w1_weight_global_scale must match w3_weight_global_scale`), die in
  **keinem** unserer Logs vorkommt. Nicht unser Fall.
- **Der Abfall über die Draft-Positionen ist normal.** Ein MTP-Kopf wird auf
  Ground-Truth-Hidden-States trainiert, konditioniert bei Inferenz aber auf
  seine eigenen — die Akzeptanz sinkt mit der Drafttiefe bauartbedingt. Unser
  Profil (1,00 / 0,50 / 0,50 / 0,50 bei leichtem Text) ist unauffällig.
- **Pipeline-Parallelismus ist offiziell nicht vorgesehen.** Die vLLM-Recipe
  nennt einen „pipeline-parallel startup error", weil das N-Gram-Embedding PP
  zunächst nicht unterstützt. Wir fahren PP=2 — das läuft, ist aber
  ungetestetes Gelände.
- **Windowed-MTP** (arXiv 2607.21535, Code unter
  `github.com/avalliappan-nvidia/windowed-mtp-b200`) löst dieselbe Kostenfrage
  für den Langkontext-Fall: Sliding Window nur für die Draft-Attention, +28–44 %
  je Decode-Schritt, keine Trainingsänderung. Für uns zweitrangig, solange der
  Indexer N-mal läuft — aber der nächste Hebel danach.

## Nächster Schritt

IndexShare in den Qwen4Exp-Proposer verdrahten: `skip_topk` im QSA-Indexer des
`amd/`-Zweigs implementieren (Top-k-Auswahl einmal pro MTP-Iteration berechnen
und für die Draft-Schritte einfrieren) und aus dem Proposer heraus rufen. Das
ist Python plus Triton-Aufrufpfad, kein neuer Kernel. Vorbild und Nachweis der
Wirksamkeit: SGLang. Erst danach lohnt eine erneute Tempomessung.

## GEKLÄRT: der MTP-Block ist unquantisiert — er liest fast so viel wie das ganze Modell

**Das ist die Ursache, quantitativ belegt.** GPU-Zeiten aus dem eingebauten
MTP-Profiling (CUDA-Events, Mittel über 120 Iterationen, k=4, vorhersagbarer
Prompt, Akzeptanzlänge 2,88):

| Posten | ms je Iteration | Anteil |
|---|---:|---:|
| `target_forward` (komplettes 125B-Modell) | **24,3** | 17 % |
| `target_logits` | 1,7 | 1 % |
| **`draft_total` (vier Draft-Schritte)** | **83,9** | **58 %** |
| `draft_wall_cpu` (Wall-Clock inkl. Sync) | 119,9 | — |
| `state_update_*` gesamt | 1,5 | 1 % |
| `bookkeeping` | 0,1 | <1 % |

**Der Draft-Pfad kostet das 3,4-fache des kompletten Zielmodells.** Je Schritt
sind das 21 ms gegen 24,3 ms für alle 48 Layer des Targets. Gegenprobe mit dem
gemessenen Durchsatz: 120 + 24 = 144 ms je Iteration bei 2,88 akzeptierten
Token ergibt 20,0 tok/s — gemessen 19,4. Die Rechnung geht auf.

### Warum: der Checkpoint nimmt den MTP-Block von der Quantisierung aus

| Teil des Checkpoints | Größe | Format |
|---|---:|---|
| Hauptmodell | 73,27 GiB | NVFP4 |
| PLE-Tabelle | 47,75 GiB | Q8 |
| **MTP-Block** | **4,86 GiB** | **BF16, alle 31 Tensoren** |

`exclude_modules` enthält `mtp.*` und `model.mtp.*`; unter `mtp.` liegt **kein
einziger Skalen-Tensor**. Der Draft-Block liegt also in voller BF16-Breite vor,
während das Hauptmodell auf ein Viertel komprimiert ist.

Decode ist bandbreitenlimitiert: Das Hauptmodell liest ~6,5 GiB je Token, der
Draft-Block 4,86 GiB je Schritt. Vier Schritte lesen 19,4 GiB gegen 6,5 GiB des
Targets — Verhältnis 3,0 gegen gemessene 3,45. **Kein Akzeptanzgewinn kann das
aufholen:** Selbst bei 100 % Akzeptanz bliebe MTP hier ein Verlustgeschäft.

### Das erklärt auch, warum es auf der Referenzhardware nicht auffällt

Auf B200 (HBM3e, ~8 TB/s) sind 4,86 GiB rund 0,6 ms — verschwindend gegen den
Spekulationsgewinn; dort meldet die offizielle Referenz 540 tok/s bei
Akzeptanzlänge 3,3. Auf RTX 8000 (672 GB/s) und V100 (900 GB/s), zusätzlich
über zwei PP-Stufen verteilt, wird genau dieser Posten zum Hauptkostenträger.
**Es ist kein Fehler im Modell und keiner im Port, sondern eine Eigenschaft
dieses Checkpoints, die nur auf langsamerem Speicher sichtbar wird.**

### Der Hebel, mit Erwartungswert

Läge der MTP-Block ebenfalls in NVFP4 vor (4,86 → ~1,2 GiB), fiele ein
Draft-Schritt von 21 auf ~5 ms. Eine Iteration käme auf 24 + 21 + 3 = ~48 ms
für 2,88 Token, also **~60 tok/s gegen 32,2 ohne Spekulation — Faktor 1,9.**
Das ist der Ertrag, den MTP hier haben sollte.

Umsetzung: den `mtp.*`-Teil des Checkpoints nachträglich quantisieren
(ModelOpt/llm-compressor, NVFP4 wie das Hauptmodell; FP8 wäre der halbe
Gewinn und der einfachere Weg). Der PR bringt die Infrastruktur dafür schon
mit — `get_draft_quant_config` und `_remap_ignored_layers`
(`models/qwen4_exp/amd/mtp.py:117-135`) lösen die Draft-Quant-Config
unabhängig vom Target auf; ein quantisierter MTP-Block würde also sauber
geladen. Alternativ nach einem Anbieter-Checkpoint mit quantisiertem
MTP-Block suchen (für DeepSeek-V4-Flash existiert so etwas bereits:
`canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP`).

### Widerlegt: IndexShare ist NICHT unser Engpass

Der Async-CPU-Trace weist `draft_ms=6,8…8,0` je Iteration aus — im selben
Bereich wie beim 27B (6,4 ms), wo MTP 2,5x bringt. CPU-seitig ist am
Draft-Pfad nichts auffällig; die 83,9 ms sind reine GPU-Rechenzeit. Der
IndexShare-Befund der Vorsitzung (`set_skip_topk`/`compact_topk_indices` sind
toter Code) bleibt sachlich richtig und wäre eine sinnvolle Optimierung —
**aber er erklärt unsere Zahlen nicht** und ist nicht der erste Hebel.

### Nebenbefund: das MTP-Profiling ist bei PP blind

`_sm70_mtp_profile_report` bricht bei `not is_global_first_rank()` ab
(`gpu_model_runner.py:2163`), die MTP-Events entstehen aber auf der **letzten**
PP-Stufe. Bei Pipeline-Parallelismus schweigt also genau der Rang, der die
Daten hat. Für diese Messung wurde das Gate temporär aufgehoben (Diagnose-Patch
in der venv, danach zurückgenommen, venv wieder byte-identisch). Wer das
Profiling bei PP braucht, muss das Gate erneut lockern — ein Kandidat für einen
echten Fix in `fork_patches/`.

## Es gibt bereits Checkpoints mit quantisiertem MTP-Block

Recherche über die HF-API (128 Flash-Next-Repos gesichtet, Kandidaten über die
`model.safetensors.index.json` geprüft — Skalen-Tensoren unter `mtp.`). Die
MTP-Block-Größen wurden per HTTP-Range auf die Safetensors-Header gemessen,
ohne Download:

| Repo | gesamt | MTP-Block | PLE quantisiert | Downloads |
|---|---:|---:|---|---:|
| **RadixArk** (unser Bestand) | 125,9 GiB | **4,86 GiB BF16** | nein | 2297 |
| Inferact/…-NVFP4 | 170,2 GiB | 1,49 GiB NVFP4 | nein | 359 |
| starkweatherdigital/…-nvfp4 | **101,7 GiB** | **1,49 GiB NVFP4** | ja (129 Skalen) | 0 |
| provsalt/…-NVFP4-PLE-NVFP4 | **101,7 GiB** | **1,49 GiB NVFP4** | ja (256 Skalen) | 0 |
| local-inference-lab/…-4p89 | **98,6 GiB** | **1,49 GiB NVFP4** | ja, MIXED_PRECISION | 0 |
| mbehr90/…-nvfp4 | 170,2 GiB | 1,49 GiB (compressed-tensors) | nein | 0 |

Ohne quantisierten MTP-Block (alle 4,86 GiB BF16, wie unserer): PixelML,
hn7305, lovedheart (beide), primitive-ai (beide), axiomofmind, gorbatjovy,
lesj0610, wtdcode (AWQ), Intel-AutoRound. `MESHIVEAI` hat **gar keinen**
MTP-Block. Selbst quantisieren ist also unnötig.

**Der MTP-Block schrumpft von 4,86 auf 1,49 GiB — Faktor 3,3.** Hochgerechnet
auf die gemessenen GPU-Zeiten: Draft-Schritt von 21 auf ~6,4 ms, vier Schritte
25,7 statt 83,9 ms, Iteration ~53 statt 144 ms. Bei Akzeptanzlänge 2,88 ergibt
das **~54 tok/s gegen 32,2 ohne Spekulation — Faktor 1,7.**

**Zweiter, unabhängiger Gewinn:** drei der Kandidaten quantisieren zusätzlich
die PLE-Tabelle und kommen damit auf **98–102 GiB statt 125,9 GiB**. Das nimmt
24+ GiB VRAM-Druck und entschärft die Split- und Offload-Klemme, an der diese
Sitzung mehrfach hing (Split 18/30 ohne KV-Cache auf der V100-Stufe,
`PLE_HOST_GIB`-Kaskade). Ob die quantisierte PLE Qualität kostet, ist
ungeprüft — sie ist eine Hash-N-Gram-Tabelle, kein gewöhnliches Gewicht.

**Vorbehalte:** Die drei kleinen Kandidaten haben 0 Downloads (brandneu,
ungetestet). Ihr Experten-Layout unterscheidet sich vom Bestand — 6173 bzw.
4637 MTP-Tensoren statt 31, also einzelne statt gestapelter Experten; ob der
Loader beide Formen frisst, ist zu prüfen (der PR bringt `WeightsMapper` mit
`orig_to_new_stacked` mit, spricht also dafür). Inferact ist mit 359 Downloads
der einzige mit etwas Verbreitung, aber 170,2 GiB gross.

**Praktische Hürde:** Auf `/` sind nur **51 GB frei** (916 G, 95 % belegt),
HF-Cache bereits 152 G. Ein 100-GiB-Checkpoint passt erst, wenn Platz
geschaffen wird — naheliegend durch Löschen des RadixArk-Bestands (125,9 GiB),
was aber die Rückfallebene nimmt. Reihenfolge und Freigabe liegen bei Peuqui.

# Sitzung 2026-08-28 (Abend II): GELÖST — MTP bringt jetzt Faktor 1,5 bis 2,1

> **Der MTP-Block wurde durch eine NVFP4-quantisierte Fassung ersetzt. Damit
> ist das Problem erledigt.** Kein Kernel-Eingriff, kein Code-Patch, kein
> Neu-Download des Modells — eine Datei von 1,49 GiB und zwei JSON-Dateien.

## Ergebnis

Alle Werte am identischen Betriebspunkt (`CUDA_VISIBLE_DEVICES=0,2,1,4`,
TP=2 PP=2, Split 24/24, GMU 0,95, MML 16384, `PLE_HOST_GIB=6`,
Capture `[1,2,4,8]`, `VLLM_SM70_E5_CACHE=0`, 200 Token, `ignore_eos`):

| Konfiguration | schwerer Prompt | vorhersagbarer Prompt |
|---|---:|---:|
| k=0 (keine Spekulation) | 32,2 | — |
| k=4, MTP-Block **BF16** (Ausgangslage) | 14,0 | 19,4 |
| **k=4, MTP-Block NVFP4 (neu)** | **49,2** | **67,2** |

**Faktor 1,53 bzw. 2,09 gegenüber k=0 — und Faktor 3,5 gegenüber dem
BF16-Block.** Die Prognose aus der Vorsitzung (~54 tok/s) ist getroffen.

Akzeptanz steigt mit: **73,1 %, Akzeptanzlänge 3,92**, Per-Position
0,897 / 0,772 / 0,662 / 0,593 (vorher 51 % / 3,04). Prefill-Gesundheit
**−0,11**, Kohärenz einwandfrei (Hauptstädte, Zahlenfolge, Code, deutsche
Fachantwort, Faktenwissen — alles korrekt). Null Fehler im Log.

## Was gebaut wurde

`/home/mp/models/Qwen3.8-Flash-Next-180B-A4B-NVFP4-MTPQ` — 419 Dateien,
**35 MB echter Platzbedarf**. Alle Gewichte sind Symlinks in den bestehenden
RadixArk-Cache; neu geschrieben sind nur `config.json` und
`model.safetensors.index.json`, dazu ein Symlink auf den 1,49-GiB-Block aus
`provsalt/Qwen3.8-Flash-Next-NVFP4-PLE-NVFP4` (Datei
`nvfp4_experts_mtp.safetensors`, dort exklusiv). **Der RadixArk-Bestand bleibt
unangetastet und ist die Rückfallebene.**

Damit bleibt auch die PLE-Tabelle in ihrem originalen fp8 — die
Fertig-Checkpoints mit quantisiertem MTP hätten sie auf 4 Bit gedrückt
(26,88 statt 47,75 GiB) oder auf BF16 aufgebläht (Inferact: 95,43 GiB). Die
Transplantation nimmt nur das Gute aus beiden.

### Zwei Fallen beim Nachbauen

1. **Die Quant-Config dieses Checkpoints heißt `ignore`, nicht
   `exclude_modules`.** Wer `exclude_modules` bearbeitet, greift ins Leere und
   merkt es nicht.
2. **Die Wildcards in `ignore` reichen in den MTP-Block hinein.** Nach dem
   Entfernen von `mtp.*` / `model.mtp.*` bleiben vier Tensoren ungedeckt, die
   im neuen Block bewusst BF16 sind: `mtp.fc_embedding`, `mtp.fc_hidden`,
   `mtp.pre_fc_norm_embedding`, `mtp.pre_fc_norm_hidden`. Die ersten beiden
   sind echte Linear-Layer — ohne expliziten Eintrag erwartet der Loader dort
   Skalen und stirbt. Sie müssen namentlich in `ignore`. Gegenprobe: von den
   1536 quantisierten Gewichten darf keines von einem Muster erfasst werden.

Warum die Layouts trotz 31 gegen 6173 Tensoren zusammenpassen: Der neue Block
legt die 512 Experten **einzeln** ab (`weight` U8 + `weight_scale` F8_E4M3 +
`weight_scale_2`/`input_scale` F32, group_size 16) — exakt so, wie das
**Hauptmodell** des Bestands schon vorliegt. Der Loader verarbeitet diese Form
also längst; nur der alte MTP-Block war mit gestapelten, fusionierten Experten
der Sonderfall.

## Einordnung

Die Ursachenkette dieser Untersuchung, rückblickend:
Graph-Capture (falsch) → Verifier rechnet still falsch (falsch) → Drafter
produziert Müll (falsch) → **Draft-Block ist unquantisiert und liest fast so
viel wie das ganze Modell (richtig)**. Nur die letzte These überlebte die
Messung, und sie war mit einer Datei zu beheben.

Offen und weiterhin lohnend, aber nicht mehr dringend: IndexShare
(`set_skip_topk`/`compact_topk_indices` sind toter Code im PR), das die
QSA-Top-k-Auswahl einmal je MTP-Iteration statt k-mal berechnen würde.

## k-Sweep und Capture-Feintuning: k=4 mit [1,2,4,5,8] ist der Betriebspunkt

Alle Läufe am MTPQ-Modell, identischer Betriebspunkt, 300 Token, n=2:

| Konfiguration | vorhersagbar | schwer | Akzeptanz | Länge |
|---|---:|---:|---:|---:|
| k=3, capture [1,2,4,8] | 42,8 | 37,3 | 33,7 % | 2,01 |
| k=3, capture [4,8] | 43,2 | 37,6 | 34,8 % | 2,04 |
| k=4, capture [1,2,4,8] | 67,2 | 49,2 | 73,1 % | 3,92 |
| **k=4, capture [1,2,4,5,8]** | **68,2** | **51,9** | 69,8 % | 3,79 |
| k=4, capture [5,10] | 31,1 | 39,5 | 33,5 % | 2,34 |
| k=9, capture [10,20] | (54,4)* | 13,3 | **0,0 %** | 1,00 |

*k=9 „vorhersagbar" ist Prefix-Cache, keine Spekulation — 0 von 1080 Drafts
akzeptiert.

**Erkenntnisse:**
1. **Das Referenz-Schema `[k+1, 2(k+1)]` der 27B ist für dieses Modell
   FALSCH.** Es legt Größe 10 in den kaputten Bereich (>8) und drückt die
   Akzeptanz von 73 auf 33 %. Der Capture-Größen-Befund der Abend-Sitzung
   vom 27.08. (jede Größe >8 → Hänger oder Degradation) gilt unverändert —
   mit quantisiertem Draftkopf zeigt er sich als stiller Qualitätsverlust
   statt als Absturz.
2. **Größe 5 (= Verifier-Batch k+1) gehört in die Liste:** +5 % beim
   schweren Prompt (49,2 → 51,9), weil der Verifier sonst ungecaptured läuft.
3. **k=9 ist doppelt tot:** Es erzwingt Capture-Vielfache von 10 (im
   kaputten Bereich) UND die Akzeptanz kollabiert vollständig. k=5–8 bleiben
   durch die QSA-Ring-Blockgrößen-Kopplung gesperrt (Kapazität 12 → Block 48).
   k=3 verliert ein Drittel. **k=4 ist das Optimum, und es ist das einzige
   sinnvolle k.**

**Betriebsempfehlung Stand 28.08. abends:**
`K=4`, `--compilation-config {"cudagraph_capture_sizes":[1,2,4,5,8]}`,
`VLLM_SM70_E5_CACHE=0`, Modell MTPQ. Ergebnis: **51,9 / 68,2 tok/s**
(schwer / vorhersagbar) gegen 32,2 ohne Spekulation.

## IndexShare vermessen: auf kurzen Kontexten nur 1–4 % — zurückgestellt

Frage: Wie viel der Draft-Zeit entfällt auf den QSA-Indexer, den IndexShare
(SGLang) von k Aufrufen auf einen je MTP-Iteration reduzieren würde?

**Messung** (CUDA-Events um den Indexer-Aufruf in `_run_qsa`, getrennt nach
MTP-Draftkopf und Hauptmodell-Layern; eager, MML 4096, k=4, Sequenzen
<500 Token — Graph-Replay enthält die Events nicht, deshalb eager):

| Rang | Indexer je Aufruf (mtp) | je Aufruf (main) |
|---|---:|---:|
| PP1_TP0 | 0,18–0,38 ms | 0,16–0,35 ms |
| PP1_TP1 | 0,76–0,77 ms | 0,89–0,91 ms |

Die Rang-Diskrepanz ist mutmaßlich mitgemessene Sync-Wartezeit (Events
messen alles zwischen den Markern auf dem Stream); die echte Rechenzeit
liegt beim schnelleren Rang. Der Drafter hat einen Indexer-Layer, k=4
Schritte ⇒ **0,8–3,1 ms Indexer je MTP-Iteration**. IndexShare spart drei
Viertel davon: 0,6–2,3 ms auf eine ~64-ms-Iteration (Graph-Betrieb) =
**1–3,6 %**, also bestenfalls 51,9 → ~53,8 tok/s, realistisch weniger.

**Einordnung:** Der Indexer skaliert mit der Kontextlänge (Top-k über den
gesamten Kontext). Gemessen wurde bei Sequenzen unter 500 Token — bei
50k–250k Token wächst der Anteil erheblich; genau dafür hat SGLang
IndexShare gebaut. **Entscheidung: zurückgestellt.** Lohnt erst, wenn
Lang-Kontext-Betrieb real ansteht; dann `skip_topk` im Triton-Indexer
implementieren + Proposer-Verdrahtung (`set_skip_topk`/
`compact_topk_indices` liegen als tote Haken im PR bereit).

### Betriebsgrenzen, beim Messen kartiert

- `--compilation-config {"cudagraph_mode":"NONE"}` läuft in denselben
  PP-Deadlock wie PIECEWISE (shm_broadcast-Spin, 0 % Util) — der gesunde
  „NONE"-Befund der Abend-Sitzung kam über einen anderen Mechanismus.
- Eager auf Split 24/24 verträgt kein GMU 0,97: `Triton Error [CUDA]: out
  of memory` zur Laufzeit. GMU 0,90 + MML 4096 läuft.
- CUDA-Timing-Events innerhalb von Graph-Capture sind nicht auslesbar —
  Instrumentierung braucht eager UND einen `is_current_stream_capturing`-
  Guard.

Diagnose-Patch (qsa.py) nach der Messung zurückgenommen, venv byte-identisch
zum Backup `backups/2026-08-28-mtp-quant-indexshare/`.
