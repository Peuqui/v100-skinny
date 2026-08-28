# Übergabe: 1Cat-vLLM 1.3.0 + Qwen3.8-Flash-Next (Qwen4Exp) Portierung

Stand: 2026-08-27 · Vorgeschichte: [DEEPSEEK-VLLM-HANDOVER.md](DEEPSEEK-VLLM-HANDOVER.md)
(DeepSeek-Schnitt, Begründung für den Schwenk) ·
Referenzkonfiguration und 85-tok/s-Beleg: [MERGE-PROJECT-HANDOVER.md](MERGE-PROJECT-HANDOVER.md)

> ## ⚠️ ZUERST LESEN — Abend-Sitzung 27.08. hat mehrere Aussagen widerlegt
>
> Der Abschnitt „Sitzung 2026-08-27 (Abend)" ganz unten ist der aktuelle Stand.
> Vier Dinge, die weiter oben im Dokument **falsch oder überholt** stehen:
>
> 1. **`VLLM_SM70_E5_CACHE=0` ist wirkungslos**, nicht der dokumentierte
>    Betriebsparameter. Läufe mit `=0` und `=1` sind byte-gleich.
> 2. **Das Schadensbild ist ein anderes**: nicht „Prefill korrekt, NaN im
>    Spekulationsschritt", sondern **Prefill zerstört** — es entsteht nie ein
>    zweites Token.
> 3. **MTP ist derzeit langsamer als kein MTP** (14,4 gegen 31,4 tok/s).
> 4. **Der eigentliche Befund liegt vor MTP**: Qwen4Exp läuft komplett am
>    skinny-NVFP4-Kernelpfad des Forks vorbei. Das ist das offene Arbeitspaket.

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
