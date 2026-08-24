# Übergabe: PP×TP-Merge-Projekt (MTP über Pipeline-Parallelism im v100-skinny-Fork)

Stand: 2026-08-24 spätabends · Vorarbeit-Session: Benchmark-Kampagne (siehe
`/home/mp/Projekte/vllm-bench/RESULTS.md` — Methodik, alle Zahlen, Plattform-Fixe).

> **UPDATE (gleicher Abend, Folgesession): PP=2 + MTP k=7 LÄUFT.**
> Der „GDN-Spec-Commit-Deadlock" unten war eine Fehldiagnose zweiter Ordnung —
> die echten Ursachen waren (a) fehlender Draft-Token-Transport zur ersten
> PP-Stufe und (b) ein Rank-divergierendes Output-Längen-Bookkeeping. Beide
> gefixt (Patches 6+7, Abschnitt „Session 2" unten). Gemessen (bench.py,
> gleiche Methodik): **math 79,8 ± 1,2 tok/s, code 58,6 ± 2,1 tok/s** auf
> 2× V100 (PP-plain: 32,3 → 2,5×; TP2+k7-Referenz: 88 → 91 % davon).
> K=0-Regression: 32,2 tok/s (unverändert). Weiter bei Arbeitsplan-Punkt 2.

## Ziel (Peuquis Ansage)

Ein einzelnes vLLM (1Cat-Fork + v100-skinny) über die heterogene 5-GPU-Kiste
für GROSSE Modelle: **TP=2 innerhalb der RTX-Gruppe + TP=2 innerhalb der
V100-Gruppe, PP über die Generationsgrenze** (vLLM-Standard-Gitter 2×2; die
5. Karte/GPU 3 bleibt beim Vigilantia-VLM). **MTP ist Pflicht** („ohne MTP
verschenken wir jede Menge Performance"). Zielmodelle: DeepSeek-V4-Flash-Klasse
(122B ist NICHT das Wunschmodell, nur Testvehikel). Fixes sollen möglichst
universell sein — sind sie (siehe „Gültigkeitsbereich" unten).

## Was heute erreicht wurde

1. **PP=2 ohne MTP läuft** auf 2× V100 (kohärent, benchmarked: 32,3 tok/s
   plain — PP allein bringt erwartungsgemäß kein Decode-Tempo, nur Kapazität).
2. **PP=2 MIT MTP k=7: Boot komplett, alle Gates PASS, Runde 0 läuft durch
   (inkl. Drafting!), Deadlock exakt beim Eintritt in Runde 1.** Fünf Fixes
   angewendet (unten), ein lokalisiertes Problem offen (GDN-Spec-Commit über
   die PP-Grenze, Diagnose unten).

## Angewendete Patches (alle Python-Ebene, kein CUDA-Rebuild)

Deployte Dateien leben in `.venv-sm70/lib/python3.12/site-packages/vllm/...`.
Gesicherte Kopien in `fork_patches/` — ABER: `bootstrap-sm70.sh` deployt nur
seine feste Liste; **`fork_patches/qwen3_5_mtp.py` ist NICHT in der
Deploy-Liste** (bei Re-Bootstrap/pip-Reinstall von Hand nach
`vllm/model_executor/models/qwen3_5_mtp.py` kopieren!).

| Datei | Patch | Zweck |
|---|---|---|
| `v1/worker/gpu_model_runner.py` | 2× Trace `tuple(hidden_states.shape)` → hasattr-Guard | IntermediateTensors hat kein .shape (Crash in _dummy_run/profile_run auf Nicht-Letzt-Rank) |
| ebd. | 3× Spekulations-Blöcke: `and get_pp_group().is_last_rank` ergänzt (Zeilen ~11764, ~12734, ~12787) | drafter existiert nur auf letzter Stufe |
| ebd. | Init: `self.drafter = None` auf Nicht-Letzt-Ranks (~Zeile 2349) | isinstance()-Prüfungen laufen natürlich ins Leere statt AttributeError |
| `model_executor/models/qwen3_5_mtp.py` | `SupportsPP` in Basisklassen der äußeren Klasse + Import; `self.make_empty_intermediate_tensors = self.model.make_empty…` nach self.model-Erzeugung; inneres forward: Verzweigung `if intermediate_tensors is None:` statt `if get_pp_group().is_first_rank:` | Registry-Check `is_pp_supported_model` für die Draft-Config; Drafter ist aus eigener Sicht einstufig |
| `scripts/serve-qwen38-mini.sh` | von serve-qwen38-native.sh abgeleitet: TP/PP/K/GDN_PREFILL/ATTN_BACKEND/EXPECT_FP8/DISABLE_CAR/ASYNC_SCHED als Env-Parameter, Gates angepasst (Zensus 128×TP, K=0-Modus, FP8-lose Checkpoints), GPU-frei-Check nur auf CUDA_VISIBLE_DEVICES | Original unangetastet lassen |

Laufzeit-Umgebung (Pflicht auf dieser Kiste): `NCCL_P2P_DISABLE=1` und
`DISABLE_CAR=1` (Ryzen-Root-Port-P2P defekt — Symptom: beide Ranks 100 % Util
bei ~40 W im Spin), `CUDA_DEVICE_ORDER=PCI_BUS_ID` + numerische GPU-Indizes
(1Cat parst UUIDs mit int()), `CUDA_HOME=/home/mp/Projekte/v100-skinny/.cuda-nvcc-deb/usr/local/cuda-12.8`
(nvcc 12.8 aus entpackten NVIDIA-debs — apt hat nur 12.0, pip-Wheel hat keinen nvcc).
Für PP+MTP zusätzlich: `VLLM_QWEN35_MTP_SHARE_IO_WEIGHTS=0` (geteiltes
Embedding liegt auf Stufe 0, Drafter auf letzter Stufe → PPMissingLayer-Crash).

## Repro-Kommando (aktueller Stand → Deadlock in Runde 1)

```bash
cd /home/mp/Projekte/v100-skinny
SNAP=$(ls -d /home/mp/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/*/)
TP=1 PP=2 PORT=8020 K=7 DISABLE_CAR=1 NCCL_P2P_DISABLE=1 ASYNC_SCHED=1 \
VLLM_QWEN35_MTP_SHARE_IO_WEIGHTS=0 \
VLLM_SM70_ASYNC_CPU_TRACE=1 VLLM_SM70_ASYNC_CPU_TRACE_EVERY=1 \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,4 \
CUDA_HOME=$PWD/.cuda-nvcc-deb/usr/local/cuda-12.8 \
LOG=$PWD/serve-pp2-k7-trace.log \
bash scripts/serve-qwen38-mini.sh "$SNAP"
```
GPU 1+4 = die zwei freien V100 (GPU 3 hält produktiv das Vigilantia-VLM).
K=0 statt K=7 → läuft durch (PP-Basisfall, Regressionstest).

## Deadlock-Diagnose (Trace-Log `serve-pp2-k7-trace.log`)

- Symptom: `TimeoutError: RPC call to sample_tokens timed out` beim Warm-Request.
- Trace (VLLM_SM70_ASYNC_CPU_TRACE=1): **Runde 0 komplett** — PP1 (letzte
  Stufe): execute step=0 ok, sample step=0 ok mit `draft_ms=6.4` (MTP lief!),
  `output_ms=0.007` (gesendet), auffällig `state_update_ms=360`. PP0: 10×
  input_prep + 3× sample (mode=no_state, je <0,3 ms), **niemals ein
  execute-Trace für Runde 1**. Danach Stillstand beider Ranks.
- **Hypothese:** Die GDN-Recurrent-State-Verwaltung (48 der 64 Layer) braucht
  nach jeder Spekulationsrunde die Akzeptanzzahlen („spec commit"), um den
  Zustand auf den akzeptierten Stand zu setzen. Diese Info entsteht auf der
  letzten Stufe; die GDN-Layer der ERSTEN Stufe warten in ihrem Runde-1-Forward
  auf einen Commit/Zustand, der die PP-Grenze nie überquert.
- **Einstiegspunkte für die Analyse** (alle in `.venv-sm70/.../vllm/`):
  - `v1/worker/gpu_model_runner.py`: `sample_tokens` (~9649),
    `_pp_receive_prev_sampled_token_ids_to_input_batch`, Suchbegriffe
    `spec_commit`, `num_accepted`, `mamba_cache_mode` (MTP-Default „align").
  - `model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`: Ops
    `qwen_gdn_attention_core_spec_commit`, `…_003_spec`, GDN-State-Handling.
  - Frage an den Code: Wie erreichen die akzeptierten Token-Zahlen heute
    (TP-Fall) die GDN-Layer, und welcher Transportweg fehlt bei PP? Vermutlich
    huckepack auf dem Async-Kanal der Sampled-Token-IDs lösbar.
  - Falls py-spy gewünscht (Stack der hängenden Worker): NICHT installiert,
    Install braucht Peuquis Freigabe.

## Gültigkeitsbereich der Fixes (Peuquis Frage beantwortet)

Runner-Fixes: universell (alle Modelle). qwen3_5_mtp-Fixes: ganze
qwen3_5-Familie (Qwen3.5/3.6/3.8, dense+MoE, inkl. 122B); neue Familien
brauchen dasselbe Drei-Zeilen-Muster in ihrer MTP-Klasse. GDN-Commit-Problem:
nur Hybrid-Modelle (GDN/lineare Attention); reine Attention-Modelle (z. B.
DeepSeek-V4/MLA) sind davon nicht betroffen — dort könnte MTP+PP mit den
heutigen Fixes bereits laufen (ungetestet!). Plattform-Workarounds: pro
Maschine, modellunabhängig.

## Session 2 (24.08. spätabends): PP+MTP-Durchbruch — Diagnose & Patches 6+7

**Korrigierte Diagnose:** Der vermeintliche Deadlock war ein maskierter Crash:
PP0 warf in Runde 1 `RuntimeError` in `_prepare_input_ids` (~5483) — der
Scheduler plant Draft-Slots, aber die Draft-Token-WERTE existieren bei async
scheduling nur als GPU-Tensor auf der letzten Stufe (`self._draft_token_ids`).
Der Executor propagiert Worker-Exceptions bei PP nicht sauber → EngineCore
hängt in shm_broadcast → wirkt wie Deadlock. Der GDN-State-Align war das
Zweitproblem und wird vom selben Fix miterledigt.

**Patch 6 (gpu_model_runner.py): PP-Spec-Transport.** Der bestehende
PP+async-Broadcast (`_pp_broadcast_prev_sampled_token_ids`, [num_reqs,1])
wird im Spec-Fall zu einem Zwei-Teile-Protokoll mit statischen Shapes:
1. volle Sampled-Matrix `[num_reqs, k+1]`, −1-gepaddet (am alten Sendepunkt);
2. neu `_pp_broadcast_draft_token_ids`: Draft-Tokens `[num_reqs, k]` nach
   Abschluss aller Draft-Pfade (vor KV-Finalize); Drafter-Skip ⇒ Nullen
   (werden nie gelesen, Scheduler plant dann keine Spec-Slots).
Empfänger (neu `_pp_receive_spec_decode_state`): leitet aus der Matrix lokal
ab — `_count_contiguous_spec_tokens` → Akzeptanzzahlen, Gather → Next-Token,
dann `_copy_valid_sampled_token_count` (CPU-Puffer+Event+prev_sampled, exakt
wie letzte Stufe), setzt `_draft_token_ids`, ruft
`_update_states_after_model_execute(matrix, scheduler_output)` → erledigt
num_accepted-Puffer + GDN/Mamba-Align auf Rank-0-Layern (SSOT, keine
Duplikation). Dafür stasht `execute_model` auf Nicht-Letzt-Ranks das
`scheduler_output` (`_pp_nonlast_scheduler_output`). k=0-Pfad unverändert.

**Patch 7 (gpu_model_runner.py, ~3328): Output-Trim auf allen Ranks.**
In `_update_states` hing der Abgleich „output_token_ids auf num_output_tokens
zurücktrimmen" als `elif` hinter `if not is_last_rank:` → lief nur auf der
letzten Stufe. Auf PP0 akkumulierte der optimistische Spec-Extend (alle k
akzeptiert angenommen) unbegrenzten Overshoot → `num_tokens` lief davon →
`discard_request_mask` kippte → `_is_all_reqs_chunked_prefill()` divergierte
zwischen den Ranks → PP0 skippte Receives, PP1 sendete weiter → NCCL-Streams
verklemmt nach ~15 Runden. Fix: `elif` → eigenständiges `if` (läuft auf allen
Ranks). Diagnostiziert per Send/Recv-Zähler-Logging (PPDBG, wieder entfernt):
PP1 send #10-15 vs. PP0 „recv SKIP chunked".

**Verifikation:** 5+ Requests inkl. 3265-Token-Prompt (Chunked-Prefill,
Skip symmetrisch ✓), 2× 4096-Token-Läufe, Mathe korrekt, Send=Recv exakt.
Zahlen siehe UPDATE-Block oben. Beide Patches liegen in der venv;
**fork_patches-Sync + Git-Commit standen bei Sessionende noch aus.**

## Danach (Reihenfolge)

1. ~~GDN-Spec-Commit über PP lösen → PP2+k7 auf V100s messen~~ **ERLEDIGT
   (Session 2): 79,8 tok/s math / 58,6 code auf 2× V100.**
2. **Heterogener Fall — Befunde aus dem ersten 2×2-Versuch (Session 2):**
   Boot 2×RTX (Stufe 0) + 2×V100 (Stufe 1) scheitert an zwei Stellen mit
   einer Wurzel:
   - V100-Stufe: TileLang-JIT kompiliert die GDN-Kernel mit dem Target des
     falschen Devices (RTX/sm75 vorn in CUDA_VISIBLE_DEVICES) →
     `CUDA_ERROR_NO_BINARY_FOR_GPU` beim Laden auf sm70.
   - RTX-Stufe: FlashQLA-SM70-GDN braucht 86016 B dynamic shared memory —
     Volta kann 96 KB, **Turing nur 64 KB** → `tvm.error.InternalError`.
     Der sm70-Pfad ist auf Turing hardwareseitig unmöglich; trotzdem mappt
     `_resolve_gdn_prefill_backend` bei „auto" auch sm75 auf flashqla_sm70
     (qwen_gdn_linear_attn.py ~1520: `capability.minor in (0, 5)`).
   - RTX-Solo-Gegenprobe (TP=2, GDN_PREFILL=triton, ATTN_BACKEND=TRITON_ATTN,
     K=0): bootet, generiert aber Müll (temp 0 → nur „." und EOS) — der
     Fork-Triton-Zweig ist inhaltlich kaputt, wie vermutet.
   - Rückport-Basis: beide venvs sind vLLM **0.27.1**; `layers/fla/` ist
     zwischen Fork und upstream IDENTISCH — der Bruch liegt allein in den
     zwei Fork-Dateien qwen_gdn_linear_attn.py (7013 vs. 1748 Zeilen) und
     v1/attention/backends/gdn_attn.py (2029 vs. 538). Upstream 0.27.1 lief
     auf den RTX nachweislich korrekt (Kampagne: 42 tok/s plain TP=2).
   **Session 3 (24.08. nachts): sm75-Paket gebaut — heterogenes PP läuft
   inhaltlich korrekt (K=0, eager); zwei Restbaustellen.**
   Umsetzung „Per-Stufe-Dispatch auf validierten Code":
   - `qwen_gdn_linear_attn_sm75.py` + `gdn_attn_sm75.py` = unveränderte
     upstream-0.27.1-Kopien als in sich geschlossenes Paar (Querverweise
     aufeinander umgebogen); dazu upstream-FLA-Ops unverändert unter ihrem
     Originalpfad `vllm/third_party/flash_linear_attention/` (548 KB) —
     WICHTIG: die Fork-FLA-Ops (`layers/fla/ops`) sind INHALTLICH verändert
     (mutmaßliche Mojibake-Quelle des Fork-Triton-Pfads), daher upstream-Baum.
   - Registry-Entschärfungen in den Kopien: torch-Op
     `qwen_gdn_attention_core→…_sm75`, `CustomOp.register` chunk_gated_
     delta_rule/fused_rms_norm_gated →`…_sm75`, `PluggableLayer.register`
     →`…_sm75`; AWQ-Import auf Fork-`awq.AWQConfig` gemappt; lokaler
     `async_tensor_h2d` mit upstream-Semantik in gdn_attn_sm75 (Fork-Helper
     hat inkompatible Signatur). Doppel-Import-Smoke-Test beider Varianten
     in einem Prozess ist der Regressionstest dafür.
   - Einzige Weiche: qwen3_5.py instanziiert bei lokaler capability (7,5)
     `QwenGatedDeltaNetAttentionForkCall` (Adapter: Fork-Konvention
     `forward(hidden_states, output=buffer)` → upstream-Return, eine Kopie);
     die sm75-Layer-Klasse überschreibt `get_attn_backend` → sm75-Backend.
   - Runner: `is_gdn_sm75_builder`-Guard — Standard-Spec-Args ja,
     Fork-only-Args (spec_state_slot_selectors, ddtree, current_state_
     block_ids) nein.
   - cuda.py: Backend-Wahl nutzt lokales Worker-Device statt Device 0;
     sm75-Prioritätsliste hart auf [TRITON_ATTN, FLEX] (FLASHINFER-Prefill
     stirbt auf sm75 mit invalid argument). tilelang/utils/target.py:
     Compile-Target = current_device statt Device 0. Serve-Script:
     ATTN_BACKEND=AUTO (kein globales Backend) + EXTRA_ARGS-Durchreicher.
   **Testmatrix (Qwen3.8-27B-NVFP4):** RTX-solo TP2 K0 ✓ kohärent (auch
   NVFP4-Skinny-Kernel auf sm75 rechnen korrekt!); het PP2 (RTX+V100) K0
   mit CUDA-Graphs ✗ deterministisch degradiert, mit `--enforce-eager`
   ✓ KOHÄRENT → Restbaustelle 1: Fork-Graph-Capture verträgt die
   upstream-GDN-Builder nicht (Metadata-Tensoren pro Step neu allokiert,
   Replay friert alte Adressen ein). K=7 heterogen (auch eager) ✗
   Wortsalat ab Token 1, deterministisch; K=0 kohärent → Restbaustelle 2
   sitzt in der Spec-Interaktion mit dem upstream-Builder schon im
   Prefill (Verdacht: uninitialisierte num_accepted/Spec-Metadata-Puffer
   in Runde 0, oder Spec-State-Slot-Layout-Differenz upstream vs Fork;
   Page-Geometrie ist über alle Worker identisch, 832/1,71 % — das ist
   es NICHT). Rank-0-Spec-Empfang füllt auf sm75 nur noch die
   Akzeptanz-Puffer statt des Fork-Align-Postprocess (Runner,
   `_pp_receive_spec_decode_state`).
   Committed als 6dd5854 (Branch pp-mtp-merge, inkl. Bootstrap-Deploy).

   **Repro heterogenes 2×2 (aktueller Stand, K=0 kohärent nur mit eager):**
   ```bash
   cd /home/mp/Projekte/v100-skinny
   SNAP=$(ls -d /home/mp/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/*/)
   TP=2 PP=2 PORT=8020 K=0 ATTN_BACKEND=AUTO EXTRA_ARGS="--enforce-eager" \
   DISABLE_CAR=1 NCCL_P2P_DISABLE=1 ASYNC_SCHED=1 \
   CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2,1,4 \
   CUDA_HOME=$PWD/.cuda-nvcc-deb/usr/local/cuda-12.8 \
   LOG=$PWD/serve-2x2.log bash scripts/serve-qwen38-mini.sh "$SNAP"
   # 0,2 = RTX-Stufe 0 · 1,4 = V100-Stufe 1 · GPU 3 bleibt frei (VLM/TTS)
   # K=7: zusätzlich VLLM_QWEN35_MTP_SHARE_IO_WEIGHTS=0 — liefert derzeit
   # deterministischen Wortsalat (Restbaustelle 2). Kleinster Repro der
   # Restbaustellen: TP=1 PP=2 CUDA_VISIBLE_DEVICES=0,1.
   ```
   **Einstiegspunkte Restbaustelle 1 (CUDA-Graphs):** gdn_attn_sm75.py
   `build()` allokiert via async_tensor_h2d/torch.tensor pro Step neue
   GPU-Tensoren (chunk_indices/offsets, spec_sequence_masks …) — für
   FULL-Graph-Replay müssen die in persistente Puffer (Vorbild: upstream
   nutzt build_for_cudagraph_capture; Fork-Runner ruft nur build()).
   **Einstiegspunkte Restbaustelle 2 (K>0 heterogen):** Wortsalat ab
   Token 1 auch eager ⇒ schon der Prefill der RTX-Stufe kippt, sobald
   Spec aktiv ist. Verdächtige: (a) uninitialisierte
   num_accepted/num_decode_draft_tokens-Puffer, die der Runner dem
   sm75-Builder in Runde 0 reicht (Zeile ~6290 _build_attn_group_metadata);
   (b) get_state_shape/num_spec-Slot-Layout upstream vs Fork (abstract.py
   num_speculative_blocks); (c) VLLM_QWEN35_MTP_SHARE_IO_WEIGHTS=0-Effekte
   aufs Weight-Loading der Stufe 0. Diagnose-Hebel: K=7 auf RTX-SOLO
   (TP=2, letzte Stufe = Drafter lokal) testen — läuft dort Spec mit den
   sm75-Layern korrekt, liegt es an der PP-Naht, sonst an sm75+Spec selbst.
   Ursprünglicher Plan dazu: SM75-Prefill-Pfade aus upstream 0.27
   (`/home/mp/Projekte/vllm-bench/.venv/.../vllm/`) in den Fork zurückportieren
   (Fork rechnet mit Triton-Prefill auf Turing UND Volta falsch — Mojibake) und
   **Per-Stufe-Attention-Backend** bauen (PP-Stufen sind getrennte Prozesse,
   keine Attention über Stufengrenzen → chirurgisch machbar; „Triton überall"
   ist als Ausweg tot).
3. Gitter 2×2 (TP2-RTX-Stufe + TP2-V100-Stufe), ungleiche Layer-Partition via
   `VLLM_PP_LAYER_PARTITION`.
4. Zielmodell-Support prüfen (DeepSeek-V4-Ops sind in der Fork-Basis vorhanden).
5. Erkenntnisse als Issue/PR an 1Cat & v100-skinny (Autor bittet explizit um
   Reproduktionen; unsere PCIe/TP2/P2P-Befunde sind neu).
