# GLM-5.3-Flash Decode: Hostlücken (Vorbereitung vor Stufe 0, Übergaben, Lücken im Durchlauf), nsys 23.11.2026

Nur Analyse (kein Code, keine GPU, kein Git). Quelle: `~/.cache/nsys-runs/glm-v14k4-decode16k-1123/` (Capture 1 = Decode kurz, 38 Runden, 60,74 ms; Capture 2 = Decode bei 16k, 35 Runden, 62,68 ms). Baum: `1Cat-vLLM-v14`. Ergänzt `ANALYSE.md` (dort Abschnitt 2, 6, 7).

Kennzeichnung: **[gemessen]** = direkt aus der sqlite (Tabellen `CUPTI_ACTIVITY_KIND_RUNTIME/KERNEL/MEMCPY/SYNCHRONIZATION`, `NVTX_EVENTS`), **[Code]** = aus dem Quelltext gelesen, **[abgeleitet]** = Rechnung aus gemessenen Werten, **[Schätzung]** = Annahme, nicht gemessen, **[offen]** = aus nsys nicht entscheidbar. Zeiten in ms bzw. µs, jeweils „weniger ist besser". Mittelwerte über alle Decode-Runden, in Klammern [min..max].

Grenzen der Quelle: nsys hat hier keine OSRT- und keine CPU-Sampling-Tabellen; Python-Funktionen auf dem Host sind nicht sichtbar, nur die CUDA-Aufrufe und die NVTX-Bereiche. Die Zuordnung Host zu Stufe geschieht über den Thread, der die Kerne der Karte startet (Stufe 0 = Thread 2411666 / deviceId 0, Stufe 1 = 2411668 / 1, Stufe 2 = 2411667 / 2, Stufe 3 = 2411669 / 3, Stufe 4 = 2411670 / 4) [gemessen, über `correlationId`]. Der NVTX-Bereich `execute_context_0(0)_generation_1(5)` ist die Decode-Runde einer Stufe (38 bzw. 35 Stück je Stufe) und stammt aus `gpu_worker.py:1484-1505` (`annotate_profile`, um `model_runner.execute_model` herum). nsys verlängert jeden CUDA-Aufruf um einige µs; die absoluten Hostzeiten können unter nsys etwas größer sein als in Produktion [Schätzung, nicht gemessen].

## 0. Kurzfassung

| Frage | Befund |
|---|---|
| 1,55 ms vor dem ersten Stufe-0-Schichtkern (Capture 1) | **Hostgebunden, nicht GPU-gebunden.** Die GPU wartet erst auf den Broadcast der Tokens von Stufe 4 und führt danach 29 winzige Vorbereitungskerne (je 2 bis 5 µs) aus, die der Host einzeln und mit etwa 50 bis 190 µs Abstand startet (Python, Torch-Eager, Metadaten-Builder). Das Ende der Kette ist der `cudaGraphLaunch` der Stufe 0. |
| Warum in Capture 2 nur 0,26 ms | Der Host braucht in beiden Captures dasselbe: 3,48 ms (c1) bzw. 3,37 ms (c2) vom Beginn von `execute_model` bis zum `cudaGraphLaunch`. Der Broadcast kommt aber in c2 1,44 ms später an (Stufe 4 hat bei 16k Kontext einen 1,3 ms längeren Entwurf), also verdeckt die GPU-Wartezeit fast die ganze Host-Vorbereitung. Es ist derselbe Hostpfad, nur besser versteckt. |
| Übergaben 1,35 ms | Reine NCCL-Latenz (je Tensor ein eigener `SendRecv`-Kern, 5 bis 8 Stück je Übergabe, seriell) etwa 75 %, danach 25 bis 108 µs Kopien der empfangenen Tensoren in die Graph-Eingangspuffer auf der Empfängerstufe (25 %). Nicht die Bandbreite: pro Übergabe nur 0,25 bis 0,37 MB. |
| Lücken im Durchlauf 1,44 ms (Stufen 1 bis 4) | Zu 1,23 ms Abstände zwischen aufeinanderfolgenden Knoten **innerhalb** des CUDA-Graphen (ein Graph je Stufe, eine einzige Stream-Kette, 1576 Knoten je Runde über alle Stufen), 0,2 ms Kerne außerhalb des Graphen. Kein Host-Sync, keine Eager-Stücke zwischen Graph-Segmenten. |
| Entfernbarer Posten (Obergrenze, je Runde) | Vorbereitung vor Stufe 0: etwa 1,2 bis 1,5 ms bei kurzem Kontext, 0,1 bis 0,25 ms bei 16k [Schätzung, Obergrenze = gemessener Leerlauf nach dem Broadcast]. Übergaben: 0,5 bis 0,85 ms [Schätzung]. Graph-Lücken: 0,4 bis 0,8 ms nur durch Wegfall von Knoten [Schätzung]. Posten sind nicht addierbar ohne Messung (Abschnitt 5). |

## 1. Vorbereitung vor Stufe 0 (Frage 1)

### 1.1 Die Kette von Stufe 4 bis zum ersten Stufe-0-Kern, ein Durchlauf (Capture 1, Runde 10)

Zeiten relativ zum Beginn von `execute_model` auf Stufe 0 (NVTX-Start H0 = 0) [gemessen]:

| Zeit (µs) | Ort | Ereignis |
|---|---|---|
| −1806 | Stufe 4 GPU | D2H-Kopie (40 B, Stream 24) der verifizierten Token; direkt danach läuft auf Stufe 4 der Entwurf (fc + 5 Schichten, Kopf 2, Entwurfs-Sampling) |
| −1806 + 15 | Stufe 4 Host (Async-Output-Thread 2431591) | `cudaEventSynchronize` kehrt 12 bis 23 µs nach dem Kopierende zurück |
| 0 | Stufe 0 Host | `execute_model` beginnt (NVTX), nach der Engine-Core-Rundreise |
| 462 | Stufe 0 Host | erster CUDA-Aufruf, `cudaStreamWaitEvent` (`get_prev_sampled_outputs`: `main_stream.wait_event(slot.event)`, `pp_utils.py:154`) |
| 594 bis 2008 | Stufe 0 Host | 26 Startaufrufe der Vorbereitung (Kerne und Kopien) abgesetzt; die GPU führt sie noch nicht aus |
| 2119 bis 2194 | Stufe 4/0 GPU | zwei NCCL-Broadcasts (`token_packet` int64, dann `combined` int32) von Stufe 4 nach Stufe 0 laufen durch; Ende des zweiten = 2194 |
| 2129 bis 2200 | Stufe 0 GPU | die ersten 15 Vorbereitungskerne laufen sofort hintereinander (je 2 bis 4,5 µs, Abstand 3 bis 5 µs) |
| 2273 bis 3539 | Stufe 0 Host + GPU | die restlichen 14 Kerne und 7 Kopien: jeder Kern startet 8 bis 12 µs nach seinem Startaufruf, die Aufrufe liegen 20 bis 200 µs auseinander (Host-Python dazwischen) |
| 3588 | Stufe 0 Host | `cudaGraphLaunch` (Aufruf dauert 955 µs, siehe 1.4) |
| 3639 | Stufe 0 GPU | erster Graphkern (`_sm70_mhc_prenorm_staging_kernel`) = 50 µs nach dem Start des Graph-Aufrufs |

Die „Vorbereitung vor dem ersten Kern" in `ANALYSE.md` (1,55 ms) ist die GPU-Spanne vom ersten Vorbereitungskern (`_post_update_kernel`) bis zum ersten Graphkern: 1,52 [1,10..2,94] ms in c1 [gemessen].

### 1.2 Worauf wartet Stufe 0? Drei Abhängigkeiten in Reihe [gemessen]

| Größe | Capture 1 (kurz) | Capture 2 (16k) |
|---|---|---|
| H0 bis Ende des zweiten Broadcasts auf Stufe 0 (ms) | 1,98 [−0,53..2,21] | 3,42 [3,15..3,58] |
| H0 bis `cudaGraphLaunch` (Hostpfad, ms) | 3,48 [3,16..4,11] | 3,38 [3,04..4,83] |
| Leerlauf der GPU nach dem Broadcast bis `cudaGraphLaunch` (ms) | 1,50 [1,05..3,68] | −0,04 [−0,49..1,40] |
| Broadcast-Ende bis erster Vorbereitungskern (ms) | 0,02 | 0,00 |
| erster Vorbereitungskern bis erster Graphkern (ms) = „Vorbereitung" der ANALYSE | 1,52 [1,10..2,94] | 0,23 [0,15..1,45] |
| `cudaGraphLaunch`-Aufruf bis erster Graphkern (ms) | 0,05 | 0,27 |
| CUDA-Aufrufe H0 bis `cudaGraphLaunch` (ohne Capture-Abfragen) | 55 | 60 |
| davon nach dem Broadcast-Ende abgesetzt | 23 | 2 |
| Vorbereitungskerne (nicht NCCL) zwischen H0 und Graph | 29 | 29 |

1. **Broadcast der Tokens (GPU-Abhängigkeit):** Der erste Vorbereitungskern startet 0,00 bis 0,02 ms nach dem Ende des zweiten Broadcasts. Der Grund ist `main_stream.wait_event(slot.event)` in `PPHandler.get_prev_sampled_outputs` (`pp_utils.py:136-155`, aufgerufen aus `model_runner.py:1086-1111` `update_pp_decode_requests` ganz am Anfang von `execute_model`). Die Daten (gezogene Token, Annahmezahl, Entwurf der nächsten Runde) kommen von Stufe 4 (`PPHandler.broadcast`, `pp_utils.py:209-245`) und liegen erst nach dem Entwurf vor. Jede Runde hat auf jeder Stufe vier weitere leere `execute_model`-Aufrufe (NVTX `generation_0(0)`, 23 bis 52 µs auf Stufe 0), die den Slot-Ring der Tiefe `pp.world_size` weiterdrehen (`pp_utils.py:125-130`, Code); sie liegen 6,1 bis 6,3 ms nach H0 und sind nicht kritisch.
2. **Scheduler-Rundreise (reiner Hostschritt):** H0 beginnt 2,02 [1,80..4,51] ms (c1) bzw. 1,89 [1,80..2,05] ms (c2) nach dem Ende der D2H-Kopie der verifizierten Token auf Stufe 4 [gemessen]. Dazwischen liegen nach `cudaEventSynchronize` (12 bis 23 µs) nur Hostschritte: AsyncOutput, Engine-Core `future.result()`, `scheduler.update_from_output`, `scheduler.schedule()`, `execute_model(non_block=True)` an fünf Worker (`core.py:623-900`, Code). Wie sich die 2 ms verteilen, ist aus nsys **[offen]**; der Engine-Core kann es selbst loggen (`observability_config.runtime_trace.async_cpu`, Felder `schedule_ms`, `update_ms`, `wait_ms`, `execute_submit_ms` in `core.py:~650-662` und `~873-895`).
3. **Eingabevorbereitung auf dem Host (Hostgebunden):** siehe 1.3.

Die Kette vom Ende der D2H-Kopie bis zum `cudaGraphLaunch` der Stufe 0 ist 5,50 [5,04..7,67] ms (c1) bzw. 5,26 [4,84..6,73] ms (c2) lang [gemessen]. Parallel läuft auf Stufe 4 die GPU-Kette D2H-Kopie bis Broadcast-Ende: 3,99 [3,97..4,02] ms (c1) bzw. 5,30 [5,10..5,44] ms (c2). Der Leerlauf ist die Differenz, soweit positiv:

| Kette ab D2H-Kopie (ms, weniger ist besser) | c1 | c2 |
|---|---|---|
| Hostkette bis `cudaGraphLaunch` Stufe 0 (2,0 Scheduler + 3,4 Vorbereitung) | 5,50 | 5,26 |
| GPU-Kette Stufe 4 bis Broadcast-Ende (Entwurf + Kopf 2 + Sampling) | 3,99 | 5,30 |
| Leerlauf der GPU (Hostkette länger) | 1,50 | 0,11 (Mittel aus max(0, ·)) |

Das erklärt die zweite Frage vollständig: In c2 ist die Entwurfsstufe 1,31 ms länger (`ANALYSE.md` Abschnitt 8: Entwurf 2,68 auf 3,99 ms), die Hostkette bleibt gleich, und beide Ketten sind fast gleich lang. Bei kurzem Kontext ist die Hostkette der begrenzende Teil; bei 16k sind es beide nahezu gleich [abgeleitet].

### 1.3 Was der Host in den 3,4 ms tut (Zuordnung Kern zu Quelltext, [Code])

Jeder Vorbereitungskern ist aus dem Kernnamen einer Stelle im Quelltext zuordenbar; die Zeiten sind Aufrufbeginn relativ zu H0 in Runde 10 (c1):

| H0 + µs | Kerne und Kopien | Quelltext |
|---|---|---|
| 0 bis 462 | Python vor dem ersten CUDA-Aufruf (`_wait_pp_sends`, `annotate_profile`, `mixed_prefill_timer.begin`) | `gpu_worker.py:1780-1870`, `model_runner.py:1572-1585` |
| 594 bis 908 | `_post_update_kernel`, `_scatter_num_accepted_kernel`, Kopie 4 B, `_postprocess_mamba_align_kernel`, `_scatter_draft_tokens_kernel` (Übernahme der Tokens der letzten Runde) | `model_runner.py:1086-1111`; `input_batch.py:461`; `model_states/mamba_hybrid.py:708`; `mamba_align.py:255`; `pp_utils.py:92-105` |
| 1032 bis 1457 | H2D-Kopien 4 B, 8 B, 8 B; `_expand_idx_mapping_kernel`, `_prepare_pos_seq_lens_kernel`, `_combine_sampled_and_draft_tokens_kernel` | `model_runner.py:1179-1352` (`prepare_inputs`); `input_batch.py:259,317,594` |
| 1518 bis 1650 | `_gather_block_tables_kernel`, `_compute_slot_mappings_kernel` | `model_runner.py:1382-1405` (`prepare_attn`); `block_table.py:236,276` |
| 1711 bis 1756 | `preprocess_mamba_align_fused_kernel`, `_precopy_mamba_align_kernel` | `mamba_hybrid.py:338` (`preprocess_state`); `mamba_align.py:149,197` |
| 1896 bis 2165 | `vectorized/unrolled/index_elementwise`, Kopie 4 B und 1 B H2D (Eager-Torch: `new_ones`, indizierte Zuweisung, `.item()`-freie Hilfsgrößen) | `mamba_hybrid.py:436-500` (`prepare_attn` des Modellzustands) |
| 2273 bis 3017 | `elementwise_kernel_with_index`, `_dflash2_gdn_group_metadata_kernel`, mehrere Eager-Elementwise, `_compressed_slot_mapping_kernel`, `_prepare_uniform_decode_kernel`, Kopie 41 300 B | `gdn_attn.py:2120-2170`; `mla/compressor_utils.py:9,75`; `mla/indexer.py:36,576`; Builder-Schleifen in `mamba_hybrid.py:436-650` |
| 3137 bis 3539 | Eager-Elementwise, `indexSelectSmallIndex` (Einbettung, 17,9 µs), Kopie 40 960 B (Eingabe-Einbettung in den Graph-Eingangspuffer) | `model_runner.py:1756-1760` (`prepare_inputs` des Modellzustands), Offload-Einbettung |
| 3588 | `cudaGraphLaunch` | `cudagraph_utils.py:513-526`, `875-888`; `model_runner.py:1822` |

Die Kerne selbst sind nur 2 bis 5 µs lang (29 Kerne zusammen etwa 0,1 ms GPU-Zeit); die Wartezeit der GPU ist also ausschließlich das Absetzen durch den Host [abgeleitet]. Im Aufrufbild sieht man das: zwischen zwei Startaufrufen liegen im Mittel etwa 60 bis 100 µs (Python plus nsys-Overhead), der Kern läuft 8 bis 12 µs nach seinem Startaufruf [gemessen].

Welche Teile nach dem Broadcast noch auf dem kritischen Pfad liegen, hängt vom Kontext ab: In c1 sind 32 der 55 Aufrufe vor dem Broadcast-Ende abgesetzt (dieser Teil ist versteckt), 23 danach (kritisch, 1,50 ms); in c2 sind es 58 gegen 2 [gemessen]. Die Aufrufe nach dem Broadcast sind genau die Modellzustand-Metadaten (GDN-, Kompressor-, Indexer-Builder) und die Eingabe-Einbettung, also die Stellen 2273 bis 3539 der Tabelle.

### 1.4 Der `cudaGraphLaunch` selbst

Der Aufruf dauert auf Stufe 0 1,00 [0,89..1,20] ms (450 Kerne im Graph), auf Stufen 1 bis 4 0,52/0,49/0,42/0,72 ms [gemessen]. Er liegt **nicht** auf dem kritischen Pfad, solange die GPU schneller verbraucht als der Aufruf nachliefert: Der erste Graphkern startet 0,05 ms nach dem Aufrufbeginn (c1), und die Lücken innerhalb des Graphen sind kurz (Abschnitt 3). In c2 sind es 0,27 ms, davon 0,23 ms Leerlauf nach dem Broadcast, der Rest liegt im Aufruf [abgeleitet].

### 1.5 Was sich ändern ließe (mit Schätzung)

| Hebel | Stelle | Wirkung c1 (ms je Runde, [Schätzung]) | Wirkung c2 |
|---|---|---|---|
| (a) Die 29 Kerne und 12 Kopien der Vorbereitung zu einem oder wenigen Kernen bündeln: ein H2D-Stagingpuffer mit allen Hostwerten (`idx_mapping`, `query_start_loc`, `cu_num_logits`, `num_decode_draft_tokens`, Blocktabelle), ein persistenter „Decode-Schritt-Vorbereiter" für die Positionen, Sequenzlängen, Slot-Abbildungen, GDN-/Indexer-/Kompressor-Metadaten, `last_sampled_tokens`-Übernahme. | `model_runner.py:1086-1111, 1179-1405`; `mamba_hybrid.py:338-650`; `gdn_attn.py:2120-2170`; `mla/indexer.py:576`; `mla/compressor_utils.py:75` | bis 1,2 (Leerlauf 1,50 abzüglich der Reste des Hostpfads, der nur noch wenige Aufrufe enthält) | 0,1 bis 0,2 (Leerlauf 0,11 bis 0,23; danach begrenzt die Entwurfskette) |
| (b) Die Scheduler-Rundreise (2,0 ms) verkürzen oder überlappen. Erst messen: `runtime_trace.async_cpu` aktivieren, dann Beitrag von `update_from_output`, `schedule`, Dispatch an fünf Worker trennen. | `core.py:623-900` | proportional, solange die Hostkette länger als die GPU-Kette ist: höchstens 1,5 (die Differenz der Ketten) | höchstens 0,1 |
| (c) Die GPU-Kette nach der D2H-Kopie verkürzen (Entwurf, Kopf 2) | Stufe 4 | in c1 kein Gewinn, solange die Hostkette länger ist | in c2 voll wirksam, wenn zugleich (a)/(b) die Hostkette unter die GPU-Kette bringen |

Die Hebel (a) bis (c) wirken nur zusammen: Die Rundenzeit richtet sich nach dem Längeren von Hostkette und GPU-Kette. Es fehlt eine Messung ohne nsys (Host-Wanduhr je Schritt), die zeigt, ob der Hostpfad in Produktion kürzer ist als unter nsys [offen].

## 2. Übergaben (Frage 2)

### 2.1 Was gesendet wird [Code + gemessen]

Je Übergabe sendet `gpu_worker.py:1956` (`isend_tensor_dict`, `parallel_state.py:1331-1380`) **ein Metadatenobjekt über die CPU-Gruppe** (`send_object`) und danach **jeden Tensor einzeln** mit `torch.distributed.isend` auf die NCCL-Gruppe. Jeder `isend` ist eine eigene NCCL-Gruppe und damit ein eigener `ncclDevKernel_SendRecv` auf dem NCCL-Stream (Stream 44 bzw. 52). Die Tensoren des GLM-5.3-Zustands sind (`vllm/models/glm5next/nvidia/model.py:1130-1145`, ohne `glm_pp_mhc_materialize`) bei 5 Zeilen:

| Tensor | Form | Bytes | Quelle der Größe |
|---|---|---|---|
| `hidden_states` | 5 x 4096 fp16 | 40 960 | gemessen, Kopie 40 960 B auf dem Empfänger; Code |
| `residual` (mHC-Ströme) | 5 x 4 x 4096 fp16 | 163 840 | gemessen, Kopie 163 840 B |
| `post` | 5 x 4 x 1 fp32 | 80 | gemessen, Kopie 80 B |
| `comb` | 5 x 4 x 4 fp32 | 320 | gemessen, Kopie 320 B |
| Hilfs-Zustand je DFlash2-Schicht (Schichten 5, 14, 24, 33), von jeder Stufe weitergereicht, solange `layer_boundary <= end_layer` | 5 x 4096 fp16 | 40 960 je Stück | Code, gemessen über die Kopien |

Übergang 0 zu 1: 4 + 1 Tensoren = 246 160 B; 1 zu 2: 6 Tensoren = 287 120 B; 2 zu 3: 7 = 328 080 B; 3 zu 4: 8 = 369 040 B [abgeleitet]. Die Zahl der `SendRecv`-Kerne je Runde passt (5, 6, 7, 8) [gemessen]. Die Übergabe von Stufe 4 nach Stufe 0 ist kein `SendRecv`, sondern der Broadcast der Token (`pp_utils.py:209-245`, `[1,9]` int64 = 72 B und `[2,1]` int32 = 8 B, abgeleitet aus Code).

Der Kern `ncclDevKernel_SendRecv` trägt keine Größenangabe in den Kernelparametern (nur Gitter 1x1, Block 640); die Größen stammen aus den passenden Empfängerkopien und dem Code.

NCCL läuft mit `NCCL_P2P_DISABLE=1` (Eintrag in `/home/mp/.config/llama-swap/config.yaml`, GLM-Eintrag), also über Shared Memory und Proxy-Thread statt direktem Peer-Zugriff. Das erklärt Latenzen von 20 bis 60 µs je Nachricht schon bei 41 kB [vermutet, wegen der gemessenen Dauern].

### 2.2 Zahlen je Übergabe (Capture 1, Mittel über 38 Runden; Capture 2 ist gleich bis auf ±2 µs) [gemessen]

Zeiten ab dem Ende des letzten Graphkerns der Senderstufe (µs, weniger ist besser):

| Größe | 0 zu 1 | 1 zu 2 | 2 zu 3 | 3 zu 4 |
|---|---|---|---|---|
| Übergabezeit gesamt (letzter Graphkern Sender bis erster Graphkern Empfänger) | 289,6 | 398,6 | 409,6 | 356,7 |
| Zahl `SendRecv`-Kerne (Sender = Empfänger) | 5 | 6 | 7 | 8 |
| erster Sendekern beginnt | 11,8 | 21,1 | 29,2 | 29,4 |
| erster Empfangskern endet (Eintreffen von `hidden_states`) | 57,9 | 73,9 | 82,7 | 79,0 |
| letzter Sendekern endet | 146,5 | 196,0 | 282,9 | 258,1 |
| letzter Empfangskern endet | 219,9 | 290,7 | 311,1 | 331,9 |
| Letzter Empfang bis erster Graphkern Empfänger (D2D-Kopien in die Eingangspuffer + Start) | 69,7 | 107,9 | 98,5 | 24,8 |
| Anteil NCCL (letzter Empfang) an Übergabezeit (%) | 76 | 73 | 76 | 93 |

Dauer der Sendekerne in der Reihenfolge `hidden_states`, `residual`, `post`, `comb`, Hilfszustände (µs): Stufe 0: 27,7 / 63,8 / 8,1 / 7,5 / 22,9; Stufe 1: 30,5 / 64,3 / 10,5 / 9,5 / 24,0 / 23,6; Stufe 2: 37,5 / 71,2 / 14,2 / 13,7 / 30,0 / 29,8 / 30,2; Stufe 3: 30,4 / 65,2 / 10,4 / 9,4 / 23,9 / 23,5 / 23,8 / 23,6. Die zugehörigen Empfangskerne enden 18 µs (erste Nachricht), später etwa 60 µs nach dem Sendekern der großen `residual`-Nachricht, die Reihe holt ab dort auf (letzter Empfang minus letzter Sendekern 73, 95, 28, 74 µs).

Empfängerkopien (D2D, `model_runner.py:1776-1783`: je Schlüssel `v[:n].copy_(intermediate_tensors.tensors[k][:n])`) [gemessen]: auf V100 je 8,8 bis 11,7 µs, Abstand 13,6 bis 18,4 µs (5, 6, 7 Kopien: 68, 103, 93 µs); auf der RTX der Stufe 4 je 2,3 bis 2,5 µs, Abstand 3 µs (8 Kopien: 24 µs).

Befund [abgeleitet]: **Latenz, nicht Bandbreite**. Eine 41-kB-Nachricht braucht auf dem Sendekern 22 bis 37 µs (etwa 1,5 GB/s), die 164-kB-Nachricht 64 bis 71 µs (2,5 GB/s); beides weit unter jeder PCIe- oder Shared-Memory-Bandbreite. Die Übergabe ist die Summe von 5 bis 8 seriellen Einzelnachrichten (je 8 bis 71 µs Kern, dazu 1 bis 3 µs Abstand und Empfangsversatz) plus fünf bis acht einzelne Kopien auf dem Empfänger. Die Übergabe 4 zu 0 ist der Broadcast: vom Ende des letzten Sampling-Kerns auf Stufe 4 bis zum Ende des zweiten Broadcasts auf Stufe 0 33 µs (eine Runde gemessen; `ANALYSE.md`: 0,03 ms).

### 2.3 Was sich ändern ließe (Schätzung, nicht gemessen)

Untergrenze einer Übergabe, wenn nur noch eine 164-kB-Nachricht gesendet wird und der Empfänger direkt in den Graph-Eingangspuffer schreibt: Sendebeginn etwa 12 µs, Sendekern 64 µs, Empfang etwa 40 bis 60 µs danach, ein Kopierschritt 9 bis 14 µs und 4 µs Start: etwa 140 bis 155 µs gegen gemessen 290 bis 410 µs. Das wären je Übergabe 135, 245, 255, 205 µs, zusammen **0,5 bis 0,85 ms je Runde** (untere Grenze, wenn die Hilfszustände weiterhin mitlaufen; obere, wenn sie ganz vom kritischen Pfad verschwinden) [Schätzung].

| Maßnahme | Stelle | Wirkung (µs je Runde, [Schätzung]) |
|---|---|---|
| Hilfszustände (Schichten 5, 14, 24, 33) nicht Schritt für Schritt über jede Stufe weiterreichen, sondern direkt von der besitzenden Stufe an Stufe 4 senden (eigener Strom, erst vor dem Entwurf gebraucht) | `model.py:1085-1145` (`tensors[...] = aux_hidden_states`, `_incoming_aux_hidden_states` 1001-1010); Senden `gpu_worker.py:1956` | 1 + 2 + 3 weniger Nachrichten auf den Übergaben 1 zu 2, 2 zu 3, 3 zu 4, je etwa 25 µs Sendekern plus 13 µs Kopie (V100) bzw. 3 µs (RTX): etwa 200 |
| Vorhandenen Schalter `glm_pp_mhc_materialize` (Standard aus, `execution_policy.py:444`, `model.py:916-922`) prüfen: dann gehen `hidden_states`, `residual`, `post`, `comb` als **ein** Tensor `[5,4,4096]` über die Grenze (`hc_post` einmal auf der Senderstufe) | `model.py:1023-1043, 1083-1091, 1117-1131`; `config.py:72` | je Übergabe drei Nachrichten und drei Kopien weniger: etwa 60 bis 90 µs, zusammen etwa 250 bis 350. Vorher auf Gleichheit der Ausgabe und Nebenwirkung auf die fusionierte `post_dot`-Route prüfen |
| Alle Tensoren einer Übergabe in einen flachen Puffer packen (einmaliger `isend`) und beim Empfänger direkt in den Graph-Eingangspuffer empfangen (statt je Schlüssel `copy_`); die Metadatennachricht über die CPU-Gruppe entfällt, die statischen Pfade `isend_tensor_dict_static`/`irecv_tensor_dict_static` (`parallel_state.py:1388-1450`) und `_rank0_pp_recv` (`gpu_worker.py:1636-1760`, Standard aus) bestehen schon, sind aber an PP2/TP4 und `hidden_states` allein gebunden (`gpu_worker.py:1556-1610, 1760-1775`) | `model_runner.py:1776-1783`; `parallel_state.py:1331-1420, 1488-1560`; `gpu_worker.py:1556-1610, 1956` | Empfängerkopien 68, 103, 93, 24 µs entfallen (zusammen 288); Nachrichtenzahl auf 1 |
| Die zwei Broadcasts Stufe 4 zu Stufe 0 (`token_packet`, `combined`) zu einem verbinden | `pp_utils.py:209-245` | etwa 10 µs [Schätzung] |

Die Maßnahmen überlappen sich; die obere Grenze 0,85 ms setzt alle drei großen Maßnahmen voraus.

## 3. Lücken innerhalb der Stufendurchläufe (Frage 3)

### 3.1 Was gemessen ist

Die Lücken stecken überwiegend zwischen den Knoten **innerhalb** des CUDA-Graphen (je Stufe ein FULL-Graph, ein Replay, alle Kerne auf einem Stream, `graphNodeId` gesetzt, Stream 20) [gemessen]. Summe der Abstände zwischen aufeinanderfolgenden Graphkernen je Runde (µs, weniger ist besser):

| Stufe | Knoten je Runde | Summe der Abstände c1 | c2 | mittlerer Abstand (µs) |
|---|---|---|---|---|
| S0 (RTX) | 450 | 428 [393..464] | 430 | 0,95 |
| S1 (V100) | 259 | 324 [289..365] | 321 | 1,25 |
| S2 (V100) | 259 | 435 [388..483] | 429 | 1,68 |
| S3 (V100) | 225 | 273 [254..298] | 276 | 1,21 |
| S4 (RTX, nur Zielmodell-Graph) | 383 | 196 [194..198] | 196 | 0,51 |
| Summe | 1576 | 1655 | 1652 | 1,05 |

Mit Stufen 1 bis 4: 1228 µs im Graphen; dazu kommen etwa 0,2 ms außerhalb des Graphen (Kerne `_post_update_num_computed_tokens_kernel` u. ä. nach dem Graph; auf Stufe 4 sind es 124 µs Lücken im Entwurf, dort liegen die Kerne dicht), zusammen das 1,44 ms der `ANALYSE.md` [abgeleitet]. Stufe 0 hat darüber hinaus die 0,43 ms im Graphen (die ANALYSE zählt sie in die 1,91 ms mit den 1,52 ms Vorbereitung).

Verteilung (c1, alle Stufen, µs je Runde): 1375 von 1576 Abständen sind kürzer als 1 µs und zusammen 567 µs; die rund 200 Abstände von 1 µs und mehr summieren sich zu 1087 µs. Nach Klassen (Abstand vor dem jeweils zweiten Kern der Paarung):

| Klasse | Anzahl je Runde | Summe der Abstände (µs) | Bemerkung |
|---|---|---|---|
| `_mtp_weighted_reduce_kernel` zu `vectorized_elementwise_kernel` (Ende der MoE-Schicht) | 42 | 166 (4,2 bis 4,9 µs je Stück auf S0 bis S3, 2,4 auf S4) | auffällig groß gegen 0,3 bis 0,6 µs typisch |
| `_paged_index_logits_relu_kernel` zu `persistent_topk_kernel` (DSA-Indexer) | 11 (eine je DSA-Schicht) | 128 (12,5 bis 17,5 µs je Stück auf S0 bis S3, 5,0 auf S4) | größter Einzelabstand je Schicht |
| `Kernel2` zu `splitKreduce_kernel` (cuBLAS Split-K, 169 Paare) | 169 | 214 (1,0 bis 2,3 µs) | der Reduce-Kern selbst braucht zusätzlich 624 µs GPU-Zeit je Runde (2,4 bis 3,9 µs je Stück) |
| mHC (`post_dot` zu `pre_norm` u. a.) | 278 | 195 | |
| MoE-Übergänge Up, Aktivierung, Down | 213 | 217 | |
| übrige | 858 | 736 | |

### 3.2 Ursache

Keine Hostsynchronisation, kein Eager-Zwischenstück im Graphen: Der Abstand ist der Knotenwechsel des Graphen auf einem Stream (etwa 0,3 bis 0,6 µs Mindestwert, Mittel 1,05 µs). Die größeren Abstände (4 bis 17 µs) betreffen bestimmte Paarungen, deren Ursache aus nsys **[offen]** ist; Hinweise [vermutet]: Wechsel der Shared-Memory-Konfiguration beim `persistent_topk_kernel` (große dynamische Shared-Memory-Zuteilung) und beim Kern nach dem `_mtp_weighted_reduce_kernel`. Eine Aufteilung nach „piecewise gegen FULL" gibt es hier nicht: Alle Decode-Schritte laufen als FULL-Graph (`model_runner.py:1822`, `cudagraph_utils.py:513-526`; Capturegröße 5, `--compilation-config {"cudagraph_capture_sizes":[5]}`).

### 3.3 Was sich ändern ließe (Schätzung, nicht gemessen)

| Maßnahme | Stelle | Wirkung (µs je Runde, [Schätzung]) |
|---|---|---|
| cuBLAS-Split-K-Paare (`Kernel2` plus `splitKreduce`) ersetzen: GEMMs ohne Split-K oder in einem Kern (Zuordnung zu den Tensoren laut `ANALYSE.md`: KDA f/g/b-Projektionen, Router, DSA `kv_b`, Indexer `wq_b`); die Schalter `glm_cublaslt`, `glm_fused_fg_b`, `glm_exact_kda_gemv` (`execution_policy.py:475-526`) gehören dazu | glm5next-Modell (`nvidia/model.py`, `nvidia/kda.py`, `nvidia/attention.py`) | bis 840 (214 Abstände + 624 Reduce-Kern) als Obergrenze bei vollständigem Wegfall des Reduce; 400 bis 600 realistisch, weil der GEMM-Kern selbst nicht kostenlos wird |
| `_mtp_weighted_reduce_kernel` mit dem folgenden `vectorized_elementwise_kernel` verschmelzen | MoE-Kombination (`skinny`-Pfad) | 166 Abstand plus Dauer des Elementwise-Kerns (1,6 bis 2,6 µs je Stück): etwa 200 bis 270 |
| `persistent_topk_kernel`: den Abstand vor dem Kern klären (Shared-Memory-Konfiguration) | DSA-Indexer (`mla/indexer.py`) | bis 128 [abhängig von der Ursache] |

Untere Grenze der Posten (Mindestabstand 0,3 bis 0,6 µs je Knoten) bleibt: etwa 0,57 ms je Runde bei 1576 Knoten, nur durch Wegfall von Knoten (Fusion) kleiner [abgeleitet].

## 4. Offene Messpunkte, ohne die die Schätzungen nicht gesichert sind

1. Host-Wanduhr je Schritt ohne nsys (`runtime_trace.async_cpu` des Engine-Cores plus Zeitstempel im Worker): zeigt, ob die 3,4 ms bis `cudaGraphLaunch` und die 2,0 ms Scheduler-Rundreise in Produktion kleiner sind (nsys-Overhead).
2. Aufteilung der 2,0 ms zwischen Kopierende und H0: nicht aus nsys lesbar (keine OSRT-/CPU-Daten).
3. Ursache der Abstände von 4 bis 17 µs hinter `_mtp_weighted_reduce_kernel` und vor `persistent_topk_kernel`: Kernel-Profiler (ncu) oder Test mit geänderter Shared-Memory-Konfiguration.
4. Beitrag je Tensor im `SendRecv`-Strang (Latenz je Nachricht über Shared-Memory-Transport): ein Mikrotest mit einem gepackten Puffer gegen fünf Einzelnachrichten.

## 5. Zusammenfassung der Posten mit Obergrenzen (ms je Runde; weniger ist besser, nicht addierbar)

| Posten | Gemessen (c1) | Erreichbar (Obergrenze, [Schätzung]) | Bedingung |
|---|---|---|---|
| Vorbereitung vor Stufe 0 (GPU-Leerlauf nach dem Broadcast) | 1,50 | bis 1,2 (c1) bzw. 0,1 bis 0,2 (c2) | Hostpfad unter die GPU-Kette bringen; bei 16k begrenzt die Entwurfskette |
| Übergaben | 1,46 (letzter Graphkern bis erster Graphkern, vier Übergaben) | 0,5 bis 0,85 | weniger Nachrichten, direkter Empfang in Graphpuffer, Hilfszustände abseits des Pfads |
| Lücken im Graphen | 1,66 (alle Stufen) | 0,4 bis 0,8 | Wegfall von Knoten (Split-K, MoE-Kombination) |
| Summe der drei Obergrenzen | 4,6 | 2,1 bis 2,85 | nicht addierbar bevor je Posten einzeln gemessen ist |

Die zweite Spalte der Übergaben (1,46 ms) ist vom letzten Graphkern der Senderstufe an gemessen und daher um 0,01 bis 0,03 ms je Übergabe größer als die 1,35 ms der `ANALYSE.md`, die ab dem letzten Kern des Durchlaufs rechnet (einschließlich der Nachkerne nach dem Graphen).

## Nachtrag 24.11.2026: Host-Vorbereitung gebündelt und gemessen

Zweig `host-prep-staging` (3b06bda1a GDN-Hostzahlen ohne Torch-Dispatch + Spec-Index bei Bedarf; 508ac9f8f eine Kopie für die
Index-Arrays + Host-Blocktabelle nur für ihren Leser), Gegenprobe am alten Code 14 von 16 neuen Tests rot. A/B `ab_hpo_{glm,dsv4}_2026-11-24a.log`
(T P T P, Env wie Produktion, gleicher Compile-Hash): GLM Runde 53,6 gegen 53,6–53,7 ms, AIfred-Sampling +0,15 %, Prefill gleich, erstes
Token 150k–516k −0,4 bis −0,7 %, Texte tokengleich; DeepSeek gleich. Die unter nsys gemessene GPU-Leerlaufzeit vor Stufe 0 (1,5 ms, Capture 1)
begrenzt die Runde im Betrieb ohne Profiler offenbar nicht. Änderungen als verlustfreie Bereinigung in den Entwicklungszweig
`merge-1cat-2d218ab03` übernommen (fährt mit dem nächsten Bündel), kein eigener Produktionswechsel. Weitere Hostarbeit (Deskriptor,
Scheduler-Rundreise) erst nach einer Hostzeitmessung ohne nsys, die zeigt, dass der Host die Runde begrenzt.
