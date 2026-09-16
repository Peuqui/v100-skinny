# Entwurf: PLE-Überlaufkaskade VRAM → Host → freie GPU → SSD

Stand 2026-09-16: Pakete 1 bis 3 gebaut und abgenommen (Abschnitte 10 bis 12);
offen sind Paket 4 (Loader-Spitze) und Paket 5 (AIfred berechnet das Budget). Umsetzung im Fork
`1Cat-vLLM-work` (work-main). Hintergrund und frühere Entscheidungen:
Projekt-Memory `project_ple_tier_cascade`.

## 1. Anlass

Flash-Next (nvidia NVFP4, TP2 × PP2) legt beim Kaltstart 12 GiB der PLE-Tabelle
in gepinnten Host-RAM. Gemessen am 15.09. (`coldstart_mem.csv` im Scratchpad):

| | Wert |
|---|---|
| Tabelle gesamt | 320.001.536 Zeilen × 160 B = 47,7 GiB, TP-geteilt |
| je Rang der Stufe 0 (RTX 8000) | 17,84 GiB VRAM + 6,0 GiB gepinnt |
| gepinnt gesamt | 12 GiB, nicht auslagerbar |
| nach dem Start | MemAvailable 3,8 GiB, Swap 19,8 GiB |
| Loader-Spitze der Worker | 6,8 GiB anonym, vorübergehend |

Der Rechner hat 30 GiB RAM. Der Rest der Maschine wird ausgelagert und bleibt
zäh. Künftige Modelle mit PLE bringen größere Tabellen mit, die weder in den
VRAM der Rechenkarten noch in den Host passen.

Ziel: Die Tabelle verteilt sich nach gemessenen Budgets über vier Stufen, der
Host-RAM behält eine harte Reserve, und die Ausgabe bleibt bitgleich zu heute.

## 2. Stand im Code

- `plan_ple_placement` (`models/qwen4_exp/common/ple.py`) kennt zwei Stufen:
  VRAM und Host. Die Zeilen sind hash-adressiert, die Aufteilung ist eine reine
  Kapazitätsfrage; jede Stufe wird bei jedem Schritt anteilig getroffen.
- `Qwen4ExpPinnedHostEmbedding` (`nvidia/ple_layer.py`) hält je TP-Rang zwei
  zusammenhängende Bereiche der eigenen Tabellenhälfte. Der Gather ist eine
  undurchsichtige Custom-Op (`qwen4_exp_ple_pinned_gather`); die Host-Hälfte
  liest die Rechenkarte direkt über eine UVA-Sicht. Beide Gathers laufen immer,
  zusammengeführt per `where`, ohne Host-Sync. Das läuft im vollen CUDA-Graphen.
- Produktion läuft mit `cudagraph_mode=FULL_AND_PIECEWISE`: Decode-Schritte
  sind volle Graphen. Dort gibt es keinen Punkt, an dem Python eingreifen kann.
- Ein gesetztes `VLLM_QWEN4EXP_PLE_HOST_GIB` überspringt die Host-Kappung
  (`_resolve_host_budget`); nur der Automatikmodus kappt auf den verfügbaren RAM.
- Vorhanden und wiederverwendbar: der **PLE-Offload-Worker** (`v1/ple_offload/`).
  Ein eigener Prozess hält PLE-Tabellen, liest die Eingaben der Runner über
  geteilten Speicher (MRV2 mit asynchronem D2H und Events), schreibt Ergebnisse
  per CUDA-IPC in Ausgabepuffer auf den Rechenkarten und gibt per
  `cuStreamWriteValue32` frei. Die Rechenkarte wartet mit
  `cuStreamWaitValue32` als Custom-Op (`ple_offload_wait`) — **das ist im
  vollen CUDA-Graphen aufnehmbar**. Der Worker kann Zeilen per mmap aus dem
  Checkpoint lesen (`VLLM_PLE_DISK_OFFLOAD`, `MADV_RANDOM`, Thread-Pool) und
  kennt Spekulation (`_clamp_input_ids`). Bisher bedient er nur ganze Tabellen
  oder die Hybrid-Spur ohne MTP.
- Das MTP-Drafter-Modell schaltet PLE ab. PLE läuft nur im Verifizierer-Schritt
  des Hauptmodells (k+1 Token).

## 3. Stufen und Budgets

Reihenfolge wie von Peuqui vorgegeben:

| Stufe | Ort | Zugriff im Decode | Budget |
|---|---|---|---|
| 1 VRAM | Rechenkarte des Rangs | im Graphen, direkt | automatisch wie heute: nutzbarer VRAM − Gewichte − KV für `max_model_len` − Reserve |
| 2 Host | gepinnter RAM | im Graphen, UVA | `VLLM_QWEN4EXP_PLE_HOST_GIB` je Rang oder automatisch; **immer** dynamisch gekappt auf (MemAvailable zum Platzierungszeitpunkt − `VLLM_QWEN4EXP_PLE_HOST_RESERVE_GIB`) / Ränge; Überschreitung eines expliziten Werts ist ein Startfehler, keine stille Kürzung |
| 3 freie GPU | Speicherkarte, z. B. GPU 4 | Offload-Worker, Warten im Graphen | `VLLM_QWEN4EXP_PLE_STORE_DEVICE` (sichtbarer Index) + `VLLM_QWEN4EXP_PLE_STORE_GIB` (gesamt, gleichmäßig auf die TP-Ränge) |
| 4 SSD | mmap auf die Checkpoint-Dateien | Offload-Worker, Warten im Graphen | Rest; nur mit `VLLM_QWEN4EXP_PLE_DISK=1`, sonst Startfehler „Tabelle passt nicht“ |

Host-Budget und Host-Reserve bleiben konfigurierbar und werden zur Laufzeit
gegen den tatsächlich verfügbaren RAM geprüft. Beides existiert schon
(`_ple_host_budget_bytes`, `_ple_host_reserve_bytes`, `cap_host_budget_bytes`,
Reserve-Vorgabe ein Viertel des RAMs); der Umbau wendet die vorhandene Kappung
auch auf explizite Werte an, statt sie zu umgehen. Wie viel der Host auf dem
Mini trägt, ist Einstellung, z. B. 10 GiB gesamt, der Rest geht auf GPU 4.
Die gemessene Loader-Spitze (6,8 GiB) ist ein Anhaltspunkt für die Reserve,
keine eingebaute Konstante.

Das Budget für Stufe 3 berechnet AIfred beim Erzeugen des llama-swap-Eintrags:
freier Speicher der Karte minus eingebrannter Bedarf von VLM und TTS. vLLM
selbst misst nicht, was andere Prozesse später auf der Karte brauchen.

Jede Stufe ist ein zusammenhängender Bereich der TP-lokalen Zeilen. Der Planer
liefert vier Grenzen statt einer; `copy_ple_embedding_shard_split_` wird von
zwei auf vier Ziele verallgemeinert.

## 4. Ablauf pro Schritt

1. Runner startet den Schritt. Wie heute beim Offload-Pfad: TP-Rang 0 kopiert
   `input_ids`, `query_start_loc`, `ngram_context` asynchron in die geteilten
   Puffer und stellt die Anfrage in die Warteschlange.
2. Der Offload-Worker berechnet die N-Gramm-IDs einmal (Eingaben sind über die
   TP-Ränge gleich), teilt sie je Rang in dessen Stufe-3- und Stufe-4-Bereich,
   holt die Zeilen
   - Stufe 3: `index_select` auf der Speicherkarte, D2H in gepinnten Puffer,
   - Stufe 4: mmap-Lesen im Thread-Pool,
   und schreibt sie positionsgenau in den Ausgabepuffer jedes Rangs; danach
   Freigabe über das Stream-Flag.
3. Auf der Rechenkarte laufen im Graphen: Gather Stufe 1, Gather Stufe 2,
   Warten auf das Flag, Zusammenführen der drei Quellen per `where` nach
   Zeilenbereich, Dequantisierung wie heute. Keine neue Arithmetik, daher
   bitgleich.
4. Nach dem Forward setzt der Runner das Flag zurück (`release_outputs`).

Ist Stufe 3 und 4 leer, entfällt Schritt 2 und das Warten ganz; der Pfad ist
dann identisch mit heute.

Datenmenge im Decode: k+1 = 5 Token × 16 Köpfe × 160 B = 12,8 KB je Rang, davon
bei GPU 4 mit 25 % Anteil rund 3 KB. Die Latenz besteht fast nur aus
Prozessübergang, Stream-Synchronisation und zwei PCIe-Hops (GPU 4 hängt am
USB4-Tunnel). Schätzung 1–3 ms je Schritt, **nicht gemessen**.

## 5. Laden

- Stufe 1 und 2 laden wie heute im Rang.
- Stufe 3 lädt der Offload-Worker direkt vom mmap auf die Speicherkarte, ohne
  Zwischenkopie im Host.
- Stufe 4 wird nicht geladen; die Checkpoint-Tensoren bleiben gemappt
  (Page-Cache, freigebbar).
- Die Ränge legen für Stufe 3 und 4 keinen Speicher an.
- Die Loader-Spitze von 6,8 GiB anonym ist davon getrennt. Ursache noch nicht
  eingegrenzt; eigenes Paket, siehe 8.

## 6. Konfiguration auf dem Mini (Ziel)

Im llama-swap-Eintrag von Flash-Next:

    CUDA_VISIBLE_DEVICES=0,2,1,3,4        # GPU 4 als Index 4, Ränge unverändert
    VLLM_QWEN4EXP_PLE_HOST_GIB=5          # Beispiel: 10 GiB gesamt über zwei Ränge
    VLLM_QWEN4EXP_PLE_HOST_RESERVE_GIB=<Einstellung>
    VLLM_QWEN4EXP_PLE_STORE_DEVICE=4
    VLLM_QWEN4EXP_PLE_STORE_GIB=<von AIfred berechnet>

Die Namen sind Vorschläge.

## 7. Risiken und offene Punkte

1. **Offload-Worker mit MTP + PP2 + asynchronem Scheduling ist ungetestet.**
   Er lief bisher in der Hybrid-Spur ohne MTP. Der erste Schritt von Paket 1 ist
   ein Durchstich mit leerer Stufe 3 und 4, der nur den Mechanismus in unserer
   Topologie beweist.
2. **Latenz des Wartens.** Die PLE-Schicht liegt sehr früh im Modell (Schicht 1).
   Der Worker muss fertig sein, bevor die Rechenkarte dort ankommt, sonst
   verlängert sich jeder Schritt. Messen gegen heutige 46–50 tok/s.
3. **USB4-Strecke von GPU 4.** Gen3 x4 über USB4-Tunnel; Latenz je Hop
   unbekannt, im Durchstich messen.
4. **SSD im Prefill.** Bei 40.000 Prompt-Token und nennenswertem SSD-Anteil
   sind es hunderttausende zufällige Lesezugriffe; bei kaltem Page-Cache
   Sekunden bis Minuten. Stufe 4 ist ein Generalitäts-Feature; auf dem Mini
   planmäßig leer.
5. **GPU 4 teilt sich mit VLM und TTS.** Holen die sich später mehr Speicher
   als eingebrannt, gibt es dort einen OOM. Das Budget muss deren Spitzenbedarf
   abziehen, nicht den Leerlauf.
6. **Langsamste Stufe bestimmt die Schrittzeit** (Hash-Adressierung). Eine
   heiße Stufe gibt es nicht.
7. **Dummy- und Capture-Läufe** müssen das Flag weiter lokal setzen
   (`signal_dummy_outputs`), auch wenn nur Teile der Tabelle ausgelagert sind.

## 8. Pakete und Abnahme

**Paket 1 — Durchstich Mechanismus**
- Offload-Worker in unserer Topologie (MTP k=4, PP2, async) mit leerer
  Stufe 3/4 hochfahren; Warten im vollen Graphen, Zusammenführen.
- Abnahme: Text bitgleich zu heute (Hash über die ersten 260 Token),
  tok/s unverändert innerhalb der Messstreuung.

**Paket 2 — Planer vierstufig + Stufe 3**
- `PLEPlacement` mit vier Bereichen, Host-Kappung immer aktiv, explizite
  Überschreitung als Fehler, neue Umgebungsvariablen.
- Offload-Worker hält Stufe 3 auf der Speicherkarte, lädt direkt vom mmap.
- Unit-Tests: Planer-Grenzen, Kopie in vier Ziele, Zusammenführung bitgleich
  (Muster `scratchpad/ple_split_unit.py`).
- Abnahme auf dem Mini: Host-Anteil wie eingestellt, Rest auf GPU 4;
  MemAvailable nach dem Start, Swap-Zuwachs beim Kaltstart, tok/s Decode,
  Prefill-Zeit, Kuanda. Mehrere Host-Einstellungen vergleichen.

**Paket 3 — Stufe 4 SSD**
- mmap-Leser des Offload-Workers an die Stufengrenzen anbinden.
- Abnahme mit künstlich kleinem Budget für Stufe 3, damit die SSD getroffen
  wird; Latenz Decode und Prefill dokumentieren.

**Paket 4 — Loader-Spitze** eingrenzen (6,8 GiB anonym beim Laden) und,
wenn möglich, entschärfen.

**Paket 5 — AIfred**: Budget für Stufe 3 aus den eingebrannten VLM-/TTS-Werten
berechnen und in die llama-swap-Einträge schreiben; Kalibration neu.

## 9. Entschieden (Peuqui 15.09.)

1. Reihenfolge VRAM → Host → freie GPU → SSD. Der Host trägt, was eingestellt
   ist (z. B. 10 GiB), der Rest geht weiter.
2. SSOT: bestehende Strukturen übernehmen und anpassen — Planer
   `plan_ple_placement`, Host-Budget/-Reserve-Funktionen, PLE-Offload-Worker mit
   mmap-Leser und Stream-Semaphore. Keine Parallelstrukturen.
3. Budgets und Reserven sind konfigurierbar und werden dynamisch gegen den
   gemessenen Speicher geprüft; nichts wird fest eingebaut.
4. Keine Vorab-Ankündigung bei 1Cat. Gebaut und abgenommen wird im Fork,
   danach als PR angeboten.

## 10. Paket 1 — Umsetzung (begonnen 2026-09-15 nachts)

Ziel unverändert: den Mechanismus in der Produktions-Topologie beweisen
(MTP k=4, TP2 × PP2, asynchrones Scheduling, `FULL_AND_PIECEWISE`), Stufe 3
und 4 leer, Text bitgleich. Entscheidungen beim Bauen:

1. **Ein Schalter:** `VLLM_QWEN4EXP_PLE_STORE_DEVICE=<sichtbarer Index>`.
   Gesetzt → Kaskade. Die Config leitet daraus `VLLM_PLE_CPU_OFFLOAD=1` und
   den IPC-Pfad ab (dasselbe Muster wie die Hybrid-Vorgabe
   `_apply_sm70_qwen38_hybrid_ple_defaults`), damit Executor, Worker-Spawn
   und Connector unverändert greifen. Hybrid-Spur oder `VLLM_PLE_DISK_OFFLOAD`
   gleichzeitig → Startfehler; Modell ohne PLE → Startfehler. Der Schalter ist
   in `envs.py` deklariert und geht damit automatisch in den
   Compile-Cache-Schlüssel ein (`compile_factors`).
2. **Rechenkarte:** `Qwen4ExpNGramEmbedding` behält Konstruktor und Tabellen
   (neuer Basisklassen-Haken `offload_keeps_local_tables()`), gathert Stufe 1
   und 2 wie heute, wartet dann per `ple_offload_wait` auf den Worker und
   führt VOR dem TP-All-Reduce per `where` nach rang-lokalem Zeilenindex
   zusammen (`local_id >= device_rows + host_rows`). Die Worker-Bytes werden
   mit demselben E4M3-Kernel und derselben Skala in fp16 gebracht wie die
   lokalen Gathers — gleiche Arithmetik, daher bitgleich. In Paket 1 ist die
   Maske immer leer, der Graph enthält aber Warten, Dequant und Merge.
3. **Worker:** hält die Checkpoint-Shards nur als mmap (kein anonymer RAM,
   wie die Disk-Spur), berechnet je Schritt die N-Gramm-IDs auf der CPU,
   füllt seinen Ausgabepuffer mit Nullen, kopiert ihn über den Copy-Stream
   und signalisiert. Die Rang-Geometrie (`tp_start`, `tp_end`, `local_rows`)
   kommt mit der Registrierung (`PLERemotePlacement`, neues Feld
   `remote_placements` in `PleOffloadRegistration`). Meldet ein Rang Zeilen,
   die er nicht selbst hält, bricht der Worker mit „Store-Stufe noch nicht
   gebaut“ ab (kommt in Paket 2). Beim Binden öffnet er die Speicherkarte
   (`mem_get_info`) und loggt freien Speicher — Nachweis, dass GPU 4 aus dem
   Worker erreichbar ist.
4. **PP-Fix:** Der Connector wird nur auf Rängen aufgebaut, deren Modell
   einen `PleOffloadLayer` hat (Stufe 0). Bisher hätte Stufe 1 in
   `_setup_layers` an „kein PleOffloadLayer“ abgebrochen (Journal
   TURING-COEXISTENCE 1604) — der Worker lief hier nie unter PP2.
5. **Unangetastet:** der Produktionspfad ohne Schalter. Jeder neue Zweig
   hängt an `_is_cpu_offloaded` (nur mit Connector gesetzt) oder am Schalter.

**Test-Boot (nach Ansage):** Flash-Next-Eintrag mit
`CUDA_VISIBLE_DEVICES=0,2,1,3,4` und `VLLM_QWEN4EXP_PLE_STORE_DEVICE=4`,
sonst unverändert. Beleg: Boot-Log (Worker READY, zwei Registrierungen,
Placements, Speicherkarte), Text-Hash 260 Token gegen Referenz, tok/s,
Schrittzeit. Danach Kontroll-Boot ohne Schalter, bitgleich.

### Stand 2026-09-15 spät: Code fertig und getestet

- Umgesetzt wie oben (Punkte 1–5), uncommittet im Branch
  `qwen4exp-ple-tier-cascade`. Unit-Tests für Merge (bitgleich, auch im
  CUDA-Graphen mit Replay), Worker-Bindung, Registrierung, PP-Rang ohne
  PLE-Layer und Config-Vertrag; 129 Tests in beiden Reihenfolgen grün,
  ruff/typos/mypy sauber.
- Nebenbefund in den Tests: `monkeypatch.setattr` auf `vllm.envs` hinterlässt
  echte Modulattribute, die `envs.__getattr__` für spätere Tests überdecken;
  `delenv` auf eine fehlende Variable merkt sich nichts. Behoben über einen
  Helper `set_lazy_env` in `tests/utils.py`, genutzt von den neuen Tests und
  den zwei Hybrid-PLE-Tests, die daran scheiterten.
- Referenz gegen die Produktion vor der Änderung (Boot 17:39, work-main):
  `handover/2026-09-15/ref_prod.json`, Sonde `ple_probe.py` (drei Prompts,
  greedy, 260 Token, Prompt 1 doppelt, Wiederholung gleich), 57–65 tok/s.
- Boot-Skript `handover/2026-09-15/ple_cascade_boot.sh OUTDIR`: entlädt die
  Flash-Next-Variante in llama-swap, bootet den Eintrag exakt nach (Port 8093)
  mit nur `CUDA_VISIBLE_DEVICES=0,2,1,3,4` und
  `VLLM_QWEN4EXP_PLE_STORE_DEVICE=4`, Sonde + Log-Belege, Stopp über
  `vllm-swap-stop`, danach Kontroll-Boot über llama-swap mit unverändertem
  Eintrag und Vergleich beider Sonden gegen die Referenz. Erwartung: neuer
  Env-Wert → neuer Compile-Schlüssel → kalter Compile beim Kaskaden-Boot.

### Test-Boots 2026-09-15 abends

| Lauf | Ergebnis |
|---|---|
| Boot 1 Kaskade | Abbruch: eigener Config-Check lehnte die modellose `VllmConfig()` des Offload-Workers ab. Behoben (Check nur mit Modell), Regressionstest |
| Boot 1 Kontrolle | unveränderter Eintrag, 4/4 hashgleich zur Referenz |
| Boot 2 Kaskade | läuft durch, 0 Tracebacks; 3/4 hashgleich, p2 weicht ab; 2–5 % langsamer |
| Boot 2 Kontrolle | 4/4 hashgleich |
| Boot 3 Kaskade (Diagnose) | p2 weicht bei Zeichen 306 / Token 76 ab, Beinahe-Gleichstand „ to“ gegen „:“ (Abstand 0,016, gekippt); Logprobs driften ab Token 0 (mittel 0,003, max. 0,043) |

Belegt im Boot-Log: Worker mit 128 mmap-Shards, RSS 0,37 GiB; beide Stufe-0-Ränge
registriert; PP1 „kein PleOffloadLayer, kein Connector“; GPU 4 aus dem Worker
erreichbar (31,43 von 31,73 GiB frei); Warten und Zusammenführen im vollen
Graphen unter MTP k=4, TP2 × PP2, async. Host nach dem Start weiter knapp
(MemAvailable 2,0 GiB, Swap ~10 GiB) — erwartbar, Stufe 3 ist noch leer.

Ursache der Drift: Die FX-Graph-Differenz (Cache `547fc7c767` gegen
`b607d7b65a`, `computation_graph.py`) beschränkt sich auf das PLE-Teilstück.
Das Zusammenführen selbst ist unter Inductor bitgleich zum Eager-Pfad
(1–2048 Token, 25 % Host-Anteil; Test
`test_pinned_host_ple_merge_stays_bit_identical_under_inductor`, schlägt bei
entferntem Host-Merge an). Die Drift entsteht also aus der anderen Bündelung der
fp16-Folgeoperationen im neu übersetzten Teilstück, nicht aus falschen Zeilen.
Einschränkung: der Inductor-Test läuft ohne vLLMs Compile-Pässe.

Abnahmekriterium angepasst (Peuqui 15.09.: bitgleiche Ausgaben sind bei
Flash-Next über Graph-Änderungen nicht zu verlangen): unveränderter Pfad
bitgleich (erfüllt, 3×), Kaskade ohne Fehler, Abweichung nur an
Beinahe-Gleichständen, Tempo-Kosten dokumentiert. Werkzeuge:
`handover/2026-09-15/ple_probe.py`, `ple_logprobs.py`, `ple_cascade_boot.sh`
(Parameter: OUTDIR, SWAP_MODEL, REF_JSON, SKIP_CONTROL).

## 11. Paket 2 — Umsetzung (2026-09-15 nachts)

Entscheidungen beim Bauen:

1. **Planer** (`plan_ple_placement`, `PLEPlacement` mit `vram_rows`,
   `host_rows`, `store_rows`): Der Host bekommt sein Budget, der VRAM den Rest
   bis zu seinem Budget, der Store den Rest. Ohne Kaskade gibt es kein
   VRAM-Budget (wie bisher: fester Host-Anteil, Rest unvermessen auf die
   Karte). Mit Kaskade wird der VRAM gemessen (`_device_spill_bytes`, dieselbe
   Rechnung wie der Automatikmodus), der Überlauf geht auf die Speicherkarte.
   Rest jenseits des Store-Budgets = Startfehler.
2. **`VLLM_QWEN4EXP_PLE_HOST_GIB` bleibt ein fester Anteil je Rang.** Er wird
   einmal geprüft, in `create_engine_config` vor dem Start der Worker
   (`check_ple_host_share`): Anteil × TP ≤ MemAvailable − Reserve, sonst
   Startfehler. Nicht in `VllmConfig.__post_init__`, weil der Modellaufbau die
   Config in jedem Worker neu baut (`with_hf_config`), während Stufe 0 schon
   pinnen kann; nicht im Rang, weil beide Ränge gleichzeitig platzieren und
   einer die Anteile des anderen doppelt zählen würde. Der Automatikmodus
   kappt weiter im Rang.
3. **`VLLM_QWEN4EXP_PLE_STORE_GIB`**: Gesamtbudget auf der Speicherkarte,
   gleichmäßig auf die TP-Ränge; Pflicht mit `STORE_DEVICE`, ohne ihn Fehler.
   Der Worker prüft beim Binden den freien Speicher der Karte.
4. **Kein Fan-out je Rang nötig** (Übergabe-Punkt 3 entfällt): Die
   Store-Bereiche der Ränge sind im globalen Id-Raum disjunkt
   (`plan_ple_store_segments`), ein Rang übernimmt nur Ids seines Bereichs und
   maskiert fremde. Ein Gather über die aneinandergelegten Bereiche
   (`ple_store_indices`) füllt einen Puffer für alle Ränge;
   `_handle_requests` bleibt unverändert. Test
   `test_one_worker_buffer_merges_into_every_rank_bit_identically` (zwei
   simulierte Ränge, Summe bitgleich zur vollen Tabelle), Mutationsprobe
   (genullte Indizes, fehlende Offsets) schlägt an.
5. **Laden:** Ränge kopieren über `copy_ple_embedding_shard_tiers_` (VRAM,
   Host, Store = `None`), der Worker lädt die Store-Bereiche beim Binden aus
   den gemappten Shards als Rohbytes auf die Speicherkarte. Pro Schritt:
   Indizes auf der CPU, `index_select` auf der Karte, blockierende Kopie in
   den gepinnten Puffer, danach wie bisher asynchron in die Ränge.
6. **SSOT:** Budget-/Reserve-Helfer liegen in `common/ple.py` und werden von
   Config und Rang genutzt; die doppelte STORE_DEVICE-Prüfung aus Paket 1 ist
   weg.

Tests: 147 in beiden Reihenfolgen grün (PLE, Offload-Worker, SM70-Config,
Executor), ruff/typos/mypy 3.10 sauber.

### Abnahme 2026-09-16 (Belege in `handover/2026-09-15/p2_*`)

| Lauf | Host gepinnt | GPU 4 | MemAvailable nach Start | Text | Decode tok/s |
|---|---|---|---|---|---|
| Produktion (Referenz `ref_full.json`) | 12 GiB | — | ~2 GiB | Referenz | 60,5 / 56,5 / 66,8 / 60,2 |
| Kaskade `PLE_HOST_GIB=2` | 4 GiB | 4,3 GiB (28,85 Mio. Zeilen) | **12,3 GiB** | 4/4 bitgleich | 59,0 / 54,9 / 65,0 / 58,8 |
| Kaskade `PLE_HOST_GIB=0` | 0 | 8,3 GiB (55,70 Mio. Zeilen) | **16,8 GiB** | 4/4 bitgleich | 59,1 / 55,1 / 65,3 / 58,8 |
| Kontrolle, Eintrag unverändert | 12 GiB | — | ~2 GiB | 4/4 bitgleich | 60,1 / 56,0 / 65,9 / 60,6 |
| llama-swap-Eintrag `-ple-cascade` | 4 GiB | 4,3 GiB | 15,7 GiB | 4/4 bitgleich | 59,2 / 55,1 / 64,9 / 59,2 |

- **Prefill unverändert** (`ple_prefill.py`, je Lauf eigener Zufallstext gegen
  den Prefix-Cache): Produktion 13k 11,8 s / 39k 35,6 s = 1.092–1.100 tok/s,
  Kaskade 13k 11,8–12,1 s / 39k 35,5 s = 1.075–1.098 tok/s.
- **Decode kostet 2,3–2,4 %.** Die Store-Stufe wird bei jedem Schritt
  anteilig getroffen (Hash-Adressierung), der Weg ist Prozesswechsel plus zwei
  PCIe-Hops über den USB4-Tunnel.
- Laden der Store-Zeilen: 84,6 s für 4,3 GiB, 157,6 s für 8,3 GiB, vom mmap
  direkt auf die Karte, ohne anonyme Kopie im Host.
- Placement je Rang bei `HOST_GIB=0`: 131,5 / 132,8 Mio. Zeilen im VRAM
  (19,6 / 19,8 GiB), Rest auf der Speicherkarte — die Ränge sind ungleich, weil
  der gemessene VRAM-Überlauf je Karte leicht abweicht.
- **Fallstrick beim Messen:** Test-Boots aus dem Terminal laufen im
  Benutzer-Slice; `systemd-oomd` killt dort ab 50 % Speicherdruck die Einheit
  mit dem größten Druck — zweimal traf es den kompletten VSCode-Scope (und
  damit Terminal, Treiberskript und Test-Server). Boots über llama-swap laufen
  im System-Slice und sind davon nicht betroffen. Deshalb gibt es den Eintrag
  `Qwen3.8-Flash-Next-180B-A4B-NVFP4-MTP-vllm-ple-cascade` (Sicherung der
  Config unter `~/.config/llama-swap/backups/`).

## 12. Paket 3 — SSD-Stufe (2026-09-16)

Umsetzung:

1. **Vierte Stufe im Planer.** Reihenfolge Host-Budget, gemessener VRAM,
   Store-Budget, Rest auf die gemappte Checkpoint-Datei. Ohne
   `VLLM_QWEN4EXP_PLE_DISK=1` ist ein Rest weiterhin ein Startfehler; die
   Meldung nennt beide Auswege (Budgets erhöhen oder SSD-Stufe erlauben).
2. **Kaskade ohne freie Karte.** Der Schalter allein startet den Worker
   (`ple_cascade_configured`), `STORE_GIB` ist nur mit `STORE_DEVICE` Pflicht.
   Damit deckt die Kaskade den Zwei-Karten-Fall ab: VRAM → Host → SSD.
3. **Der vorhandene mmap-Leser wird wiederverwendet.** Sein Kern (sortierte
   eindeutige Ids je Shard, `MADV_RANDOM`, Thread-Pool) liegt als
   `_gather_mapped_rows` und bedient die alte Disk-Spur wie die neue Stufe.
   `PLERemotePlacement` trägt jetzt zusätzlich `store_rows`; was darüber
   hinausgeht, ist die SSD-Stufe (`plan_ple_worker_segments` liefert beide
   Segmentlisten).
4. **Ein Puffer für beide Außenstufen:** Store-Zeilen per `index_select` von
   der Karte, SSD-Zeilen per mmap, beide in dieselbe Ausgabe; jeder Rang nimmt
   nur die Slots seiner eigenen Segmente.

Tests: 148 im PLE-Satz grün (beide Reihenfolgen), Executor-Tests grün bei
freien Karten; Mutationsprobe (genullte bzw. verschobene SSD-Zeilen) lässt die
beiden neuen Worker-Tests durchfallen. ruff/typos/mypy sauber.

### Abnahme 2026-09-16 (Eintrag `…-MTP-PLE-Disk-vllm`, Store-Budget 1 GiB)

| | Rang 0 | Rang 1 | Worker gesamt |
|---|---|---|---|
| VRAM | 19,60 GiB | 19,79 GiB | — |
| Host gepinnt | 2,00 GiB | 2,00 GiB | 4,0 GiB |
| Store (GPU 4) | 0,50 GiB | 0,50 GiB | 1,0 GiB (6,71 Mio. Zeilen, Laden 25,5 s) |
| SSD | 1,74 GiB | 1,56 GiB | 3,3 GiB (22,14 Mio. Zeilen) |

- Text **4/4 bitgleich** zur Produktionsreferenz, zweimal gemessen.
- Decode 58,1 / 53,7 / 63,8 / 58,7 tok/s, Wiederholung 59,1 / 55,3 / 65,6 /
  58,9 — also rund 1,5 % unter der Kaskade ohne SSD-Stufe und 3–4 % unter der
  Produktion.
- Prefill unverändert: 13k in 11,8 s, 39k in 35,5 s (1.096–1.100 tok/s).
- MemAvailable nach dem Start 12,2 GiB, GPU 4 belegt 1,8 GiB.

**Einschränkung, ehrlich gemessen:** Die SSD-Stufe wurde in diesen Läufen
**nicht von der Platte** bedient. Der Offload-Worker hält die Shards gemappt;
seine Dateiseiten bleiben resident (`RssFile` 967 MiB), und
`posix_fadvise(DONTNEED)` kann gemappte Seiten nicht verwerfen — gemessen am
Worker-Prozess: 76 KiB gelesen und 18 Major-Faults über eine ganze Sonde (warm),
0 MiB und 3 Major-Faults nach dem Freigabeversuch. Die Zahlen oben belegen also
den Mechanismus und die Bitgleichheit, nicht die Latenz echter Plattenzugriffe.
Dafür müsste der Page-Cache mit Root-Rechten geleert werden
(`drop_caches`) oder die SSD-Stufe deutlich größer als der freie RAM sein.

**Nachtrag zur Prefill-Messung (16.09., Peuqui):** `ple_prefill.py` baute seine
Prompts aus 16 Wörtern. Ein MoE-Modell routet so einen Text an wenige Experten
und misst viel zu schnell — 1.100 tok/s gegen die im Alltag üblichen 550–580.
Die Sonde liest jetzt echten Fließtext (Markdown der Repos, je Lauf ein anderer
Abschnitt gegen den Prefix-Cache) und misst auf der Produktion mit Kaskade
653–664 tok/s bei 22k und 74k Prompt-Token. Der Vergleich Kaskade gegen
Produktion bleibt gültig, weil beide Seiten denselben Text bekamen; die
absoluten Zahlen in Abschnitt 11 und 12 sind es nicht.

## 13. PR und Produktionsentscheidung (2026-09-16 abends)

**PR #646 an 1Cat gesendet** (Go Peuqui): Branch `qwen4exp-ple-tier-cascade-pr`
auf origin/main 02c87ab8, Commit 1 = Kopie von #622, Commit 2 = Kaskade mit
`docs/design/qwen4exp_ple_tier_cascade.md`. Abhängigkeiten im Text: #622 (Code),
#640 (Flash-Next-NVFP4 auf pre-Ampere), #639 (MTP unter PP). Beim Zuschneiden
gefunden und im Fork nachgezogen (e89b6543): Partition-Fix im Offload-Worker
war nur Overlay (jetzt mit Regressionstest), 1Cats Hook verbietet neue
`torch.cuda`-Aufrufe (→ `torch.accelerator`), deutsches Zitat im Testkommentar.
Fork-Boot danach sauber, Sonde 4/4 bitgleich (`p3_probe_fork_commit.json`).
Entwurf und Belege: `upstream-contrib/03-1cat-issues/pr-qwen4exp-ple-tier-cascade.md`.

**Produktion zurück auf PLE ohne Kaskade, aber `HOST_GIB=3` statt 6.** Die
RTX 8000 hatten auf beiden Wegen KV-Überschuss (Stufe 0 bekam 4,2–7,2 GiB,
262k brauchen 1,76 GiB; der KV-Pool hängt an der V100-Stufe). Messung
(`handover/2026-09-15/p4_classic_host3.json`, kalter Compile):

| | PLE-VRAM je Rang | Host | GPU 4 | KV Stufe 0 | MemAvailable | Text | Decode |
|---|---|---|---|---|---|---|---|
| Classic 6 GiB | 17,84 GiB | 12 GiB | frei | 6,14 GiB | 2,5 GiB | Referenz | 60,1 / 56,0 / 65,9 / 60,6 |
| Classic 3 GiB | 20,84 GiB | 6 GiB | frei | 3,15 GiB | 8,2 GiB | 4/4 | 59,7 / 56,0 / 66,3 / 60,1 |
| Kaskade | 19,6 / 19,8 GiB | 4 GiB | 4,3 GiB | 4,19 GiB | 11,8 GiB | 4/4 | 59,0 / 54,9 / 65,0 / 58,8 |

Die vier Produktionseinträge in llama-swap stehen seitdem auf
`CUDA_VISIBLE_DEVICES=0,2,1,3` und `VLLM_QWEN4EXP_PLE_HOST_GIB=3` (Sicherung
`~/.config/llama-swap/backups/config.yaml.pre-prod-classic-host3-20260916`).
`…-MTP-PLE-Classic-vllm` ist damit gleich der Produktion, `…-MTP-PLE-Disk-vllm`
bleibt als Kaskaden-Testeintrag. Die Kaskade bleibt im Fork und ist für
Tabellen gedacht, die deutlich nicht mehr in VRAM plus Host passen.
