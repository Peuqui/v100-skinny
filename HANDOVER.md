# Übergabe — Stand 15.09.2026 spät

**Auftrag für die nächste Instanz: PLE-Überlaufkaskade, Paket 2 — vierstufiger
Planer und echte Zeilen auf GPU 4.** Paket 1 (Durchstich) ist gebaut, gebootet
und nach angepasstem Kriterium abgenommen. Stand in `STAND.md` Punkt 6,
Entwurf und alle Messungen in `docs/PLE-KASKADE-ENTWURF.md` (zuerst Abschnitte
3, 4, 8, 9, 10 lesen).

**Arbeitsort:** 1Cat-Fork, Branch `qwen4exp-ple-tier-cascade` im
PRODUKTIONS-Checkout `1Cat-vLLM-work`, Paket 1 = `98cba400`, gepusht nach `fork`. llama-swap lädt beim
nächsten Flash-Next-Boot diesen Branch; ohne Schalter ist er bitgleich zur
Produktion belegt (3 Kontroll-Läufe). Rückweg `git switch work-main`.

**Was Paket 1 gebaut hat (Einstiegspunkte im Code):**
- Schalter `VLLM_QWEN4EXP_PLE_STORE_DEVICE` (envs.py, Compile-Schlüssel).
  `config/vllm.py`: `_qwen4exp_ple_cascade_requested` prüft den Vertrag nur
  für Configs mit Modell (der Offload-Worker baut eine modellose
  `VllmConfig()` — daran scheiterte Boot 1), `_apply_qwen4exp_ple_cascade_defaults`
  setzt `VLLM_PLE_CPU_OFFLOAD` und den IPC-Pfad.
- `PleOffloadLayer.offload_keeps_local_tables()`: Konstruktor und Gewichte
  bleiben in den Rängen; `wait_offloaded_output` ist das Warten im Graphen.
- `Qwen4ExpPinnedHostEmbedding.forward(ids, remote_rows=…)`: Maskierung wie
  VocabParallelEmbedding, Zusammenführen per `where` nach rang-lokaler Id
  `>= local_rows` VOR dem All-Reduce, Dequant der Worker-Bytes mit demselben
  Kernel.
- Registrierung trägt `remote_placements` (`PLERemotePlacement`: tp_start,
  tp_end, local_rows); Worker bindet sie in `_bind_remote_placements`.
- Worker-Seite `Qwen4ExpNGramEmbedding`: mmap-Shards (`_file_backed_shards`),
  `bind_remote_placements` (lehnt `remote_rows > 0` noch ab, öffnet GPU 4),
  `_remote_lookup` (füllt Nullen).
- MRV2 `_setup_ple_offload`: Ränge ohne PleOffloadLayer bauen keinen Connector.

**Paket 2 — was zu tun ist:**
1. `plan_ple_placement`/`PLEPlacement` (`common/ple.py`) auf Bereiche
   VRAM / Host / Store erweitern; Host-Kappung (`cap_host_budget_bytes`) immer
   anwenden, explizite Überschreitung = Startfehler (heute umgeht ein
   gesetztes `VLLM_QWEN4EXP_PLE_HOST_GIB` die Kappung); neues Budget
   `VLLM_QWEN4EXP_PLE_STORE_GIB`. Die Ränge allokieren den Store-Anteil nicht.
2. Worker lädt je TP-Rang dessen Store-Bereich vom mmap direkt auf die
   Speicherkarte und gathert dort (`index_select`), statt Nullen zu schreiben.
3. **Achtung Fan-out:** `PleOffloadRunner._handle_requests` rechnet EIN Ergebnis
   und kopiert es in alle TP-Ränge. Mit Kaskade hat jeder Rang eigene Zeilen
   (eigener Vokabelbereich) — die Ausgabe muss je Rang gebaut werden. Reihenfolge
   der Bytes: je N-Gramm-Id in `(tokens × 16)`-Ordnung, 160 B je Id; Ids
   außerhalb des Rangbereichs werden im Rang ohnehin maskiert.
4. Unit-Tests: Planer-Grenzen, Kopie in drei Ziele, Worker-Gather gegen
   Referenz, Merge mit echten Worker-Zeilen (Muster: die Tests
   `test_pinned_host_ple_merges_the_workers_rows_bit_identically` und
   `…_merge_stays_bit_identical_under_inductor`).
5. Test-Boot vorher bei Peuqui ansagen. Abnahme: Host-Anteil wie eingestellt,
   Rest auf GPU 4; MemAvailable und Swap nach dem Start (heute ~2 GiB / ~10
   GiB), tok/s, Prefill-Zeit, Text gegen `ref_full.json` (Abweichung nur an
   Beinahe-Gleichständen, siehe Kriterium), mehrere Host-Einstellungen.

**Werkzeuge (`handover/2026-09-15/`):**
- `ple_cascade_boot.sh OUTDIR [SWAP_MODEL] [REF_JSON] [SKIP_CONTROL]` — entlädt
  Flash-Next, bootet den llama-swap-Eintrag exakt nach (Port 8093) mit
  `CUDA_VISIBLE_DEVICES=0,2,1,3,4` und dem Schalter, Sonden, Log-Belege, Stopp
  über `vllm-swap-stop`, optional Kontroll-Boot. Für Paket 2 die zusätzlichen
  Env-Werte in der `overrides`-Zeile ergänzen. AIfred während des Boots
  stoppen, sonst fordert Vigilantia Flash-Next an.
- `ple_probe.py` (3 Prompts, greedy, 260 Token, voller Text),
  `ple_logprobs.py` (Top-2 für Prompt 3), Referenzen `ref_full.json`,
  `ref_logprobs.json` (Produktion 15.09., Tempo 56–67 tok/s).
- Kalter Boot mit neuem Schalterwert ≈ 13,5 min, warm ≈ 7,5 min.

**Nebenbefunde, nicht behoben (eigenes Paket vorgeschlagen):**
- `tests/compile/passes/test_functionalization.py`: 7 bf16-Fälle scheitern auf
  SM70 (Inductor lehnt BF16 ab).
- `tests/models/qwen4_exp/test_qsa_reference.py`: 3× fehlendes Attribut
  `block_table_buffer`, 1 Vergleichsfehler; weitere GPU-Tests laufen bei
  belegten Karten in OOM.
- Test-Verschmutzung über `vllm.envs` (setattr hinterlässt Modulattribute,
  delenv merkt sich fehlende Variablen nicht): Helper `set_lazy_env` in
  `tests/utils.py`, von neuen Tests genutzt; ältere Tests nutzen weiter
  `monkeypatch.setattr(envs, …)`.

**AIfred (erledigt, gepusht `18197f85`):** Chat-Liste sortiert nach
`last_message_at` (nur neue Nachrichten heben an), Login-Autoload bleibt bei
`last_seen`.

## Übergabe — Stand 15.09.2026 abends (abgelöst)

**Auftrag für die nächste Instanz: PLE-Überlaufkaskade bauen, Paket 1
(Durchstich).** Entwurf mit allen Entscheidungen: `docs/PLE-KASKADE-ENTWURF.md`
(zuerst lesen, Abschnitte 2, 4, 7, 8, 9).

- Anlass: Flash-Next pinnt 12 GiB PLE im Host-RAM (je Rang 6 GiB, Tabelle
  TP-geteilt, NICHT doppelt); der Mini swappt ~20 GiB. Messung und Sampler in
  `handover/2026-09-15/` (`coldstart_mem.csv`, `memsample.sh`).
- Reihenfolge der Stufen: VRAM → Host (konfiguriert, z. B. 10 GiB gesamt) →
  freie GPU (GPU 4) → SSD. Budgets und Reserven konfigurierbar und dynamisch,
  nichts fest eingebaut. SSOT: vorhandenen Planer, Host-Budget-/Reserve-
  Funktionen und den PLE-Offload-Worker (`vllm/v1/ple_offload/`) übernehmen
  und anpassen.
- NIE vorschlagen, die ganze PLE in den VRAM der Rechenkarten zu legen
  (`PLE_HOST_GIB=0`) — passt nicht.
- Keine Ankündigung bei 1Cat; im Fork bauen und abnehmen, danach PR anbieten.
- **Arbeitsort:** Branch `qwen4exp-ple-tier-cascade` im PRODUKTIONS-Checkout
  `1Cat-vLLM-work` (von work-main `d228c725`, noch ohne Änderung). llama-swap
  lädt beim nächsten Flash-Next-Boot diesen Branch. Neue Stufen nur per
  Umgebungsvariable aktiv; vor jedem Test-Boot Unit-Tests; jeden Test-Boot
  (≈ 8 min ohne Flash-Next in AIfred) vorher bei Peuqui ansagen; zum Schluss
  Kontroll-Boot mit unverändertem Eintrag, bitgleich. Rückweg:
  `git switch work-main`.
- Größtes Risiko: Offload-Worker lief nie mit MTP k=4 + PP2 + async; Produktion
  nutzt `FULL_AND_PIECEWISE` (volle Decode-Graphen), Warten über
  `ple_offload_wait` im Graphen.
- Nicht committet: `docs/PLE-KASKADE-ENTWURF.md`, `handover/2026-09-15/`.
  Commit/Push nur auf Peuquis Ansage.

## Übergabe — Stand 14.09.2026 abends (abgelöst)

**Stand in einem Absatz (14.09. abends):** 1Cat main `80c88e8d` ist in work-main
gemergt (`1d3439f2`, Neubau, Tag `verified-2026-09-14`), danach der Overlay
zurückgebaut (`ebc5dc52`, Tag `verified-2026-09-14b`), beides gepusht. Neuer PR
**#636** (Qwen3.5-MTP unter PP), **#611** auf a6f5e834 rebased, neu gemessen (RTX
+3,8 %) und kommentiert. Der PR-Kandidat PP5-Warteschlangen-Deckel ist verworfen
(DeepSeek braucht ihn auf main nicht mehr). Details im Nachtrag 14.09. am
Dateiende; der Absatz darunter ist der Stand vom 13.09.

**Stand in einem Absatz (13.09. abends):** Paket C (FA2 für Turing) ist im Fork
und als PR #623 bei 1Cat; der Compile-Cache ist in der Produktion an und belegt,
nachdem zwei Fehler gefunden und behoben wurden (PLE-Zeiger im AOT-Artefakt →
PR #622; torch 2.10.0 ohne Kernel-Tabelle → Backport in `tools/torch_patches/`,
auf #621). work-main `f6c42de7`, Produktions-venv gepatcht, Abnahme aller vier
Produktionseinträge kalt und warm bestanden. Alles committet und gepusht.
Details im Nachtrag 13.09. abends am Dateiende; der Kopf darunter ist der
Stand vom 12.09. früh und gilt, wo der Nachtrag nichts anderes sagt.

**Ergebnis der Nacht 11./12.09. in einem Absatz:** Punkte 1, 3, 4, 6, 7 des Plans sind
erledigt und in `STAND.md` (Punkte 13, 14, 15, 10 der offenen Liste)
dokumentiert; Punkt 5 (Paket G) ist als zwei PR-Worktrees mit Entwürfen
vorbereitet, nicht eröffnet; Punkt 10 (DeepSeek-Tool-Call) ist am
Morgen ERLEDIGT: ein echter Bug (Decode-Schwellen zweier Metadaten-Bauer
laufen unter DSpark auseinander, Query-Längen 7–11 stürzen ab), gefunden,
gefixt, verifiziert — Tool-Call beide Runden bestanden. Fix uncommitted in
work-main und PR-Worktree. Nichts committet außer `e0b000b`
(Denkfrage). Entscheidungen für Peuqui stehen am Ende dieses Dokuments.

**Betriebsstand steht in `STAND.md`. Damit anfangen, nicht mit diesem Dokument.**
Hier steht, was als Nächstes ansteht und was du wissen musst, um nicht
dieselben Wege noch einmal zu gehen. Skripte, Patches, Referenzen und
Ergebnisse der Runden vom 10./11.09. liegen in `handover/2026-09-11/`
(siehe dortige `README.md`), die Logbücher in `docs/journal/`.

---

## Stand der Repos (alles committet und gepusht)

| Repo / Worktree | Stand | Inhalt |
|---|---|---|
| `1Cat-vLLM-work` (work-main) = Produktion | **13.09. abends: `f6c42de7` auf fork/work-main** (vorher `433dfa10` = Tag `verified-2026-09-11b`) | 1Cat main `fe67339d` + fünf offene PRs + Overlay; Merge-Reste bereinigt (`43ccb9b8`), **E5 ausgebaut**, Befund 2 `index_share=True` |
| `1Cat-vLLM-pr-dflash2` | `90c9efee` | PR #599 |
| `1Cat-vLLM-pr-devcap` | `2b59521a` | PR #600 |
| `1Cat-vLLM-editable-pr` | `e5e3e9af` (+ `build/` 2,1 GB für inkrementelle Nachbauten) | PR #601 |
| `1Cat-vLLM-pr-fa2sm75` (13.09.) | `d22daa74` auf fork/sm75-fa2-pr | PR #623 Paket C |
| `1Cat-vLLM-pr-plegather` (13.09.) | `5f668ebb` auf fork/ple-gather-no-baked-pointer | PR #622 PLE-Gather |
| `1Cat-vLLM-pr-compilecache` (13.09.) | `7fa6b04a` auf fork/drop-forced-compile-cache-off | PR #621 + tools/torch_patches |
| `1Cat-vLLM-e2e-fa2mixed` (13.09., Wegwerf) | Branch `e2e-fa2mixed`, WIP-Commits | Testgerüst main+#604+#600+PR-C, löschbar |
| `v100-skinny` (work) | `639d750` auf `fork/work` | Doku, Skripte ohne E5-Schalter, Handover-Material |
| llama-swap `config.yaml` | ohne `VLLM_SM70_E5_CACHE`, Sicherung `backups/config.yaml.bak-2026-09-11-vor-e5` | jeder vLLM-Eintrag bootet beim nächsten Laden einmal kalt |

**Acht PRs bei 1Cat offen:** #572 #573 #574 #576 #592 #599 #600 #601
(Tabelle mit Entwürfen in `upstream-contrib/README.md`). Neun gemergt.
vLLM-Upstream und flash-attention sind **eingefroren** (Entscheidung Peuqui
11.09.): #54758, #54260, #190, #191 bleiben offen, nichts Neues dort.

**Regel seit 11.09.:** Merged 1Cat einen eigenen PR, wird der zugehörige
Overlay-Teil beim nächsten Hereinholen von main ENTFERNT, nicht neu
darübergelegt. Der Diff `work-main` gegen `origin/main` ist die Liste dessen,
was noch als PR raus muss (`upstream-contrib/OVERLAY-INVENTUR.md`). Nach jedem
Merge die hinzugefügten Zeilen gegen die Upstream-Fassung derselben Datei
abgleichen — „konfliktfrei" heißt nicht „sauber".

---

## ~~Zuerst klären: Flash-Next mit Denken bei 13k Kontext~~ — ERLEDIGT 11.09. spät

Die drei Kuanda-Ausfälle kamen vom `MML=16384` des Abnahme-Skripts, nicht
vom Betriebspunkt: der llama-swap-Eintrag fährt MML 262144 (KV 400k Token).
Nachmessung gegen Produktion mit 39k Prompt und Denken: alle drei Fragen
beantwortet, q3 als Zurückweisung ohne Erfindung (`STAND.md` Punkt 16).
**Denken bleibt an.** Die PLE-Kaskade ist dafür nicht nötig; sie ist
beschlossen (Zukunft: Qwen4 mit größeren PLE-Tabellen, Zwei-Karten-Nutzer),
rückt aber ans Ende des Plans hinter Punkt 10 (`STAND.md` Punkt 6).
Abnahme-Skripte stehen jetzt auf `MML=262144`.

---

## Der Zehn-Punkte-Plan (Freigabe Peuqui 11.09. nachts, Reihenfolge so)

1. ✅ 12.09. 00:30: kein neuer Commit, keine Kommentare. **Täglicher Upstream-Check und Reaktionen auf die acht PRs.** `git fetch`
   in `1Cat-vLLM`, `gh pr list --author Peuqui --state open`, Kommentare
   lesen, AGENTS.md vor jeder Antwort frisch lesen. Merges kommen in ein bis
   zwei Tagen.
2. **Nach jedem Merge den Overlay-Teil entfernen** (Regel oben), danach die
   übliche Abnahme (`handover/2026-09-11/scripts/abnahme2/driver.sh` als
   Vorlage: 27B-MTP, DFlash2 beide Paare, Flash-Next 3×, DeepSeek).
3. ✅ ERLEDIGT 12.09. (STAND Punkt 13): kein Hebel, nur `NCCL_BUFFSIZE` 512K–1M +1 %. **Punkt 13, NCCL-Schalter gegen den AllReduce.** 31,8 % des DFlash2-
   Decode-Profils, reine Env-Experimente:
   `handover/2026-09-11/scripts/nccl_sweep.sh <DEVS> <tag> base NCCL_PROTO=Simple
   NCCL_PROTO=LL NCCL_PROTO=LL128 NCCL_ALGO=Ring NCCL_ALGO=Tree
   NCCL_BUFFSIZE=1048576` — fünf DFlash2-Läufe je Variante, Median, Annahme,
   SHA muss konstant bleiben. Erst RTX-Paar `0,2`, dann `1,3`.
4. ✅ ERLEDIGT 12.09. (STAND Punkt 15): bitgleich, sm_75 belegt, kein Tempo-Gewinn; Patch auf work-main angewendet, uncommitted. **Punkt 15, Skinny-Build pro Architektur.** Patch fertig, nicht angewendet:
   `handover/2026-09-11/patches/punkt15_skinny_per_arch.diff` (`marlin.py`,
   Name `skinny_nvfp4_v11_sm{cc}`). Abnahme: DFlash2 V100 SHA-gleich, RTX
   SHA + Tempo + `cuobjdump -lelf` auf der gebauten `.so` (muss sm_75 zeigen).
   Erwartet 1–6 % bei M ≤ 4, nichts bei M=8.
5. 🟡 VORBEREITET 12.09., nicht eröffnet: Worktrees `1Cat-vLLM-pr-timeout`, `1Cat-vLLM-pr-compilecache`, Entwürfe in `upstream-contrib/03-1cat-issues/pr-nccl-subgroup-timeout.md`, `pr-drop-forced-compile-cache-off.md`. Compile-Cache: Erzwingung sitzt ZWEIMAL (auch envs.py), kein Bootzeit-Gewinn gemessen — PR-Würdigkeit entscheidet Peuqui. **Paket G als PRs:** NCCL-Untergruppen bekommen `--distributed-timeout-
   seconds` (`distributed/parallel_state.py`, kalter PP-Boot riss am 600-s-
   Wachhund); erzwungenes `VLLM_DISABLE_COMPILE_CACHE=1` für den 0DOT3-Graph
   entfernen (`config/vllm.py`, Ursache seit #536 behoben). Beide klein,
   Muster wie #599/#600: Worktree auf `origin/main`, Test, pre-commit +
   mypy-3.10, Duplikatsprüfung am Tag, Body aus Entwurf + Checkliste.
6. ✅ ERLEDIGT 12.09. (STAND Punkt 14): 5.194 ms, AllReduce 34 %, qpn2 27 %, qpn8 18 %; fp16-GEMMs weg. **Punkt 14, Decode-Profil mit NVFP4-Entwurfskopf:** `prof_dflash.sh <name>
   0,2` mit `DRAFT=<maurienne>`. Zeigt, was nach Block-Pack und Kopfwechsel
   von den 435 ms übrig ist (`STAND.md` Punkt 13).
7. ✅ ERLEDIGT 12.09. (STAND Punkt 10): alle Kernel-Tests grün auf V100 und RTX, DeepSeek 8/8 byteidentisch, 27B SHA gleich; Entwurf `comment-tilelang-pin-0114.md`, nicht gepostet. **Punkt 10, TileLang-Pin testen:** venv `.venv-sm70-tltest` (0.1.14) steht.
   `tests/kernels/test_mhc_kernels.py` und `test_mhc_sm70_fp16.py` je auf
   V100 und RTX, DeepSeek PP5 über `ds_accept.sh` mit `VENV=.../.venv-sm70-
   tltest`, ein GDN-Modell booten. Danach Pin-Vorschlag an 1Cat (tilelang
   und apache-tvm-ffi, 1Cat pinnt 0.1.10); unser Overlay-Patch
   `fork_patches_150/tilelang_target.py` wird damit überflüssig. Danach die
   11-GB-venv löschen.
8. **Punkt 9, Routenzähler, dann Skinny-Angebot an 1Cat (Paket E):**
   `VLLM_SKINNY_ROUTE_COUNT_FILE` in den Produktionskonfigurationen laufen
   lassen; dann (a) Turing-Pfad für 1Cats QPN-Kopie, (b) Block-Pack, (c)
   MoE-Backend erst nach Rückfrage in #441. `VLLM_SKINNY_*` in `envs.py`
   anmelden. Das größte Paket, Tage.
9. **Paket 11c, FA2 auf Turing für 1Cat** — erst wenn #572 gemergt ist (der
   Lackmustest, ob 1Cat Turing will). Weg: sm75-FA als gepinnter Fork plus
   Patch-Dateien in `cmake/patches/`, wie 1Cat zhinianqins V100-FA einbindet;
   zweite Bibliothek `_vllm_fa2_C_sm75` plus unser `load_fa2_library(device)`,
   plus Marlin-Arch-Gate in 1Cats CMakeLists (Zeile ~454). 11d (Triton-3D-
   Split-KV) dahinter; 11e (DeepSeek-Gerät-0-Gates) ist durch #600
   gegenstandslos, sobald der gemergt ist.
10. ✅ ERLEDIGT 12.09. ~05:00: Ursache gefunden (SWA-Decode-Schwelle 6 gegen C128A-Schwelle 11 unter parallelem Drafting, Fenster 7–11 Query-Token, trifft Kurzprompts UND Prefix-Cache-Reste wie die Werkzeug-Rückrunde), Fix im Worktree `1Cat-vLLM-pr-swathreshold` + als Overlay auf work-main, verifiziert: 13/13 Längen, Tool-Runde 1 und 2 bestanden. Entwurf `pr-sparse-swa-spec-threshold.md`. UNCOMMITTED. **DeepSeek-Eintrag einmal ein Werkzeug aufrufen lassen.** Regel vom 11.09.:
    ein Profil ist erst abgenommen nach einem echten Tool-Call. Nie geprüft;
    1Cat-Issue #597 (delubee, DSML-Tool-Calls kaputt auf 8× V100) trifft
    genau das. Ergebnis ggf. dort als Hinweis.

Nicht in der Liste, weil entschieden: E5 bleibt draußen; Bitgleichheit wird
nicht geopfert; keine sm75-MMA-Variante; `qpn8` bleibt ungepackt; batch-
invariante Kernel kommen nicht; PIECEWISE bleibt unangefasst.

---

## Fallen dieser Runde (teuer erkauft, 11.09.)

- **`pgrep -f <muster>` trifft die eigene Shell**, wenn deren Befehlszeile das
  Muster enthält — hat mich zweimal am Tag erwischt (einmal „Kette tot"
  gemeldet, die lebte). Muster mit Klammer entschärfen: `pgrep -f
  "[a]bnahme2/driver.sh"`.
- **Ohne `CUDA_DEVICE_ORDER=PCI_BUS_ID` ist `CUDA_VISIBLE_DEVICES=4` eine
  RTX 8000**, nicht die V100 — der SM70-Rerank-Test übersprang sich deshalb
  still. In jedem pytest-Aufruf mit GPU die Variable setzen.
- **`systemd-run --user` überlebt Sitzungsneustarts, taugt aber nicht für
  Bauten**: minimaler PATH (kein `ninja` aus dem venv-bin), CMake findet
  darunter die Python-Header nicht. Bauten aus der Shell per `setsid nohup`;
  die so gestartete Kette hat den Neustart nachweislich überlebt.
- **Ein Test mit `is_device_capability`-Mocks bricht, sobald der Code
  `torch.accelerator.current_device_index()` ruft** — CPU-only-CI meldet
  „No CUDA GPUs are available". Vor jedem PR die KOMPLETTE bestehende
  Testdatei laufen lassen, nicht nur den neuen Test (Lehre aus #485, heute
  bei #599 erneut greifbar).
- **Ein reines main lädt den maurienne-NVFP4-Entwurfskopf nicht** (#592
  offen): `mat1 and mat2 shapes cannot be multiplied`. Wheel- und
  main-Abnahmen mit dem incoai-Kopf fahren.
- **Ein reines main + #572 lädt NVFP4 `modelopt_mixed` auf Turing nicht**
  („Minimum capability: 89"). Turing-E2E auf main gibt es erst mit Paket C/E.
- **Die Rohtext-Sonde `flashnext_qual.sh` misst nicht den Produktionspfad**
  (kein Template, kein `enable_thinking`); Chat-Variante
  `handover/2026-09-11/scripts/abnahme2/flashnext_qual_chat.sh`, dort maximal
  `MAXTOK=3300` bei 13k Prompt (sonst HTTP 400).
- **Kuanda-Bewertung** (Peuqui): Zurückweisen mit Coandă-Nennung ist bestanden,
  nur Erfinden oder Zerfasern ist Durchfall (`STAND.md`, Fallstricke).
- **`git apply` eines gespeicherten Diffs ist der sichere Rückweg**, nie
  `git stash` (geteilter Stash-Stapel über alle Worktrees).

## Werkzeuge

- `handover/2026-09-11/scripts/abnahme2/`: `driver.sh`/`driver2.sh`
  (Abnahmen), `flashnext_qual_chat.sh` + `chat_ask.py` (Chat-Sonde),
  `fnq_check.py` (Vorfilter), `e5_ab.sh` (E5-Messvorrichtung, historisch),
  `wheel_test.sh` (Wheel-Bau + Abnahme), `devcap_probe.sh` (Gerät-0-Probe
  auf gemischtem Knoten), `rerank_v100.sh`, `nccl_sweep.sh`.
- `handover/2026-09-11/ergebnisse/device0_capability_calls_inventory.md`:
  263 Capability-Aufrufe ohne Geräteindex in 1Cat-main (Klasse „Modulebene"
  unzuverlässig) — nach Merge von #600 nur noch für Import-Zeit-Konstanten
  relevant.
- `ncu` als normaler Nutzer; `benchmarks/qpn2_pack_ab.py --ref <rev>`;
  `DRAFT=` in `speed_dflash.sh`/`prof_dflash.sh`.


---

## Entscheidungen für Peuqui (12.09. früh)

1. **Paket G eröffnen?** Timeout-PR ist klar (Code-Beweis + Test). Compile-
   Cache-PR ist Bereinigung ohne Tempo-Gewinn; Messtabelle im Entwurf.
2. **Punkt-15-Patch behalten** (korrekt, kein Gewinn) oder `git apply -R`.
3. ✅ ERLEDIGT 13.09.: `NCCL_BUFFSIZE=1048576` in den sechs TP2-Einträgen (27B, DFlash2, vier Flash-Next-Varianten), Sicherung `backups/config.yaml.pre-nccl-buffsize-20260913-1035`, llama-swap neu gestartet.
4. **envs.py-Default im Fork** zurücknehmen (Compile-Cache auf 0DOT3-Pfad
   ist in Produktion aus) — Overlay-Änderung, unabhängig vom PR.
5. ✅ ERLEDIGT 13.09.: TileLang-Pin-Vorschlag als **Issue #620** bei 1Cat gepostet.
6. ~~DeepSeek an #597 melden~~ — ENTSCHIEDEN 12.09. früh (Peuqui): NICHT
   kommentieren. #597 (delubee, 8× V100 SXM2 NVLink, offizielles Wheel)
   meldet (1) DSML-Tool-Calls 0/10 korrekt durch falsche Sondertoken-Wahl im
   Modell und (2) Decode 16 tok/s statt 65–73. Unser Befund ist ein anderer:
   Tool-Call korrekt, aber Engine-Absturz bei Kurzprompt und Hänger bei der
   Werkzeug-Rückrunde. Unsere ähnlich niedrigen PP5-Zahlen sind nach Peuquis
   Einschätzung eher dem PP5-Betrieb geschuldet, nicht demselben Effekt —
   unbewiesen in beide Richtungen. Kurzprompt-Grenzmessung läuft
   (`abnahme2/ds_short_prompt_boundary.sh`).
7. **PLE-Kaskade** bleibt am Ende des Plans (deine Entscheidung vom Abend).
8. ~~SWA-Schwellen-Fix committen und als PR eröffnen~~ — ERLEDIGT 12.09.
   ~05:30: work-main `911c259f` (auf fork/work-main), **PR #603** offen.
9. **Paket E (12.09.):** E-1 als **PR #604**, E-2 als **PR #611** eröffnet
   (Block-Pack, Turing-Schwelle 7 gemessen). Befund Compile-Münze/AOT-
   Cache pro Env-Hash in `paket-e-plan.md` + Memory. E-1 als (Turing-Pfad
   für modelopt NVFP4+FP8 über die kompilierten QPN-Kernel; RTX 71,0 tok/s
   gegen V100 63,4, SHA gleich). Reihenfolge Peuqui: E-2 Block-Pack →
   FlashInfer-Turing-Dreizeiler → Paket C (sm75-FA2) → fp8-KV-Cast. Plan:
   `upstream-contrib/03-1cat-issues/paket-e-plan.md`.
10. **#572 und #573 sind GEMERGT (12.09. 03:30), main ist 24 Commits weiter
   (`ae75fb9b`, u. a. #602 „SM70 grouped long-context route on by default",
   #596 DFlash2-Tail-Graphen, #595 FP8-MTP-Experten).** Nach der Regel:
   work-main auf main hereinholen und die Overlay-Teile von #572/#573
   ENTFERNEN, danach Abnahme (`abnahme2/driver.sh`). Größerer Eingriff in
   den Produktionsbaum — vorher Freigabe.

## Punkt 2 ERLEDIGT: work-main auf main dfef3342 (13.09. früh)
- **Paket G, Teil 1 ERLEDIGT: PR #619** (NCCL-Untergruppen-Timeout, 13.09.). Teil 2 (Compile-Cache-Zwangsabschaltung): Kartentyp-Test 13.09. BESTANDEN (RTX bekommt eigenen Schlüssel). NEUER BEFUND: der erste Warmstart nach einem Kaltlauf lädt das Artefakt nicht (torch-Assertion `kernel_side_table` ohne Meldung), speichert neu, ab dem zweiten Warmstart lädt alles (70–75 s statt 110–120 s). Details + Traceback in `pr-drop-forced-compile-cache-off.md`. Ursache ist bei PyTorch BEHOBEN (PR #173556, gemergt 28.01.2026, `torch/_dynamo/aot_compile_types.py` serialisiert die Kernel-Tabelle; enthalten ab torch 2.11, NICHT in 2.10.0, das 1Cat pinnt; vLLM upstream ist auf 2.13). Kein PyTorch-Issue nötig. ERLEDIGT 13.09.: Fork-Default umgestellt (work-main `adf3e5bd`), **PR #621** eröffnet.
- **AIfred `fcdc9db1`:** Kalibrations-Cache wird nach abgeschlossenem Lauf geleert, Produktions-Cache nächtlich nach 21 Tagen Nichtnutzung oder über 40 GiB gestutzt (Peuqui 13.09.). Wirksam ab dem nächsten AIfred-Start.
- Merge `34f3f340` + `f381618a`, Neubau, Abnahme komplett bestanden (Tabelle in STAND.md "Abnahme work-main-Merge"). Produktionskopf: RTX 76,88 / V100 76,42 tok/s, SHA gleich; DeepSeek 8/8; Flash-Next q1/q2 sauber, Kuanda 2:1 wie alt; alle vier llama-swap-Einträge kalt+warm ok.
- ERLEDIGT 13.09.: `speed_dflash.sh` Vorgabe `DRAFT` = Produktionskopf (maurienne RTNcal), Commit 4df1d52.
- ERLEDIGT 13.09.: `1Cat-vLLM-old-prod` und `.venv-sm70-old` entfernt.
- ERLEDIGT 13.09.: Tag `verified-2026-09-13` auf f381618a gesetzt und gepusht.

## Paket C — Stand 12.09. abends
- Messung abgeschlossen (RTX-Paar, 27B, fp16-KV): FA2-sm75 gegen Triton kurz 74,19/70,91 tok/s, 13k TTFT 17,75/36,26 s, 13k Decode 56,48/15,03 tok/s, Text identisch. Details `upstream-contrib/03-1cat-issues/paket-c-plan.md`.
- Eigenständiger sm75-Bau aus dem FA-Fork mit neuer CMake-Option `VLLM_FA2_OUTPUT_NAME` (uncommitted im Fork) funktioniert; Fork liest `Python_EXECUTABLE`, nicht `VLLM_PYTHON_EXECUTABLE`.
- 1Cat-Issue #612 (Bauform, drei Formen ohne Empfehlung) veröffentlicht; wir warten auf die Antwort, sonst Form 1 als PR.
- FA2+fp8-KV auf Turing GESTRICHEN (Peuqui): kein Nutzen bei unseren Modellen, Turing rechnet fp16 am schnellsten. Vierter Schritt = Overlay-Politik `checkpoint_kv_quant_allowed` (Boot unter `auto` auf Turing).
- Worktree `1Cat-vLLM-pr-fa2sm75` (Branch `sm75-fa2-pr` auf origin/main): drei FA2-Python-Diffs + Doku-Tabelle, Lint+mypy grün, uncommitted.
- **Vierter Schritt ERLEDIGT: PR #613** (KV-Quant-Vorgabe nur auf Ampere+, Worktree `1Cat-vLLM-pr-kvpolicy`, Belege in `upstream-contrib/03-1cat-issues/pr-checkpoint-kv-quant-pre-ampere.md`). Nach Merge: Overlay-Teil `utils/torch_utils.py`/`layers/attention/attention.py` (OVERLAY-INVENTUR Z.126) entfernen; der PR-Schnitt ist NICHT identisch mit dem Overlay (Teilnahme-Helfer statt Gerät 0, kein Env-Schalter).
- **Turing-Lücke gefunden (12.09. nachts, Abnahme):** die neuen DFlash2-Tail-Cudagraphs (main #609ff., `v1/worker/gpu/cudagraph_utils.py`) und der MTP-Split-Draft-Cudagraph sind mit `is_device_capability((7, 0))` gated — exakt Volta UND Gerät 0. Turing bekommt sie nicht (RTX-Boot-Log ohne „Capturing SM70 DFlash2 target tail"), V100 +1,2 % (75,09 gegen 74,21 tok/s). ERLEDIGT: **PR #618** (13.09.), Overlay in work-main; Turing: MTP-Split +3,6 %, Tail-Graphen neutral (76,50/76,41 gegen 76,55).

## Nachtrag 13.09. abends — Paket C im Fork, zwei Fehler beim Compile-Cache gefunden und behoben

- **Paket C (FA2 für Turing) ist im Fork** (work-main `f6c42de7`, PR **#623** eröffnet 13.09. ~17:05): Lader pro Gerät mit Volta-Ladefix (`ensure_fa2_library_loaded`, drei SM70-Aufrufstellen), CMake-ExternalProject, heutige sm75-Bibliothek statt Drop-in (`.drop-in-0903` daneben). PR-Branch `sm75-fa2-pr` (1d20869e + d22daa74 Volta-Testkorrektur), Text `upstream-contrib/03-1cat-issues/pr-turing-fa2-sm75.md`, Belege: RTX-Paar, V100-Paar, PP2 gemischt (RTX Stufe 0, V100 Stufe 1, ohne MTP), Text byteidentisch. Grenzen auf main: 27B-MTP-Drafter ohne SupportsPP; TP2 gemischt hängt in `awq_sm70_warmup.py:170` (broadcast in der TP-Gruppe).
- **Fehler 1, PLE-Gather backt Host-Zeiger in den Graphen** (1Cat #403, von uns in #528 fortgeführt): Flash-Next stirbt beim ersten Warmstart mit Cache auf Stufe 0 (illegal memory access). Fix im Fork `ple_layer.py` (Layer-Name + `use_host_table` statt `weight_ptr`, Auflösung im Op über `no_compile_layers`), Tests angepasst, 60 grün. PR **#622** (Branch `ple-gather-no-baked-pointer`, Worktree `1Cat-vLLM-pr-plegather`, 5f668ebb), Text `pr-ple-gather-no-baked-pointer.md`. Beleg: kalt 340 s, warm1 312 s, warm2 343 s, je 6 Artefakte geladen.
- **Fehler 2, torch 2.10.0 serialisiert die Triton-Kernel-Tabelle nicht** (erster Warmstart verpufft): Backport pytorch #173556 als `tools/torch_patches/` im Fork, Produktions-venv gepatcht, `rebuild_work_main.sh` ruft `apply.sh`. Beleg 27B: kalt 476 s, warm1 85 s (4 geladen, 0 Ladefehler), warm2 80 s. #621 ergänzt (Commit 7fa6b04a mit tools/torch_patches, Body-Absatz, Kommentar; `pr-621-amendment.md`, `pr-621-body-new.md`).
- **AOT-Artefakte einmal gelöscht** (22 GB), Inductor-Cache blieb. Nach jedem torch-Install: `tools/torch_patches/apply.sh`, danach `torch_aot_compile/` leeren.
- **Fallen des Tages:** (1) Freeze mitten im Bau → 0-Byte-.o gelten als fertig, Link schluckt sie; Objekte aus dem Absturzfenster löschen, `nm -D | grep ' U '` prüfen. (2) `nohup`-Launcher erben das cwd der Shell → `cd /tmp` in jedes Launch-Skript, Baum-Beleg per `grep 'File "/home/mp/Projekte/vllm-research/[^/]*/'` im Boot-Log. (3) Teardown mit `pgrep -f 'VLLM::'` hat den llama-swap-Server QUASAR getötet → nur `kill -- -<PGID>` der eigenen Gruppe.
- **Turing-Testdateien auf der RTX:** `test_flash_attn.py` ist bf16-only (jeder Fall scheitert am fp16-Gate der sm75-Bibliothek, gewollt); als fp16-Variante ohne fp8-KV 160 bestanden, 160 FA3-Fälle übersprungen. `test_attention_backends.py` hier nicht ausführbar (gesperrte HF-Repos meta-llama/embeddinggemma). Beides so in #623 benannt.
- **Offen:** Reviews/Merges #621 #622 #623 abwarten, Overlay-Rückbau je Merge (fork_patches_150: cuda.py, flash_attn.py, flash_attn_interface.py, flash_attn_v100.py, sm70_e4m3_long/scalar.py für #623; ple_layer-Anteil für #622); Issue #614 (fremd, Flash-V100-Politik unter TP2, mode=none) beim nächsten Upstream-Check lesen; Tag `verified-2026-09-13` auf work-main nur auf Ansage; Wegwerf-Worktree `1Cat-vLLM-e2e-fa2mixed` + `.venv-pr-fa2sm75` + `.wheels/` aufräumbar; #621-Body war ~1 min in einer Zwischenfassung sichtbar (Bearbeitungshistorie), inhaltlich korrekt.

## Nachtrag 14.09. — main-Merge, #636, #611 neu gemessen, Overlay-Rückbau

- **Merge:** 1Cat main `80c88e8d` (20 Commits: SM70-79T/Q8000-Prefill im FA2-Target, geteilte NVFP4-Codes für DFlash2 #561, QPN2-Zeilenblock-Ordnung d66797cc) in work-main, konfliktfrei (`1d3439f2`). Neubau mit `MAX_JOBS=3` (79T-Kernel = 6.800 Zeilen, 30 GB RAM), 38 min, Skript `handover/2026-09-14/rebuild_work_main.sh`. Keine neuen `WITH_SOABI`-Targets. Tag `verified-2026-09-14`.
- **Abnahme:** 27B DFlash2 SHA `0106659946c064b1` RTX 76,72 / V100 76,48 tok/s, Annahme 3,325; alle vier vLLM-Einträge laden und antworten; DeepSeek PP5 Kohärenz 8/8, zwei Läufe identisch (`ds_merge0914`); Flash-Next q1/q2 sauber, Kuanda aus vier Läufen 1 Zurückweisung, 1 Grenzfall (zerfasert), 2 „Kunda-Effekt"-Deutungen — nicht deterministisch, nicht dem Merge zugeschrieben (A/B alt/neu nur bei Auffälligkeiten im Alltag). Decode messbar unverändert; die neuen Prefill-Kernel greifen bei uns laut Doku nicht (79T nur Hq6/Hkv1 = 27B TP4, Chunks >= 8000, V100).
- **VERLUST:** `~/.cache/mtp-diagnostics` wurde beim Plattenaufräumen gelöscht, darin die DeepSeek-Referenz `ds_tl014` und ältere `qual_*`-Ergebnisse. Neue Referenzen: `ds_merge0914`, `qual_accept0914` (RTX), `qual_accept0914v100`, `qual_accept0914mtp`.
- **#636 eröffnet** (Qwen3.5-MTP unter PP, drei gestapelte Ursachen: fehlendes SupportsPP, Weiche nach Ziel-PP-Rang, geteiltes Embedding unter PP nie ersetzt). PP2 RTX 61,05 tok/s, Text bitgleich. Text `upstream-contrib/03-1cat-issues/pr-qwen3-5-mtp-pp.md`. Der llama-swap-Eintrag `Qwen3.8-27B-NVFP4-vllm` (MTP) ist wieder aktiv.
- **Overlay-Rückbau `ebc5dc52`:** PP5-Warteschlangen-Deckel aus `multiproc_executor.py` und aus dem DeepSeek-Eintrag entfernt (vier Anfragen ohne Deckel, 22–26 tok/s, kein Deadlock — 1Cats PP-Spec-Transport reicht); tote Adapterklasse `QwenGatedDeltaNetAttentionUpstreamCall` (#572-Rest) raus; `qwen3_5_mtp.py` = #636-Fassung. Abnahme 27B MTP TP2 SHA gleich, 73,34 tok/s. Tag `verified-2026-09-14b`.
- **#611 rebased** auf a6f5e834 (Konflikt mit d66797cc: `Packed` als weiterer Template-Parameter neben `TurboMindLayout`/`CacheCodes`), force-gepusht `16c241ef`, Kommentar gepostet. Messung Testbranch `e2-test-0914` (Worktree `1Cat-vLLM-e2test`, #611+#604+#601, venv `.venv-pr-turing`): RTX MTP k=7 Pack 0 68,47 → auto 71,10 (+3,8 %), SHA und Annahme gleich. V100 TP2: NVFP4 läuft über den TurboMind-GEMM, Pack springt nicht an → keine V100-Aussage; ein V100-Ausreißer 58,88 tok/s mit anderem Text ist Compile-Münze. `SHARED_WEIGHT=1` (d66797cc-Pfad, TP4-Opt-in) unter TP2 nicht gemessen. 1Cat hat dasselbe Input-Layout intern auf V100 TP4 getestet (<1 ms/Runde, zurückgestellt, `docs/design/sm70_quasar_dflash2_15ms.md`). Die DFlash2-V100-Läufe im Testbranch scheitern ohne #592 (quantisierter Entwurfskopf).
- **Offen:** main ist mit #635 (79T Q8192) weiter → nächster regulärer Merge braucht Neubau; Overlay-Rückbau je Merge unserer 16 offenen PRs; Testbranch/Worktree `e2test` bis #611/#604 entschieden; nvidia/Qwen3.8-Flash-Next-NVFP4 wird geladen (Vergleichsmodell, Kuanda-Vergleich danach).

## Nachtrag 14./15.09. nachts — nvidia Flash-Next als Produktion, zwei PR-Entwürfe, max-num-seqs-Absturz

- **Checkpoint-Vergleich** (Skripte `handover/2026-09-14/flashnext_compare.sh`, `flashnext_deep.sh` + `flashnext_deep_driver.py`, Ergebnisse `flashnext_deep.out`): nvidia/Qwen3.8-Flash-Next-NVFP4 (fc694b54) gegen RadixArk-Verbund. Tempo gleichauf (13k-Decode 67–69 tok/s, Prefill 52k 681–693 tok/s), Qualität praktisch gleichwertig (nvidia kleine Vorteile Code/Regenbogen, RadixArk Zusammenfassung, beide erfinden manchmal beim Kuanda). Peuqui: nvidia. **RadixArk + provsalt-MTP-Block + Verbundverzeichnisse GELÖSCHT.**
- **Zwei Fork-Fixes, damit nvidia direkt aus dem HF-Snapshot bootet** (work-main `8f5480a7`, `1f200835`, `914333d6`, `d228c725`, Tag `verified-2026-09-14c`): (1) Checkpoint-FP8-MTP-Experten unter PP erlaubt (Speculator nur auf letztem PP-Rang), Online-Konvertierung bleibt PP=1; (2) FP8-PLE-Tabelle aus `ModelOptMixedPrecisionConfig` erkannt (nvidia setzt `ple_embedding_dtype` nicht). Beleg: Text SHA-gleich zu NVFP4-Kopf-Lauf. Außerdem wirkungsloser Qwen4Exp-Kandidat aus dem Overlay (`a7e8d9f7`, KV identisch mit/ohne).
- **PR-Entwürfe, NICHT gesendet, lokal committet:** Worktree `1Cat-vLLM-pr-fp8mtppp` (`475181c7`) und `1Cat-vLLM-pr-plemixed` (`6361129a`) auf origin/main 02c87ab8; Texte `upstream-contrib/03-1cat-issues/pr-qwen4exp-fp8-mtp-experts-pp.md`, `pr-qwen4exp-ple-fp8-modelopt-mixed.md`. Dublette geprüft (#553 = FP16-Konvertierung TP4, kein PP/PLE). Warten auf Go.
- **Produktionseintrag** `Qwen3.8-Flash-Next-180B-A4B-NVFP4-MTP-vllm` (+ -tts-qwen3local, -vlm-qwen3vl4b, -tts-qwen3local-vlm-qwen3vl4b): Betriebspunkt vom RadixArk-Eintrag, aber **`--max-num-seqs 4` statt 1**. Mit 1 stürzt der TurboMind-GEMM-Tuner auf der V100-Stufe (FP8-MTP-Experten) beim Profiling ab: `[TM][FATAL] kernels/gemm/tuner/measurer.cu(83) illegal memory access`, auch mit `CUDA_LAUNCH_BLOCKING=1` im Tuner selbst. Halbierung (`flashnext_bootprobe.sh`, `bootprobe.out`): Prefix-Caching allein ok, Async-Scheduling allein ok, max-num-seqs 1 allein → Absturz. **Issue-Kandidat für 1Cat, Ursache im Kernel nicht gefunden.** Warm mit 4: KV 645.599 Token (5,32 GiB/Rang V100-Stufe), Quantenfrage 49,7 tok/s (RadixArk-Prod am 14.09. 45,4), Start 402 s warm / 736 s kalt.
- **AIfred** (uncommittet, 900 Tests grün): Betriebspunkt-Profil `data/operating_points/Qwen3.8-Flash-Next-180B-A4B-NVFP4-MTP-vllm.yaml` (Eintrag 1:1, Hardware-Fingerprint); Autoscan dedupliziert vLLM-Seeds jetzt auch am `--model`-Pfad und setzt „-MTP“ bei Checkpoints mit MTP-Block (`tests/test_autoscan_vllm_seed.py`); Kalibrier-Matrix: Basiszelle „Kein VLM / Kein TTS“ zeigt den Betriebspunkt. Die übrigen Matrixzellen sind Seitenkanal-Burn-ins (VLM+TTS auf GPU 4), modellunabhängig, nie gemessen.
- **Befund Kaltstart:** Der erste Start nach Merge/gelöschtem Compile-Cache kompiliert kalt und bekommt nur ~halben KV-Cache (Flash-Next 407k/312k statt 747k/646k); ab dem zweiten Start korrekt.
- **Caches am 15.09. ~00:30 gelöscht** (torch_compile_cache 22 GB, vllm-calibration 7 GB, torchinductor), danach nvidia-Eintrag einmal kalt vorgewärmt. `~/.cache/mtp-diagnostics` (3,6 MB, Belege) bleibt.
- **15.09. früh:** max-num-seqs 2 läuft ebenfalls (warm 646.993 Token, nur +0,2 % gegenüber 4) → Eintrag bleibt auf 4 (parallele Hintergrundanfragen). Tuner-Absturz als **Issue #641** bei 1Cat eingereicht (Text `upstream-contrib/03-1cat-issues/issue-641-tm-tuner-mns1.md`), nicht weiter debuggt (Peuqui). Wieder aufgreifen, falls größere Modelle an Speichergrenzen stoßen. PRs #639/#640 gesendet.
