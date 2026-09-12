# Übergabe — Stand 12.09.2026 früh (autonome Nacht nach dem 11.09.)

**Ergebnis der Nacht in einem Absatz:** Punkte 1, 3, 4, 6, 7 des Plans sind
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
| `1Cat-vLLM-work` (work-main) = Produktion | `433dfa10` = Tag `verified-2026-09-11b`, auf `fork/verified/volta-turing` | 1Cat main `fe67339d` + fünf offene PRs + Overlay; Merge-Reste bereinigt (`43ccb9b8`), **E5 ausgebaut**, Befund 2 `index_share=True` |
| `1Cat-vLLM-pr-dflash2` | `90c9efee` | PR #599 |
| `1Cat-vLLM-pr-devcap` | `2b59521a` | PR #600 |
| `1Cat-vLLM-editable-pr` | `e5e3e9af` (+ `build/` 2,1 GB für inkrementelle Nachbauten) | PR #601 |
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
3. **`NCCL_BUFFSIZE=1048576`** in die vLLM-TP-Einträge von llama-swap (+1 %).
4. **envs.py-Default im Fork** zurücknehmen (Compile-Cache auf 0DOT3-Pfad
   ist in Produktion aus) — Overlay-Änderung, unabhängig vom PR.
5. ~~TileLang-Pin~~ — ENTSCHIEDEN 12.09. früh (Peuqui): als Wartung
   übernehmen, kein Leistungsziel. Offen bleibt nur das Wann des Postens.
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
8. **SWA-Schwellen-Fix committen und als PR eröffnen** (Entwurf
   `pr-sparse-swa-spec-threshold.md`, Worktree `1Cat-vLLM-pr-swathreshold`).
   Overlay in work-main ist angewendet, damit der DeepSeek-Eintrag in AIfred
   ab sofort stabil ist — sollte in den nächsten work-main-Commit.
