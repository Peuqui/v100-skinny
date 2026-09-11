# Übergabe — Stand 11.09.2026 nachts

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

1. **Täglicher Upstream-Check und Reaktionen auf die acht PRs.** `git fetch`
   in `1Cat-vLLM`, `gh pr list --author Peuqui --state open`, Kommentare
   lesen, AGENTS.md vor jeder Antwort frisch lesen. Merges kommen in ein bis
   zwei Tagen.
2. **Nach jedem Merge den Overlay-Teil entfernen** (Regel oben), danach die
   übliche Abnahme (`handover/2026-09-11/scripts/abnahme2/driver.sh` als
   Vorlage: 27B-MTP, DFlash2 beide Paare, Flash-Next 3×, DeepSeek).
3. **Punkt 13, NCCL-Schalter gegen den AllReduce.** 31,8 % des DFlash2-
   Decode-Profils, reine Env-Experimente:
   `handover/2026-09-11/scripts/nccl_sweep.sh <DEVS> <tag> base NCCL_PROTO=Simple
   NCCL_PROTO=LL NCCL_PROTO=LL128 NCCL_ALGO=Ring NCCL_ALGO=Tree
   NCCL_BUFFSIZE=1048576` — fünf DFlash2-Läufe je Variante, Median, Annahme,
   SHA muss konstant bleiben. Erst RTX-Paar `0,2`, dann `1,3`.
4. **Punkt 15, Skinny-Build pro Architektur.** Patch fertig, nicht angewendet:
   `handover/2026-09-11/patches/punkt15_skinny_per_arch.diff` (`marlin.py`,
   Name `skinny_nvfp4_v11_sm{cc}`). Abnahme: DFlash2 V100 SHA-gleich, RTX
   SHA + Tempo + `cuobjdump -lelf` auf der gebauten `.so` (muss sm_75 zeigen).
   Erwartet 1–6 % bei M ≤ 4, nichts bei M=8.
5. **Paket G als PRs:** NCCL-Untergruppen bekommen `--distributed-timeout-
   seconds` (`distributed/parallel_state.py`, kalter PP-Boot riss am 600-s-
   Wachhund); erzwungenes `VLLM_DISABLE_COMPILE_CACHE=1` für den 0DOT3-Graph
   entfernen (`config/vllm.py`, Ursache seit #536 behoben). Beide klein,
   Muster wie #599/#600: Worktree auf `origin/main`, Test, pre-commit +
   mypy-3.10, Duplikatsprüfung am Tag, Body aus Entwurf + Checkliste.
6. **Punkt 14, Decode-Profil mit NVFP4-Entwurfskopf:** `prof_dflash.sh <name>
   0,2` mit `DRAFT=<maurienne>`. Zeigt, was nach Block-Pack und Kopfwechsel
   von den 435 ms übrig ist (`STAND.md` Punkt 13).
7. **Punkt 10, TileLang-Pin testen:** venv `.venv-sm70-tltest` (0.1.14) steht.
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
10. **DeepSeek-Eintrag einmal ein Werkzeug aufrufen lassen.** Regel vom 11.09.:
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
