# Übergabe — Stand 11.09.2026 vormittags

**Betriebsstand steht in `STAND.md`. Damit anfangen, nicht mit diesem Dokument.**
Hier steht, was als Nächstes ansteht, was halb fertig ist und was du wissen
musst, um nicht dieselben Wege noch einmal zu gehen. Alle Skripte, Patches,
Referenz- und Ergebnisdateien dieser Runde liegen in
`handover/2026-09-11/` (siehe dortige `README.md`).

---

## Erledigt 11.09. abends: Aufräumen abgenommen und committet

`43ccb9b8` auf work-main, Tag `verified-2026-09-11`, gepusht nach
`fork/verified/volta-turing`. Ergebnisse in `STAND.md`, Punkt 16. Der
Abschnitt darunter ist damit Geschichte und bleibt nur als Beleg stehen.

## (erledigt) Halb fertige Änderung in der Produktion

Der Worktree `1Cat-vLLM-work` **ist die Produktion** (editable in
`.venv-sm70-main` hinter `~/vllm/venv`). Darin lagen seit 11.09. früh
Aufräum-Patches aus der Overlay-Inventur
(`upstream-contrib/OVERLAY-INVENTUR.md`, Befunde 1, 3, 4, 5, 6, 7, 8), in
7 Dateien:

    vllm/_custom_ops.py  vllm/config/vllm.py  vllm/model_executor/models/config.py
    vllm/v1/core/kv_cache_utils.py  vllm/v1/core/single_type_kv_cache_manager.py
    vllm/v1/worker/gpu/model_states/mamba_hybrid.py  vllm/v1/worker/gpu_model_runner.py

AIfred läuft wieder (Peuqui hat neu gestartet), llama-swap hatte um 09:46
`Qwen3.8-27B-NVFP4-vllm` geladen, also **mit diesem Code**.

Die Abnahme (`handover/2026-09-11/scripts/cleanup_accept.sh`) wurde vom
Sitzungsende unterbrochen. Stand (`ergebnisse/cleanup_accept.out`):

| Modell | Ergebnis | gegen Referenz |
|---|---|---|
| 27B-MTP (V1-Runner), Produktionsbefehl | 98 Token, Antwortanfang gleich | exakter Textvergleich **nicht gelaufen** — nachholen gegen `scripts/prod/Qwen3.8-27B-NVFP4-vllm.p1.json` |
| 27B DFlash2 RTX | SHA `0106659946c064b1`, 76,96 tok/s, Annahme 3,325 | gleich (Ref 76,65 / 3,325) |
| 27B DFlash2 V100 | SHA `0106659946c064b1`, 76,99 tok/s, Annahme **3,353** | SHA gleich; Ref hatte Annahme 3,325 bei 76,28 — **ungeklärt**, gleicher Kopf (maurienne) und `QUANT_LM_HEAD=1` in beiden boot.logs |
| Flash-Next heterogen | q1/q2 kohärent, **q3 verfehlt Coandă** („Kuanda-Effekt", `ergebnisse/flashnext_clean_q3_kuanda.txt`), SHAs alle anders | siehe unten |
| DeepSeek PP5 | **nicht gelaufen** | `scripts/ds_accept.sh` gegen `referenz/ds_main.out` |

**Flash-Next richtig bewerten:** Die SHAs sind bei Flash-Next nie stabil
gewesen — schon am 10.09. unterschieden sich alte und neue venv in allen drei
Hashes (`referenz/fnq_ref150.out` gegen `referenz/fnq_main_hetero.out`); das
Kriterium war damals „3/3 kohärent". q3 (Coandă) ist ein **bekannter
sporadischer Aussetzer** (09.09.; die alte venv hatte am 10.09. ebenfalls 2/3).
Außerdem stammt die Flash-Next-Referenz von 17:41, der Upstream-Merge
`82301e6b` (#586/#587/#589) kam erst 19:49 — **Flash-Next wurde nach dem Merge
nie abgenommen**. Ein Einzellauf entscheidet also nichts. Vorgehen:
`flashnext_qual.sh <tag> 4` mindestens dreimal auf dem jetzigen Stand und
dreimal mit zurückgenommenem Aufräumen, Kohärenzquote und Annahmelängen
vergleichen. Befund 6 (`kv_cache_utils.py`) ist der einzige Aufräumpunkt, der
Qwen4Exp überhaupt berührt — die Zuteilung sollte identisch sein, das ist aber
nicht gemessen.

Zurücknehmen: `git -C 1Cat-vLLM-work checkout HEAD -- <die 7 Dateien>`.
Wieder anwenden: `git apply handover/2026-09-11/patches/cleanup_neutral.diff
handover/2026-09-11/patches/cleanup_befund6.diff` (im Worktree).

Nebenbefund: 4 Tests in `tests/v1/core/test_kv_cache_utils.py` sind rot —
**auch auf frischem 1Cat-main**, also Upstream (`test_estimate_max_model_len`
16385/16383, `test_get_max_concurrency_for_kv_cache_config`,
`test_deepseek_v4_tuple_width_minimizes_physical_pool_pages`). Kandidat für
einen Hinweis an 1Cat.

---

## Aufträge mit Freigabe (Peuqui 10./11.09.)

### 1. ~~Abnahme des Aufräumens abschließen~~ — ERLEDIGT 11.09. abends

Alle vier Modelle abgenommen, V100-Annahmelänge geklärt (Einzelausreißer),
committet als `43ccb9b8` auf Ansage. Treiber und Ergebnisse:
`scratchpad/abnahme2/` der Sitzung vom 11.09. nachmittags (driver.sh,
driver.log) — bei Bedarf nach `handover/` sichern.

### 2. Befund 2: `index_share_for_mtp_iteration` (freigegeben)

Unser doppelter Qwen4Exp-MTP-Block in `vllm/config/speculative.py` setzt den
Wert auf `False`; Upstream (70b63a1e „Reduce Qwen4Exp MTP cost") setzt `True`.
Flash-Next läuft zwingend über Model Runner V2, dessen Speculator nur bei
`True` die QSA-Indizes aus Schritt 0 für die MTP-Schritte 1+ wiederverwendet.
Patch: `patches/befund2.diff` — danach ist die Datei gleich Upstream plus
DeepSeek-V4-Guard (behebt auch das fehlende Zeilenende am Dateischluss).
A/B auf Flash-Next: Annahmelänge und tok/s über mehrere Läufe; der Text
sollte bei greedy gleich bleiben (der Zielkopf prüft jeden Token), aber
Flash-Next ist ohnehin nicht hash-stabil — Kohärenz zählen.

### 3. Befund 9: E5-Cache — Entscheidung bei Peuqui offen

Peuqui fragte „was schlägst du vor?"; Vorschlag war **ganz ausbauen**: keiner
der sieben vLLM-Einträge nutzt ihn (`VLLM_SM70_E5_CACHE=0` überall, Vorgabe im
Code aber `"1"`), Upstream hat ihn entfernt, auf QSA stürzte er ab (#413),
~900 Zeilen in der konfliktträchtigsten Datei, die zwei offenen ruff-Fehler
(`gpu_model_runner.py:551` E501, `:1264` SIM105) liegen darin. Betrifft nur
den V1-Runner → Abnahme 27B-MTP und DeepSeek. **Nicht ohne Antwort
anfangen.** 258 E5-Zeilen, Einstiege um `execute_model` (~Z. 9599–10493) und
der Block ab Z. 433; `only_gids` in `_build_attention_metadata` gehört auch
dazu.

### ERLEDIGT 11.09. abends: drei PRs eröffnet

#599 (Paket 11a, beide Gerät-0-Gates in `qwen3_dflash2.py`), #600 (Wurzelfix:
`device_id=None` → aktuelles Gerät des Workers, torch statt NVML nach
CUDA-Init; erledigt die Klasse aus #412 für alle 263 Aufrufstellen), #601
(Editable-Bau, Wheel gebaut und abgenommen). Belege und Ergebnisse in
`upstream-contrib/03-1cat-issues/pr-*.md` und `handover/2026-09-11/`.
Damit sind die Abschnitte 4 und 5a unten Geschichte; 5c–5e bleiben, wobei
die Gerät-0-Anteile von 5e (D2) durch #600 entfallen. Offen: E5-Entscheidung
(Messung liegt vor, STAND.md Punkt 16), Befund 2 index_share, Punkte 6–10.

### 4. ~~Editable-PR (Punkt 8) — fertig, wartet auf Peuquis Go~~ → #601

Branch `editable-soabi-modules` im Worktree `1Cat-vLLM-editable-pr` auf
origin/main fe67339d, `setup.py` +14/−6. Entwurf:
`upstream-contrib/03-1cat-issues/pr-editable-soabi-modules.md`.
Belegbau auf frischem main: Exit 0 nach 52 min, alle fünf Module laden
(`ergebnisse/pr_editable_build.log`). Kernargument: #319/#320 lösten dasselbe
für `_sm70_sampler_C` per `USE_SABI 3`; das geht bei unseren fünf nicht, sie
sind pybind11. Commit als Peuqui mit `Co-authored-by: Claude` +
`Signed-off-by: Peuqui <peuqui@github.com>`, auf `fork` pushen, PR-Text ab
„## Purpose". Nachhilfen 2/3 (flash_attn_v100 im Editable-Finder,
GDN-Erweiterung nur in build_lib) sind als Folge-PR angeboten.

### 5. Paket 11 „gemischte Hardware" als PRs zu 1Cat

In unserem System ist alles davon drin und läuft; „fixen" heißt: für 1Cat
aufbereiten. Stand:

- **a) DFlash2-Emulation für alles unter SM80, am Worker-Gerät** — Worktree
  `1Cat-vLLM-pr-dflash2`, Branch `dflash2-pre-sm80-worker-device`,
  Entwurf `upstream-contrib/03-1cat-issues/pr-dflash2-pre-sm80-worker-device.md`.
  CPU-Test (7 Fälle) grün, Gegentest ohne Fix 3 rot, pre-commit + mypy-3.10
  grün, Duplikatsprüfung erledigt. **Fehlt:** GPU-Messung auf main + #572
  (Turing bootet ohne #572 nicht korrekt), mit und ohne Fix, RTX-Paar
  (früher gemessen: Annahme 1,015 → 3,353); die zwei CUDA-Fälle
  `test_sm70_dflash2_exact_rerank_matches_gathered_bmm[7/8]` auf einer
  **freien** V100, main gegen Fix (der erste Lauf lief neben einem belegten
  Modell).
- **b) `fa_utils.py`** — gestrichen: unterscheidet nur Hopper/Blackwell, auf
  Volta/Turing ohne Wirkung und hier nicht belegbar.
- **c) FA2 auf Turing** (`flash_attn_interface.py`-Lader pro Gerät,
  `flash_attn.py`-Gate, `cuda.py`-Priorität, sm75-FA2-Bibliothek) — groß:
  1Cat müsste unseren FA-Fork mitbauen wie zhinianqins V100-FA. Erst mit
  Peuqui besprechen.
- **d) Triton-3D-Split-KV** (`triton_unified_attention.py`, `triton_attn.py`)
  — vorher klären, welche Konfiguration noch über Triton-Attention läuft.
- **e) DeepSeek D1/D2** (FP8-Gates „< SM89" statt „== SM70",
  Fähigkeit worker-lokal) — Testumgebung auf main ist aufwendig, später.

**Frisches main auf der echten Hardware fahren:** `1Cat-vLLM-editable-pr`
enthält nach dem Belegbau alle `.so`. Mit `PYTHONPATH=<Worktree>` und der
Produktions-venv lädt vLLM dann den Worktree-Code (vorher per
`vllm.__file__` prüfen). Für andere PR-Worktrees die `.so` von dort
verlinken — **nicht im Haupt-Checkout `1Cat-vLLM` bauen**: dessen
`vllm/*.abi3.so` sind Symlinks auf die Produktion, ein Inplace-Bau schriebe
durch sie hindurch.

### 6. Punkt 15: Skinny-Build pro Architektur

Patch fertig, nicht angewendet: `patches/punkt15_skinny_per_arch.diff`
(`_get_skinny_ext` in `marlin.py`: Name `skinny_nvfp4_v11_sm{cc}`, Gencode
aus der Fähigkeit des Worker-Geräts). Abnahme: 27B DFlash2 auf der V100
SHA-gleich, auf der RTX SHA, Tempo und `cuobjdump -lelf` auf
`~/.cache/torch_extensions/py312_cu128/skinny_nvfp4_v11_sm75/*.so` (muss
sm_75 zeigen). Gemessen erwartet: 1–6 % bei M ≤ 4, bei M=8 nichts.

### 7. Punkt 13: NCCL-Schalter gegen den AllReduce

`scripts/nccl_sweep.sh <DEVS> <tag> base NCCL_PROTO=Simple NCCL_PROTO=LL
NCCL_PROTO=LL128 NCCL_ALGO=Ring NCCL_ALGO=Tree NCCL_BUFFSIZE=1048576` —
fünf DFlash2-Läufe je Variante, Median, Annahme, SHA (muss konstant bleiben:
die Summe zweier Ränge ist protokollunabhängig). Erst RTX-Paar (`0,2`), dann
die besten auf `1,3`.

### 8. Punkt 14: fp16-GEMM — zugeordnet, Rest nachmessen

Aus den vorhandenen nsys-Berichten (`prof_packed`, `prof_mtpctl`): Der
„unidentifizierte" Kernel ist überwiegend der **unquantisierte Entwurfskopf**
(damals fp16-DFlash2-Kopf; unter MTP der fp16-MTP-Block des
RadixArk-Checkpoints, `exclude_modules: mtp*`). Im Zielmodell nur
`in_proj_ba`, ~0,6 %. Details `STAND.md` Punkt 13. Offen: neues Profil mit
dem NVFP4-Kopf (`prof_dflash.sh <name> 0,2` mit `DRAFT=<maurienne>`), um zu
sehen, was übrig ist. Hebel für 27B-MTP wäre ein quantisierter MTP-Block
(wie Flash-Next MTPQ).

### 9. Punkt 10: TileLang-Pin-Bump testen

Test-venv `v100-skinny/.venv-sm70-tltest` (Kopie der Produktion, 11 GB) mit
`tilelang==0.1.14` und `apache-tvm-ffi==0.1.12` — Installation von Peuqui
freigegeben, Produktion unberührt auf 0.1.10. Import-Prüfung erledigt: alle
fünf genutzten TileLang-APIs vorhanden; der Geräte-Fix sitzt jetzt in
`tilelang/cuda/target.py` (`current_device()`), `tilelang/utils/target.py`
gibt es nicht mehr — unser Overlay-Patch `fork_patches_150/tilelang_target.py`
wird damit überflüssig. **Fehlt (GPU):** `tests/kernels/test_mhc_kernels.py`
und `test_mhc_sm70_fp16.py` je auf einer V100 und einer RTX; DeepSeek PP5 mit
`VENV=.../.venv-sm70-tltest` über `ds_accept.sh`; ein GDN-Modell booten
(`flash_qla` importiert TileLang beim Laden). Danach Vorschlag an 1Cat: beide
Pins anheben (`tilelang` und `apache-tvm-ffi`, 1Cat pinnt beide auf 0.1.10).

### 10. Punkt 9: Routenzähler, dann Skinny-Angebot

`VLLM_SKINNY_ROUTE_COUNT_FILE` in den Produktionskonfigurationen laufen
lassen; dann (a) Turing-Pfad für 1Cats QPN-Kopie, (b) Block-Pack, (c)
MoE-Backend erst nach Rückfrage in #441. `VLLM_SKINNY_*` in dem PR in
`envs.py` anmelden (Peuqui 10.09.).

### 11. llama-swap

- DeepSeek steht auf 64k (erledigt, Sicherung vorhanden). Mehr als 64k ist
  knapp (~80 MiB Luft bei ~98k gerechnet); nur auf Wunsch versuchen — ein
  OOM mitten in der Antwort reißt den ganzen Server um.
- **Erledigt 11.09. mittags** (Freigabe Peuqui, Sicherung
  `backups/config.yaml.bak-2026-09-11-vor-toolparser`): die zwei Müll-Einträge
  `Qwen3.8-27B-DFlash2-vllm` und `Qwen3.8-27B-DFlash2-NVFP4-RTNcal-vllm`
  samt Gruppenmitgliedschaft entfernt (der Autoscan legt sie mit
  `is_draft_head` nicht mehr neu an); alle Qwen3.8-vLLM-Einträge auf
  `--tool-call-parser qwen3_coder --reasoning-parser qwen3` — vorher `hermes`,
  damit hat vLLM jeden Tool-Call still verschluckt (STAND.md, Fallstricke).
- Datei ist Peuquis: händisch oder mit Sicherung und Freigabe, nie per Skript
  im Blindflug; `--watch-config` lädt Änderungen selbst.

### 12. Unkommittiert (nur auf „commit")

| Repo / Worktree | Inhalt |
|---|---|
| `1Cat-vLLM-work` (work-main) | ~~Aufräum-Patches~~ — committet `43ccb9b8`, gepusht 11.09. abends |
| `1Cat-vLLM-editable-pr` | `setup.py` (PR 8) |
| `1Cat-vLLM-pr-dflash2` | `qwen3_dflash2.py` + neuer Test (PR 11a) |
| `v100-skinny` (work) | `STAND.md`, `HANDOVER.md`, pgrep-Fix in `flashnext_qual.sh`/`flashnext_ab.sh`, `upstream-contrib/OVERLAY-INVENTUR.md`, zwei PR-Entwürfe, `handover/` |
| `AIfred-Intelligence` (main) | `scripts/llama-swap-autoscan.py` (`is_draft_head` + Parser aus dem Template); Tool-Parser-/Reasoning-Fix 11.09.: `aifred/backends/{base,llamacpp,vllm}.py`, `aifred/lib/calibration/{vllm_probe,vllm_model_meta,vllm_flow}.py`, `tests/test_vllm_calibration.py`, neu `tests/test_openai_backend_stream.py`; dazu vLLM-Fußzeile bei Tool-Runden (Zählerbasis direkt vor jeder Serveranfrage, `_before_stream_request`) |
| `v100-skinny` (zusätzlich) | `scripts/serve-qwen38-mini.sh`, `scripts/serve-qwen38-native.sh` (`qwen3_coder`) |

**Achtung beim Committen in v100-skinny:** `.venv-sm70-tltest/` (11 GB) steht
nicht in `.gitignore` (dort sind die venvs einzeln aufgeführt). Vorher
eintragen (mit Peuqui) oder nur gezielt Dateien adden, nie `git add -A`.

---

## Fallen dieser Runde (teuer erkauft)

- **Kein eigener Python-/pytest-Aufruf ohne `CUDA_VISIBLE_DEVICES=""`,
  solange ein Modelltest läuft.** Ein pytest belegte 398 MiB auf der V100 von
  DeepSeek-PP1 → OOM, Lauf verfälscht.
- **`pgrep -f <muster>` trifft die eigene Shell**, wenn deren Befehlszeile
  das Muster enthält. PID über `ps -eo pid,comm,args` mit Filter auf `comm`
  holen und gezielt beenden.
- **DeepSeek-Speicher:** nutzbar pro V100 31,73 GiB (nicht 32.768 MiB);
  FlashMLA-Sparse reserviert beim Start 5 × max-model-len × 576 × 2 Byte —
  „zu hohe max-model-len, Wert aus dem Fehler ablesen" funktioniert hier
  nicht, der Profillauf stirbt vorher. Mit `--num-gpu-blocks-override`
  rechnet vLLM die KV-Prüfung gegen Blöcke × Bytes, nicht gegen Messung.
- **Merge-Reste finden:** hinzugefügte Zeilen gegen die Upstream-Fassung
  derselben Datei abgleichen (Methode in `OVERLAY-INVENTUR.md`). „theirs +
  Fork-Hunks konfliktfrei" hieß mehrfach: wir fügten hinzu, was Upstream schon
  hatte.
- **Behauptungen im PR-Text belegen:** Fehlermeldungen wörtlich aus einem
  echten Log zitieren und die Herkunft nennen; nicht Wiederholtes als solches
  kennzeichnen.

## Entschieden — nicht neu aufrollen

- **Die Bitgleichheit wird nicht geopfert** (Peuqui, 10.09.). `_QPN2_TABLE`
  als Hebel vom Tisch: `nacc` UND `SPLITK` ändern die Summationsreihenfolge.
- **`skinny_fp8_qpn8` bleibt ungepackt** (1,05× RTX, −4 % V100, DRAM-gebunden).
- **Keine sm75-MMA-Variante** (`m8n8k4` = `m16n8k8` auf der RTX).
- **DeepSeek-V4-Flash (alt) bleibt**, V4.1 passt nicht auf die Hardware.

## Werkzeuge

- `ncu` als normaler Nutzer (`RmProfilingAdminOnly` muss `0` sein).
- `benchmarks/qpn2_pack_ab.py --ref <rev>` — Kernel-A/B mit Bitgleichheit.
- `DRAFT=` in `speed_dflash.sh` und `prof_dflash.sh` schaltet den Entwurfskopf.
- `handover/2026-09-11/scripts/dsv4_ctx_cycle.sh` — ein Zyklus Kontextsuche
  (Boot, VRAM-Protokoll, Langprompt, Abräumen der eigenen Prozessgruppe).
