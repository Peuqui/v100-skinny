# Übergabe — Stand 10.09.2026 abends

**Betriebsstand steht in `STAND.md`. Damit anfangen, nicht mit diesem Dokument.**
Hier steht nur, was als Nächstes ansteht und was du wissen musst, um nicht
dieselben Wege noch einmal zu gehen.

---

## Wo wir stehen

**Seit 10.09. abends läuft die Produktion auf `work-main`**: 1Cat
`origin/main` plus unsere offenen PRs plus v100-skinny-Overlay, editable aus
dem Worktree `1Cat-vLLM-work`, venv `.venv-sm70-main` hinter
`~/vllm/venv`. Alt gegen neu am selben Tag abgenommen — 27B bitgleich auf
beiden Kartenpaaren, DeepSeek byteidentisch, Flash-Next kohärent; Details und
Baurezept in `STAND.md`, Abschnitt „Laufzeitumgebung". Neu dabei: beide
FA2-Bibliotheken nebeneinander, pro Gerät geladen — die V100 hat damit
erstmals 1Cats d256-Prefill-Ops. **Den Worktree nicht für PR-Branches
benutzen, er ist die Produktion.**

Später am Abend: alle produktiven llama-swap-Einträge kalt und warm geprüft
(27B-MTP, Flash-Next, NEU 27B-DFlash2 — alle grün; DeepSeek-vLLM bootete nie,
`STAND.md` Punkt 17), die alten venvs 130/150 gelöscht, der abgenommene
Stand als `verified/volta-turing` + Tag `verified-2026-09-10` im Fork.
**Nächster Schritt:** die drei fehlenden DFlash2-PRs von 1Cat mergen und neu
abnehmen (`STAND.md` Punkt 18).

DFlash2 auf dem 27B, 2× RTX 8000: **69,13 → 77,13 tok/s** in zwei Schritten,
Text-SHA in jedem Lauf `0106659946c064b1`. Die RTX liegt jetzt **vor** der
V100 (76,33), trotz 34 % weniger Bandbreite.

| Schritt | Commit | Beleg |
|---|---|---|
| Block-Pack der Aktivierungen | `7e83385` | `STAND.md` Punkt 8 |
| quantisierter Entwurfskopf + Kontext-K/V-Override | `7ca6e20` | `STAND.md` Punkt 10 |
| 1Cat-PR #592 für den Kontext-K/V-Fix | `be2ba891` im 1Cat-Checkout | `upstream-contrib/03-1cat-issues/pr-dflash-quantized-draft-context-kv.md` |

Aufrufzeile für den schnellsten Stand: `STAND.md`, Abschnitt „Qwen3.8-27B-NVFP4
mit DFlash2".

---

## Auftrag, in dieser Reihenfolge

### 1. PR-Pakete mit Peuqui besprechen — `STAND.md` Punkt 15

Peuqui will die Reihenfolge **genauer besprechen** — nicht eigenmächtig
anfangen. Die Faktenlage vom 10.09. steht in Punkt 15: was ein neuer User aus
1Cat plus unseren offenen PRs bekäme (nicht unser System), was Skinny bringt
(RTX: Marlin als Ersatz 37 % langsamer), dass TileLang den Geräte-Fix seit
v0.1.12 selbst hat (also Pin-Bump bei 1Cat statt TileLang-PR), und dass
1Cat unsere QPN-Kernel schon übernommen hat, aber nur für exakt SM70.

### 2. Den neuen Stand unter Produktionsbedingungen messen — `STAND.md` Punkt 12

Die 77 tok/s sind Bench-Bedingungen: 32k Kontext, Prefix-Caching aus, 400
Token. Die Produktion fährt den vLLM-27B mit **256K und Prefix-Caching**, und
der Hauptpfad ist ohnehin **llama.cpp** mit Q8-GGUF. Zwei Fragen, bevor
irgendwer DFlash2 in llama-swap einträgt:

- Hält DFlash2 bei langem Kontext mit Prefix-Caching? 1Cat hat dazu #467
  offen (Allocator-Reset, KV-Überschätzung bei 256K).
- Schlägt es den produktiven llama.cpp-Pfad auf denselben Karten? Das ist
  das Kriterium, nach dem vLLM überhaupt in die Produktion kommt.

Die llama-swap-Konfiguration ist Peuquis Datei: händisch oder mit Sicherung
und Freigabe, nie per Skript.

### 3. Das AllReduce — `STAND.md` Punkt 13

**31,8 % der Decode-GPU-Zeit, größter Posten, nie untersucht.** 102 µs je
Aufruf für 80 KB Nutzlast ist Latenz, nicht Bandbreite; ohne P2P läuft alles
über Host-Staging. Billigster erster Versuch sind die NCCL-Schalter
(`NCCL_ALGO`, `NCCL_PROTO`, Puffergrößen) — reine Env-Experimente, jederzeit
zurückdrehbar. Messen mit `speed_dflash.sh` (exportierte NCCL-Variablen
werden durchgereicht) und dem Text-SHA als Anker.

### 4. Den unidentifizierten fp16-GEMM zuordnen — `STAND.md` Punkt 13

`cutlass_75_wmma…f16_16x16`, 5,4 %, einer je Schicht, **im Zielmodell**
(steht auch unter MTP im Profil). Ein Linear, der an den Skinny-Kerneln
vorbeiläuft. Erst zuordnen, dann entscheiden.

### 5. Kleineres

- **Gencode aus der Gerätefähigkeit** statt fest `sm_70` in
  `_get_skinny_ext()` (`fork_patches_150/marlin.py`). Hygiene: bringt
  gemessen bei M=8 nichts, bei M≤4 ein paar Prozent.
- **Block-Pack als 1Cat-PR** — gehört jetzt zu PR-Paket (b) in `STAND.md`
  Punkt 15: 1Cat fährt seine QPN-Kopie auf Turing gar nicht (Weiche exakt
  SM70), ein Angebot muss den Turing-Pfad mitbringen.
- **#592 beobachten.** work-main trägt die Basisklasse bereits; der Override
  in `fork_patches_150/qwen3_dflash2.py` betrifft nur noch die alte venv.
- **Befunde vom 10.09.** (`STAND.md` Punkt 16): breites `pgrep`-Killen in
  `flashnext_qual.sh`/`flashnext_ab.sh`, „Unknown vLLM environment
  variable" für `VLLM_SKINNY_*`, Werkzeuge mit fester `.venv-sm70-130`.
- **18 vorbestehende Ruff-Meldungen** in alten Benchmark-Skripten
  (`kernel_matched_bench.py`, `qpn8_*.py`, `v11_suite.py`). Technische
  Schulden, kein Laufzeitproblem.

---

## Entschieden — nicht neu aufrollen

- **Die Bitgleichheit wird nicht geopfert** (Peuqui, 10.09.). Damit ist die
  Geometrie-Tabelle `_QPN2_TABLE` als Hebel ganz vom Tisch: `nacc` UND
  `SPLITK` ändern beide die Summationsreihenfolge. Preis beziffert: ~0,5 %.
- **`skinny_fp8_qpn8` bleibt ungepackt.** 1,05× auf der RTX, −4 % auf der
  V100 — der Kernel ist DRAM-gebunden.
- **Keine sm75-MMA-Variante.** `m8n8k4` und `m16n8k8` sind auf der RTX bei
  gleicher FLOP-Zahl gleich schnell (`benchmarks/mma_probe.py`).

Die Erklärungen für die frühere Turing-Lücke, die gemessen widerlegt sind,
stehen in `STAND.md` Punkt 8.

---

## Werkzeuge, die jetzt da sind

- **`ncu` läuft als normaler Nutzer.** Prüfen mit
  `grep RmProfilingAdminOnly /proc/driver/nvidia/params` (muss `0` sein —
  der Treiber führt den Parameter unter DIESEM Namen). Für L1-Fragen zählt
  `l1tex__data_pipe_lsu_wavefronts_mem_lg.sum`, nicht die Sektoren.
- **`benchmarks/qpn2_pack_ab.py --ref <rev>`** — A/B zweier Kernel-Fassungen,
  Bitgleichheit plus Graph-Replay-Timing.
- **`DRAFT=`** in `speed_dflash.sh` und `prof_dflash.sh` schaltet den
  Entwurfskopf um.

## Fallen, die heute zugeschnappt sind

Am Abend dazugekommen, alle in `STAND.md` unter „Fallstricke": der 1Cat-Bau
nur mit `TORCH_CUDA_ARCH_LIST=7.0`, CCCL per `CPATH`, 1Cats editable Bau,
FA2-Bibliotheken nie beim Import laden, und keine Kommentarzeile in eine
Backslash-Kette (hat einen Flash-Next-Lauf mit alter venv und ohne MTP
gestartet, bemerkt nur über `speculative_config=None`).

Vom Vormittag, ebenfalls dort — die zwei, die am meisten gekostet hätten:

- **`import vllm` mit cwd im 1Cat-Checkout** findet das lokale Verzeichnis,
  nicht die venv. Der Checkout trägt einen Symlink `vllm/_C.abi3.so` auf
  `.venv-sm70-150`; deshalb prüft pytest mit cwd im Checkout wirklich den
  Checkout-Code. Aus neutralem Verzeichnis nachprüfen, woher `vllm` kommt.
- **Einen Helfer nie zwischen `@support_torch_compile` und die Klasse
  schieben.** Der Dekorator landet dann auf der Funktion, und die Tests
  sammeln nicht einmal.
