# Übergabe — Stand 10.09.2026

**Betriebsstand steht in `STAND.md`. Damit anfangen, nicht mit diesem Dokument.**
Hier steht nur, was als Nächstes ansteht und was du wissen musst, um nicht
dieselben Wege noch einmal zu gehen.

---

## Wo wir stehen

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

### 1. Den neuen Stand unter Produktionsbedingungen messen — `STAND.md` Punkt 12

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

### 2. Das AllReduce — `STAND.md` Punkt 13

**31,8 % der Decode-GPU-Zeit, größter Posten, nie untersucht.** 102 µs je
Aufruf für 80 KB Nutzlast ist Latenz, nicht Bandbreite; ohne P2P läuft alles
über Host-Staging. Billigster erster Versuch sind die NCCL-Schalter
(`NCCL_ALGO`, `NCCL_PROTO`, Puffergrößen) — reine Env-Experimente, jederzeit
zurückdrehbar. Messen mit `speed_dflash.sh` (exportierte NCCL-Variablen
werden durchgereicht) und dem Text-SHA als Anker.

### 3. Den unidentifizierten fp16-GEMM zuordnen — `STAND.md` Punkt 13

`cutlass_75_wmma…f16_16x16`, 5,4 %, einer je Schicht, **im Zielmodell**
(steht auch unter MTP im Profil). Ein Linear, der an den Skinny-Kerneln
vorbeiläuft. Erst zuordnen, dann entscheiden.

### 4. Kleineres

- **Gencode aus der Gerätefähigkeit** statt fest `sm_70` in
  `_get_skinny_ext()` (`fork_patches_150/marlin.py`). Hygiene: bringt
  gemessen bei M=8 nichts, bei M≤4 ein paar Prozent.
- **Block-Pack als 1Cat-PR** — nur als Portierung auf deren eigene
  `nvfp4_qpn2_sm70.cu`, die dieselbe Zeilenstreuung hat (`STAND.md`
  Punkt 14). Vorher klären, ob 1Cat den Kernel auf Turing überhaupt fährt.
- **#592 beobachten.** Nach dem Merge den Override aus
  `fork_patches_150/qwen3_dflash2.py` entfernen.
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

Alle stehen in `STAND.md`, Abschnitt „Fallstricke". Die zwei, die am meisten
gekostet hätten:

- **`import vllm` mit cwd im 1Cat-Checkout** findet das lokale Verzeichnis,
  nicht die venv. Der Checkout trägt einen Symlink `vllm/_C.abi3.so` auf
  `.venv-sm70-150`; deshalb prüft pytest mit cwd im Checkout wirklich den
  Checkout-Code. Aus neutralem Verzeichnis nachprüfen, woher `vllm` kommt.
- **Einen Helfer nie zwischen `@support_torch_compile` und die Klasse
  schieben.** Der Dekorator landet dann auf der Funktion, und die Tests
  sammeln nicht einmal.
