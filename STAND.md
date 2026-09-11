# Betriebsstand v100-skinny

**Stand 2026-09-11 abends.** Dieses Dokument beschreibt, WIE der Stack heute
läuft. Warum er so läuft, steht in `docs/journal/` — jede Zeile hier trägt einen
Verweis. Übergabeaufträge stehen in `HANDOVER.md`, Upstream-Beiträge in
`upstream-contrib/`.

Reihenfolge für eine neue Instanz: dieses Dokument, dann `HANDOVER.md`. Die
Logbücher im Journal sind chronologisch und groß — sie beantworten „warum ist es
so", nicht „wie ist es".

---

## Laufzeitumgebung (seit 10.09. abends)

Produktion und alle Messskripte laufen über den Symlink **`/home/mp/vllm/venv`**
→ `venv-main` → **`.venv-sm70-main`** (Python 3.12, torch 2.10.0+cu128, siehe
`~/vllm/README.md`). Darin ist 1Cat **editable** aus dem Worktree
`/home/mp/Projekte/vllm-research/1Cat-vLLM-work` installiert, Branch
`work-main` (lokaler Arbeitszweig; abgenommene Stände gehen als
`verified/volta-turing` in den Fork Peuqui/1Cat-vLLM, jeder mit Tag —
aktuell `verified-2026-09-11` → `43ccb9b8`): 1Cat `origin/main` `fe67339d`
(Merge `82301e6b`) + unsere offenen PRs #572 #573 #574 #576 #592 +
v100-skinny-Overlay (`5099866f`) + FA2-Koexistenz und Bau-Fix (`f03a7102`)
+ Aufräumen der Merge-Reste, Befunde 1 und 3–8 (`43ccb9b8`, abgenommen
11.09. auf allen vier Produktionsmodellen, siehe offener Punkt 16).

**Regel seit 11.09. (Peuqui):** Sobald 1Cat einen PR von uns merged, wird
der zugehörige Teil des Overlays beim nächsten Hereinholen von main
ENTFERNT, nicht neu darübergelegt. Der Diff des Worktrees gegen `origin/main`
darf nur enthalten, was bei 1Cat noch nicht angekommen ist — er ist damit
zugleich die Liste der noch offenen PR-Pakete. Nach jedem Merge die
hinzugefügten Zeilen gegen die Upstream-Fassung derselben Datei abgleichen
(Methode in `upstream-contrib/OVERLAY-INVENTUR.md`); „konfliktfrei" heißt
nicht „sauber".

- **Der Worktree IST die Produktion.** Dort keine anderen Branches
  auschecken; PR-Arbeit im Haupt-Checkout `1Cat-vLLM`. Jede Python-Änderung
  im Worktree wirkt beim nächsten Serverstart. Der Versionsstring
  (`…g5099866fa…`) nennt nur den Bau-Commit.
- **FA2 auf gemischter Hardware:** zwei Bibliotheken nebeneinander in
  `vllm/vllm_flash_attn/` — `_vllm_fa2_C.abi3.so` (V100: zhinianqins FA plus
  1Cats d256-Ops, sieben `sm70_*`-Ops) und `_vllm_fa2_C_sm75.abi3.so` (unsere
  sm75-FA, Kopie aus `.venv-sm70-150`, md5 `1285b8f0e013`, gitignored).
  `flash_attn_interface.load_fa2_library(device)` lädt beim ersten Op-Aufruf
  die zur Karte passende. Beleg im Boot-Log je Rang: „Loaded FA2 library …
  for compute capability …". Vorher fehlten der V100 die d256-Ops ganz.
- **Skinny** kommt weiter aus `kernels/skinny_kernels.cu` (JIT über
  `VLLM_SKINNY_NVFP4_SRC`); `fork_patches_150/` wird für diese venv NICHT
  mehr ausgerollt — die Patches stecken im Overlay-Commit.
- **Kein Symlink-Rückweg mehr:** `.venv-sm70-130` und `.venv-sm70-150` sind
  gelöscht (10.09. abends, Freigabe Peuqui). Zurück geht es nur als Neubau
  vom Tag `verified-2026-09-10` nach dem Rezept unten. Der 1Cat-Haupt-Checkout
  holt seine kompilierten Module per Symlink aus dem Worktree
  (`1Cat-vLLM/vllm/*.abi3.so` → `1Cat-vLLM-work/vllm/`).

**Bau** (~45 min mit `MAX_JOBS=4`), aus dem Worktree:
`env -u VLLM_FLASH_ATTN_SRC_DIR CPATH=<venv>/lib/python3.12/site-packages/nvidia/cuda_cccl/include CUDA_HOME=/home/mp/vllm/cuda TORCH_CUDA_ARCH_LIST=7.0 MAX_JOBS=4 <venv>/bin/python -m pip install -e . --no-build-isolation`;
danach `pip install -e ./flash-attention-v100 --no-build-isolation --no-deps`,
die GDN-Erweiterung wie `setup.py::bundle_flash_qla_sm70` nach
`flash_qla/ops/gated_delta_rule/chunk/sm70/` legen, die sm75-FA-Datei
danebenlegen und `fork_patches_150/tilelang_target.py` nach
`tilelang/utils/target.py`. Warum jeder Schritt: Fallstricke unten.

**Abnahme 10.09., alt gegen neu am selben Tag, gleiche Karten und Skripte:**

| Test | neu (`.venv-sm70-main`) | alt (`.venv-sm70-150`) |
|---|---|---|
| 27B DFlash2, V100-Paar (`speed_dflash.sh`) | SHA `0106659946c064b1`, 76,30 tok/s | SHA gleich, 76,33 |
| 27B DFlash2, RTX-Paar | SHA gleich, 77,12 / 76,75 | SHA gleich, 77,13 |
| DeepSeek-V4 PP5, alle 5 Karten (`scripts/deepseek_coherence.py`, zwei Läufe) | 8/8, beide Läufe byteidentisch | 8/8, **byteidentisch zu neu**, gleiches Tempo |
| Flash-Next TP2×PP2 heterogen (`flashnext_qual.sh … 4`) | 3/3, 25,4 / 31,8 / 29,4 tok/s | 2/3 (q3 Coandă verfehlt), 22,3 / 27,9 / 26,4 |

Flash-Next: je ein Lauf — keine Ratenaussage, das 180B ist nicht
deterministisch (q3 ist der bekannte Aussetzer vom 09.09.). Flash-Next
berührt FA2 gar nicht: Seine Attention läuft über QSA-Triton und GDN.

**Produktive llama-swap-Einträge**, je mit exakt ihrem Befehl kalt (eigener
Port, lange Geduld) und danach warm über llama-swap selbst:

| Eintrag | kalt | warm über llama-swap | Befund |
|---|---|---|---|
| `Qwen3.8-27B-NVFP4-vllm` (MTP k=3, 256K, Prefix-Caching) | 511 s | 121 s | kohärent, beide Ränge laden die sm75-FA2 |
| `Qwen3.8-Flash-Next-180B-A4B-NVFP4-MTPQ-vllm` | 612 s | 316 s | kohärent |
| `Qwen3.8-27B-NVFP4-DFlash2-vllm` (NEU 10.09., 256K, Prefix-Caching) | 430 s | 140 s | kohärent, wortgleich mit dem MTP-Eintrag |
| `DeepSeek-V4-Flash-nvfp4-DSpark-vllm` | — | — | bootet nie, offener Punkt 17 |

---

## Hardware (gemessen 07.09.)

| GPU | PCI | Karte | sm | SMs | Shared/SM | VRAM | Anbindung |
|---|---|---|---|---|---|---|---|
| 0 | 01:00.0 | Quadro RTX 8000 | 75 | 72 | 64 KB | 47 GiB | OCuLink, 1 Hop |
| 1 | 02:00.0 | Tesla V100-PCIE | 70 | 80 | 96 KB | 31 GiB | OCuLink, 1 Hop |
| 2 | 06:00.0 | Quadro RTX 8000 | 75 | 72 | 64 KB | 47 GiB | OCuLink, 1 Hop |
| 3 | 07:00.0 | Tesla V100-PCIE | 70 | 80 | 96 KB | 31 GiB | OCuLink, 1 Hop |
| 4 | **0a:00.0** | Tesla V100-PCIE | 70 | 80 | 96 KB | 31 GiB | **USB4-Tunnel, 3 Hops** |

Alle fünf laufen **Gen3 ×4** (~3,5 GB/s), von ×16 möglich. **GPU 4 ist die
einzige am USB4-Tunnel** (Bridge `00:03.1` → `08:00.0` → `09:00.0` →
`0a:00.0`, verifiziert 07.09. per `lspci`/sysfs); die anderen vier hängen über
M.2-OCuLink direkt am Root-Complex. Der Tunnel kostet rund **5 %** — spürbar,
aber unkritisch. P2P ist aus (`NCCL_P2P_DISABLE=1`), GPU-zu-GPU läuft über den
Host-Root-Complex.

**Kartenwahl (korrigiert 07.09., Peuqui):** Für die Modelle **`0,2,1,3`**
verwenden — GPU 3 ist PCIe-direkt angebunden. **Vigilantia/TTS liegt inzwischen
auf der USB4-Karte GPU 4**, wo die Tunnel-Latenz nicht ins Gewicht fällt.

Die ältere Reihenfolge `0,2,1,4` in `FLASH-NEXT-OPERATING-POINT.md` und im
Startskript-Default stammt aus der Zeit, als GPU 3 für Vigilantia reserviert
war. Sie legt das Modell auf die getunnelte Karte und lässt die direkt
angebundene brachliegen — **nicht mehr benutzen.**

**Falle:** Diese Nummern gelten nur mit `CUDA_DEVICE_ORDER=PCI_BUS_ID`. Ohne die
Variable sortiert CUDA nach „fastest first" und die V100 rutschen nach vorn —
jede Kartenangabe in Skripten wird dann falsch.

Die 64 KB gegen 96 KB Shared Memory sind der Grund, warum FlashQLA-SM70 auf
Turing nicht startet und dort der Triton/FLA-Pfad läuft.

**Speicherdurchsatz — der Nenner jeder Kernel-Aussage** (gemessen 09.09.,
reine Lesereduktion über 512 MiB, `benchmarks/kernel_matched_bench.py`):

| Karte | theoretisch | gemessen (nur lesen) |
|---|---:|---:|
| Quadro RTX 8000 (GDDR6, 6501 MHz, 384 bit) | 672 GB/s | **601 GB/s** |
| Tesla V100-PCIE (HBM2, 877 MHz, 4096 bit) | 898 GB/s | **849–853 GB/s** |

Die V100 hat **34 % mehr Bandbreite**. Wer zwei Karten über absolute GB/s
vergleicht, misst diesen Faktor und sonst nichts — Prozent der jeweils
eigenen Obergrenze ist die einzige faire Zahl. Bei DFlash2 liegt die RTX
trotz der 34 % weniger Bandbreite inzwischen **vorn** (77,13 gegen 76,33).

Der Unterschied im L1 ist der zweite, weniger bekannte: Volta hat **128 KB**,
Turing ein unified L1/Smem von **96 KB**. Das ist die Ursache der
DFlash2-Tempolücke, siehe offener Punkt 8.

---

## Betriebspunkte

### Qwen3.8-Flash-Next (180B, Qwen4Exp) — geprüft 07.09., nachverifiziert 09.09.

```bash
cd /home/mp/Projekte/vllm-research/v100-skinny
VLLM_SM70_E5_CACHE=0 CUDA_VISIBLE_DEVICES=0,2,1,3 \
TURBOMIND=1 QUANT_BACKEND=turbomind \
ENV_PREFIX=/home/mp/vllm/venv \
TP=2 PP=2 K=4 GMU=0.95 MML=16384 PP_PARTITION=24,24 PLE_HOST_GIB=6 \
PORT=8026 \
bash scripts/serve-qwen38-flash-next.sh <checkpoint>
```

Boot ~9 min. Ergebnis 07.09.: 600 Token zusammenhängendes, fachlich korrektes
Deutsch, kein Zerfasern. Drei Fragen hintereinander im selben Kontext (bis 1124
Prompt-Token) ebenfalls sauber; beim Fangbegriff „Kuanda-Effekt" keine
Halluzination, sondern korrekte Rückfrage.

**Tempo (gemessen 07.09., Streaming, Token aus `usage` — NICHT Chunks zählen,
bei MTP kommen ~3 Token pro Chunk):**

| Konfiguration | kurzer Prompt | langer Prompt, repetitiv | **langer Kontext, echter Text** |
|---|---:|---:|---:|
| k=0 (MTP aus) | 38,5 | 38,1 | — |
| k=4, BF16-Draftkopf | 34,7 (Verlust) | 36,8 | — |
| **k=4, NVFP4-Draftkopf (MTPQ)** | **56,3** | **75,6** | **20–21** |

Prefill 413–466 tok/s. **`K=4` mit dem MTPQ-Checkpoint fahren** — Faktor 1,46
(kurz) bzw. 1,98 (lang) gegenüber k=0, Akzeptanz 3,71 von 5 Token je Schritt.
Mit dem rohen RadixArk-Snapshot (BF16-Draftkopf) ist MTP dagegen ein VERLUST;
dann besser `K=0`.

**Die mittlere Spalte nicht als Alltagswert lesen:** Sie stammt von einem
Prompt aus wiederholten Absätzen — leicht vorhersagbar, daher hohe Akzeptanz.
Mit echtem deutschem Fachtext bei ~9–10k Kontext sind es **20–21 tok/s**
(gemessen 07.09., drei Fragen im selben Gespräch). Das ist der Wert für den
AIfred-Alltag.

**Gegenüber `FLASH-NEXT-OPERATING-POINT.md` (Stand 28.08.) geändert:**
- `QUANT_BACKEND=turbomind` ist **neu und nötig**. Der Skript-Default `marlin`
  sticht `TURBOMIND=1` bedingungslos aus (`envs.use_sm70_turbomind` gibt bei
  „marlin" sofort False zurück); auf den sm70-Stufen bricht NVFP4-MoE dann mit
  `NotImplementedError` ab. Die dortige Aufrufzeile bootet auf dem 1.5.0-Stand
  **nicht mehr**.
- `ENV_PREFIX`: Skript-Default ist seit 10.09. der Produktions-Symlink
  `/home/mp/vllm/venv` (vorher `.venv-sm70-130`). Nach gelöschten
  Compile-Caches bootet der Stand kalt: dann `BOOT_WAIT_S=2400` und
  `EXTRA_ARGS='--distributed-timeout-seconds 3600 …'` wie in
  `flashnext_qual.sh`.
- Checkpoint: **`/home/mp/models/Qwen3.8-Flash-Next-180B-A4B-NVFP4-MTPQ`**
  fahren. **Korrigiert 08.09.:** Der Transplant **lädt auf dem 1.5.0-Stand
  einwandfrei** (vier Boots, null „routed-expert weights were not loaded"), und
  sein Draftkopf ist **quantisiert**, nicht BF16 — 417 Symlinks auf RadixArk
  plus **eine** Datei von provsalt (der NVFP4-MTP-Block),
  `check_draft_head.py`: *QUANTIZED draft head — good to go*. Die frühere
  Angabe „gefahren wird der rohe RadixArk-Snapshot mit 31 BF16-MTP-Tensoren"
  war falsch und hat am 08.09. einen Fehlstart gekostet.

Warum k=4, warum `capture_sizes [1,2,4,5,8]`, warum Kartenreihenfolge `0,2,1,4`:
→ `FLASH-NEXT-OPERATING-POINT.md`, `docs/journal/QWEN4EXP-PORT-HANDOVER.md`

### Qwen3.8-27B-NVFP4 mit DFlash2 — schnellster Stand, geprüft 10.09.

```bash
cd /home/mp/Projekte/vllm-research/v100-skinny
DEVS=0,2 \
DRAFT=/home/mp/.cache/huggingface/hub/models--maurienne-ai--Qwen3.8-27B-DFlash2-NVFP4-RTNcal/snapshots/bd7a934213c47a9e7ef69eef36bb3325f47fd1f1 \
bash tools/mtp-diagnostics/speed_dflash.sh <name> fork dflash
```

**77,13 tok/s auf 2× RTX 8000, 76,33 auf 2× V100** (`DEVS=1,3`), Text-SHA
`0106659946c064b1`, Annahmelänge 3,325 — auf `.venv-sm70-main` bestätigt
(76,30 V100, 77,12 / 76,75 RTX, siehe Laufzeitumgebung). Braucht den
Block-Pack in `kernels/skinny_kernels.cu` und den Kontext-K/V-Fix: in der
neuen venv als Basisklasse aus PR #592, in `.venv-sm70-150` als Override aus
`fork_patches_150/qwen3_dflash2.py`. Das Skript setzt
`VLLM_SM70_DFLASH2_QUANT_LM_HEAD=1` — seit dem Upstream-Stand vom 10.09.
bricht DFlash2 ohne diesen Opt-in am quantisierten RadixArk-LM-Head ab.
Herleitung: offene Punkte 8 und 10.

**Gemessen nur unter Bench-Bedingungen:** 32k Kontext, Prefix-Caching AUS,
400 Token Ausgabe, kurzer Prompt. **Nicht** gemessen unter den Bedingungen
der Produktion (256K Kontext, Prefix-Caching an, Werkzeugaufrufe) — siehe
offener Punkt 12. `speed_dflash.sh` fährt ohne `DRAFT=` weiter den
unquantisierten incoai-Kopf, damit alte Zahlen vergleichbar bleiben.

### Qwen3.8-27B-NVFP4 — Debug-Fahrzeug, geprüft 07.09., nachverifiziert 09.09.

TP2 auf zwei RTX 8000 (GPU 0,2), `VLLM_SM70_QUANT_BACKEND=auto`,
`VLLM_SM70_NVFP4_TURBOMIND=1`, k=3, FULL-Graphen. Boot **6,5 min** auf Turing,
2 min auf Volta. Aufruf: `tools/mtp-diagnostics/probe.sh`.

Speicherlage TP2: 10,2 GiB Modell je Karte, 29,4 GiB KV-Cache je Karte,
661.796 Token — 20× Reserve für 32k Kontext. **PP ist für dieses Modell nicht
nötig.**

Die im Journal genannten „60–90 s Boot" treffen nicht zu.

### Messstand 09.09. nach dem Umstieg auf den gemeinsamen GDN-Pfad

27B, TP2 auf 2× RTX 8000, k=3, greedy Seed 1, je 5 Läufe
(`tools/mtp-diagnostics/speed_27b.sh <name> fork 3`). Referenz-SHA des
Antworttextes: `0106659946c064b1`, Annahmelänge 2,963 in allen Zeilen.

| Stand | tok/s |
|---|---:|
| fork-eigener sm75-Pfad (bis 09.09.) | 74,77 |
| gemeinsamer Pfad | 73,39 |
| nach Entfernung des toten Codes | 73,40 |
| nach Linter-Bereinigung (Endstand) | **73,35** |

Flash-Next 180B, TP2×PP2 heterogen, k=4, 13.004 Token Vorkontext, drei
30-Sätze-Fragen (`tools/mtp-diagnostics/flashnext_qual.sh <name> 4`), drei
Läufe auf dem gemeinsamen Pfad, jeder mit Rangnachweis `sm75=0 upstream=1`:
**8 von 9 Antworten einwandfrei**, keine Fremdkörper. Der eine Ausfall (q3 im
ersten Lauf: Coandă nicht erkannt, Wiederholungsschleife) war in zwei weiteren
Läufen nicht reproduzierbar — passt zur Nichtreproduzierbarkeit des 180B bei
dieser Ausgabelänge. Endstand-Lauf: 24,65 / 27,08 / 25,74 tok/s.

**Keine Ratenaussage:** drei Läufe neu gegen einen alt. Wer sie braucht,
fährt je fünf.

### DFlash2 gegen MTP (09.09., 27B RadixArk-NVFP4, TP2, greedy, je 5 Läufe)

Messvorrichtung `tools/mtp-diagnostics/speed_dflash.sh`, gegen die MTP-Referenz
geeicht (73,36 gegen 73,35 tok/s, Text-SHA `0106659946c064b1` identisch).
**Alle Läufe liefern denselben Text** — die Verifikation ist über beide
Architekturen und beide Verfahren hinweg verlustfrei.

| Karten | Verfahren | tok/s | Annahmelänge |
|---|---|---:|---:|
| 2× V100 | MTP k=3 | 66,13 | 2,963 |
| 2× V100 | DFlash2 k=7 | 74,09 | 3,381 |
| 2× V100 | **DFlash2 k=7, mit Block-Pack** | **74,02** | **3,381** |
| 2× RTX 8000 | MTP k=3 | 73,36 | 2,963 |
| 2× RTX 8000 | DFlash2 k=7, **vor** Gate-Patch | 21,35 | 1,015 |
| 2× RTX 8000 | DFlash2 k=7, nach Gate-Patch | 69,13 | 3,353 |
| 2× RTX 8000 | DFlash2 k=7, mit Block-Pack | 72,72 | 3,353 |
| 2× V100 | **DFlash2 k=7, + quantisierter Entwurfskopf** | **76,33** | **3,325** |
| 2× RTX 8000 | **DFlash2 k=7, + quantisierter Entwurfskopf** | **77,13** | **3,325** |

Die Block-Pack-Zeilen sind vom **09.09. abends**, gegen eine in derselben
Sitzung neu gefahrene Grundlinie (69,22 tok/s auf der RTX, deckt sich mit den
69,13 vom Nachmittag). Die beiden letzten Zeilen sind vom **10.09.**
Text-SHA in **allen** Zeilen `0106659946c064b1`. Siehe Punkt 8 (Block-Pack)
und Punkt 10 (Entwurfskopf).

**Seit dem 09.09. mittags: RTX 8000 69,13 → 77,13 tok/s (+11,6 %).** Die RTX
liegt damit erstmals VOR der V100, obwohl sie 34 % weniger Bandbreite hat.

**DFlash2 schlägt MTP** — auf gleicher Hardware +12,0 % bei +14 %
Annahmelänge. Zwei Patches waren dafür nötig, beide in `fork_patches_150/`:

1. **Guard in `compute_candidates`** (`qwen3_dflash2.py`) lehnte einen
   quantisierten Ziel-LM-Head ab. RadixArk quantisiert ihn mit (die
   `ignore`-Liste nennt nur `mtp*`), damit war DFlash2 mit diesem Checkpoint
   auf **keiner** Karte fahrbar. Der Kandidaten-TopK wählt nur die Entwürfe;
   verifiziert wird gegen das Ziel, also kostet Quantisierung Annahmerate und
   nie Korrektheit — belegt durch den unveränderten Text-SHA.
2. **`_use_sm70_bf16_emulation`** prüfte auf exakt sm70. Turing hat genauso
   wenig natives BF16 wie Volta, bekam die Range-Erhaltung aber nicht — daher
   die 0,2 % Annahmerate. Jetzt: Capability < sm80, gefragt wird das Gerät des
   Workers (`torch.cuda.current_device()`) statt Gerät 0.

**Preis der Quantisierung des Kandidatenkopfs:** QUASAR-QAT nimmt den lm_head
per `ignore` aus und erreicht auf V100 Annahmelänge 5,569 statt 3,381 — rund
40 % mehr. Diese Zahl ist aber **wertlos**, siehe offener Punkt 7.

**Die Tempolücke auf Turing ist zum größten Teil geschlossen** (09.09. abends):
69,22 → 72,72 tok/s, Abstand zur V100 von 6,7 % auf 1,8 %. Ursache und Fix
stehen in Punkt 8. Der frühere Verdacht — QPN8-Rerank für den Kandidaten-TopK
— ist widerlegt und dort mit den vier anderen widerlegten Erklärungen
aufgeführt.

---

## Läuft / läuft nicht (Messmatrix 07.09.)

| Konfiguration | Ergebnis |
|---|---|
| Flash-Next TP2×PP2, k=4, V2, FULL | **läuft**, 600 Token sauberes Deutsch |
| 27B TP2 RTX, k=3, V2, FULL | **läuft** |
| 27B TP2 RTX, k=1, V2, FULL | **läuft** (andere Graph-Form) |
| 27B TP2 RTX, k=3, V1, FULL | **läuft**, bitgleich zu V2 |
| 27B TP2 Volta, k=3, V2, FULL | **läuft**, bitgleich (Standard-GDN-Pfad) |
| 27B TP2 RTX, k=3, reines PIECEWISE | **defekt**: CJK-Mojibake, kein NaN |
| 27B PP2 (TP1 oder TP2), MTP | **bootet nicht**: `Eq(s72, 5120)` |

Zu den beiden Defekten:

- **Reines PIECEWISE — bewusst NICHT angefasst.** Der Modus rechnet unter
  Spekulation still falsch (CJK-Mojibake). Er ist bei uns aber nicht
  erreichbar: vLLM setzt ihn nur selbst, wenn ein Attention-Backend WENIGER als
  `AttentionCGSupport.UNIFORM_BATCH` meldet und Spekulation aktiv ist
  (`config/compilation.py`, ~Zeile 1412). Im ganzen Baum meldet nur
  `v1/attention/backends/linear_attn.py` weniger
  (`UNIFORM_SINGLE_TOKEN_DECODE`) — und den nutzen unsere Modelle nicht, die
  laufen über GDN (`UNIFORM_BATCH`). Erreicht wurde der Defekt nur durch
  manuelles Setzen von `cudagraph_mode=PIECEWISE` im Test.
  **Entscheidung 07.09. (Peuqui):** nicht fixen — ein Eingriff in einen Modus,
  den wir nie fahren, ist Risiko ohne Nutzen. Für Upstream bleibt es
  bemerkenswert: Der Code setzt dort selbst einen Modus, der mit Spekulation
  still falsch rechnet, statt ihn abzulehnen. Upstream-Code (`speculator.py`,
  `cudagraph_utils.py`), unabhängig von unserem Fork.
- **27B mit PP** ist **kein Defekt, sondern ein unfertiges eigenes Feature**:
  unser `qwen3_5_mtp.py` trägt `SupportsPP` und eine geänderte Verzweigung,
  während Qwen3_5MTP bei vLLM **und** 1Cat nicht PP-fähig ist (gegen
  `origin/main` geprüft). Der 27B braucht PP nicht (siehe Speicherlage oben).
  Rücknahme der Mitnahme wäre ehrlicher als der kryptische Compile-Abbruch.

---

## GDN-Kernel: Volta gegen Turing

**Seit 09.09.: Turing faehrt den gemeinsamen Pfad.** Bis dahin baute ein
Turing-Worker eine fork-eigene Klasse aus `qwen_gdn_linear_attn_sm75.py`, weil
die FlashQLA-SM70-Kernel dort 86016 B Shared Memory verlangen und Turing bei
65536 B deckelt. Fix 1 (PR #572) macht FlashQLA Volta-only, damit ist der
Sonderzweig ueberfluessig. Entfernt in c86fc8d/ac0b0ae; die beiden sm75-Dateien
sind aus dem Overlay raus (sie stammten ohnehin aus dem 1.5.0-Wheel und sind
upstream schon geloescht).

Kosten: 27B von 74,77 auf 73,35 tok/s (-1,9 %). Gegenwert: kein Sonderpfad,
2.352 Zeilen weniger, und weg ist eine **Unterdeklaration** —
`qwen_gdn_attention_core_sm75` rief `causal_conv1d_update` auf seinem
qkv-Eingang, ohne die Mutation in `mutates_args` zu nennen.

| | Volta (sm70) | Turing (sm75) |
|---|---|---|
| GDN-Schicht | `QwenGatedDeltaNetAttention` (gemeinsam) | dieselbe, seit 09.09. |
| GDN-Prefill | FlashQLA-SM70 | Triton/FLA |
| GDN-Decode | FlashQLA-Route | FLA (fused recurrent) |
| Lineare Projektionen | NVFP4-Skinny | NVFP4-Skinny |

Messung 07.09. auf zwei V100, 15.492-Token-Prompt, Prefix-Caching aus
(`tools/mtp-diagnostics/bench_gdn.sh`):

| | FlashQLA | Triton/FLA | Differenz |
|---|---|---|---|
| Prefill | 726,6 tok/s | 708,6 tok/s | +2,5 % |
| Decode | 60,66 tok/s | 57,06 tok/s | +6,3 % |

**Einordnung:** Das ist ein End-zu-End-Wert, kein Kernel-Wert — der GDN-Anteil
am Prefill ist nicht gemessen. Ein FlashQLA-Port auf Turing ist auf diese
Größenordnung gedeckelt, solange der Anteil klein ist. Vor Kernel-Arbeit erst
`tools/mtp-diagnostics/prof_prefill.sh` laufen lassen (nsys-Kernelsummen).

---

## Qualitätsprüfung: was ein Test abdecken MUSS

**Langer Kontext ist der kritische Fall, nicht langer Output.** Das Zerfasern
des 180B (Wortverstümmelungen, CJK-Zeichen, erfundene Wissenschaftler) trat bei
**langem Kontext** auf — Peuqui hat es in AIfred im Alltag gesehen, wo RAG,
Tool-Schemata und History den Prompt füllen. Ein Test mit kurzem Prompt und
600 Token Ausgabe sieht das NICHT (07.09. so passiert).

Die etablierte Sonde (seit 30.08., → `docs/journal/FP8-EVALUATION.md`):
drei Anfragen hintereinander im selben Kontext —

1. „Erkläre die Quantenphysik in 30 Sätzen."
2. „Erkläre den Regenbogeneffekt in 30 Sätzen."
3. „Erkläre den **Kuanda-Effekt** in 30 Sätzen."  ← Schreibfehler ABSICHTLICH

Frage 3 ist der Halluzinationstest (gemeint ist der Coandă-Effekt).

**Die Antworten MÜSSEN gelesen und fachlich beurteilt werden.** Zähler über
CJK-Zeichen oder Satzzeichen sagen nichts über Qualität — ein fachlich
unsinniger, aber sauber deutscher Text besteht jeden solchen Test
(Peuqui, 07.09.). Sie taugen allenfalls als Vorfilter für grobes Zerfasern.

### Wo Hash-Vergleiche gelten — und warum sie irgendwann aufhören

Gemessen am 08.09., alle mit `temperature 0` und festem Seed:

| Fall | Ergebnis |
|---|---|
| 19-Token-Prompt, 400 Token Ausgabe (27B) | **8 von 8 Boots** byteidentisch |
| bis 13.004 Token Kontext, 260 Token Ausgabe (27B) | **70 von 70** Vergleichen byteidentisch |
| 13.004 Token Kontext, 1.200 Token Ausgabe (27B) | **6 von 6 byteidentisch** (3× im Prozess × 2 Boots) |
| 13.005 Token Kontext, 1.200 Token Ausgabe (180B) | **k=0 gegen k=0, zwei Boots: drei verschiedene Hashes** |
| 9,4k Prompt, 663–806 Token (DeepSeek-V4) | byteidentisch (07.–09.09., Journal) |

**Der 27B ist reproduzierbar, auch bei langer Ausgabe** (`determinismus.sh`,
08.09.: dreimal dieselbe Frage im selben Serverprozess, zweimal gebootet, alle
sechs `860cbb36540edf72`). Nichtdeterministisch ist bisher **nur das 180B**.
Eine frühere Notiz, der 27B „driftet ab ~400 Token", war ein Fehlschluss aus
einem k=0-gegen-k=3-Vergleich — das war ein Konfigurationsunterschied, keine
Streuung.

**Folge daraus, die man kennen muss:** Weil der 27B reproduzierbar ist, sind
Hash-Unterschiede zwischen *Konfigurationen* dort echte Effekte. Bei 1.200
Token Ausgabe liefern verschiedene Baseline-Kombinationen verschiedene, jeweils
gelesene und fachlich korrekte Texte; bei 260 Token stimmen sie alle überein
(70 von 70). Die Schwelle liegt also nicht beim Determinismus, sondern bei der
Empfindlichkeit gegenüber winzigen numerischen Unterschieden.

**Die Ursache ist bekannt und keine Eigenheit unseres Stacks: fehlende
Batch-Invarianz der Kernel.** RMSNorm, Matrixmultiplikation und Attention
wählen ihre Reduktionsstrategie nach Form und Last — datenparallel gegen
Split-Reduktion, andere Kachelgrößen, andere Tensor-Core-Instruktionen. Damit
ändert sich die **Reihenfolge der Fließkomma-Additionen**, und die ist nicht
assoziativ. Winzige Unterschiede pflanzen sich fort, bis ein Argmax kippt; ab
da läuft der Text auseinander. Das passiert **auch bei Batchgröße 1 und
einzeln gestellten Anfragen**. Hauptverdächtiger beim Decodieren mit langem
KV-Cache ist die **Split-KV-Attention**, deren Split-Zahl zur Laufzeit gewählt
wird — `flash_fwd_splitkv_kernel` und `flash_fwd_splitkv_combine_kernel` stehen
in unseren eigenen nsys-Profilen. Das erklärt die Tabelle: Bei kurzem KV wird
nicht gesplittet, bei 13k mit kurzer Ausgabe bleibt die Split-Zahl konstant,
bei langer Ausgabe wächst der KV über eine Schwelle.

Quellen: Thinking Machines Lab, *Defeating Nondeterminism in LLM Inference*
(https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/);
arXiv 2506.09501, *Understanding and Mitigating Numerical Sources of
Nondeterminism in LLM Inference*.

**ENTSCHEIDUNG (Peuqui, 08.09.): batch-invariante Kernel werden NICHT
eingebaut.** Es gäbe sie (in vLLM per Torch.Library integrierbar) und sie
liefern bitgleiche Ergebnisse — zum Preis von **1,6- bis 2,1-fach langsamerer
Inferenz**. Der Qualitätsgewinn ist bestenfalls marginal, das Tempo geht vor.
Nicht erneut vorschlagen.

**Ebenfalls geschlossen:** Warum DeepSeek-V4 bei 600–800 Token stabil bleibt,
wo der 27B schon driftet (vermutlich andere Attention-Implementierung), wird
**nicht** weiter untersucht — kein Forschungsprojekt daraus machen
(Peuqui, 08.09.).

**Praktische Regel:** Hash-Vergleiche sind ein starkes Werkzeug — aber nur bis
etwa 260 Token Ausgabe. Darüber hinaus beweist ein abweichender Hash **nichts**
über Qualität, dort zählt ausschließlich das Lesen der Texte.

| Messung | CJK | verdächtige Wörter | Coandă-Turn |
|---|---:|---:|---|
| 30./31.08., **27B** | 7 | 183 | 2 halluzinierte Wissenschaftler |
| 07.09., **180B**, kurzer Kontext | 0 | 0 | keine Erfindung, korrekte Rückfrage |
| 07.09., **180B**, langer Kontext (8,5–9,8k) | **0** | keine | **erkennt den Tippfehler, erklärt Coandă korrekt** |

**Inhaltliche Beurteilung des Langkontext-Laufs (gelesen, 07.09.):**
- *Quantenphysik*: fachlich einwandfrei — Planck/Schwarzkörper, Einstein/
  Photoeffekt, de Broglie, Schrödinger, Heisenberg korrekt zugeordnet; Bell-
  Ungleichungen und die Unschärfe als Natureigenschaft (nicht Messproblem)
  präzise. Ein Genusfehler („das Quantenzustand").
- *Regenbogen*: dicht und korrekt — 42°, Sekundärbogen mit umgekehrter
  Farbfolge, **Alexanderband** benannt, Tropfengröße/Schärfe. Eine
  missverständliche Stelle („flacherer Winkel" für Rot).
- *Kuanda*: erkennt den Schreibfehler, nennt Henri Coandă korrekt als
  rumänischen Ingenieur, ordnet historisch differenziert ein, Physik und
  Navier-Stokes korrekt. Zwei Vereinfachungen: Auftrieb (streift den
  verbreiteten Irrtum) und Viskosität/Haftung.

Die Augustwerte stammen vom 27B, die Septemberwerte vom 180B — nicht
gleichnamig. Der Langkontext-Fall ist am 180B **bestanden**: 30 Sätze je
Antwort, null CJK, kein Zerfasern; im Coandă-Turn erkennt das Modell den
Schreibfehler und erklärt den richtigen Effekt (bei KURZEM Kontext sagt es
dagegen „kenne ich nicht" — die Langkontext-Antwort ist die bessere).
**Offen: dieselbe Sonde am 27B**, für den gleichnamigen Vergleich zu den
Augustwerten (6,5 min Boot).

---

## Fallstricke (teuer erkauft)

- **cwd nie im vLLM-Paketverzeichnis** — dort schattet `vllm/tokenizers` die
  echte `tokenizers`-Bibliothek, `transformers` bricht mit Zirkelimport.
- **Kein Skript ändern, während es läuft** — bash liest inkrementell weiter und
  stirbt mitten im Lauf; ein so abgebrochener Lauf lässt den Server stehen und
  hält die Karten. `tools/mtp-diagnostics/gpu_deadman.sh` ist der Totmannschalter.
- **Prefix-Caching bei Messungen aus** (`--no-enable-prefix-caching`), sonst
  trifft die zweite Anfrage den Cache und der Prefill entfällt.
- **Marken erst nach Laufende auswerten**, immer den ganzen Verlauf — ein `tail`
  auf die Prefill-Schritte hat schon einen falschen Teilbefund erzeugt.
- **`nvidia-smi` vor jedem Lauf** — Fremdbelegung macht Messungen wertlos.
- **`VLLM_SKINNY_SM75_GDN` existiert nicht** — die Modulwahl Fork/Upstream
  steht hart an `get_device_capability() == (7, 5)` in `qwen3_5.py` (~534)
  und `models/qwen4_exp/nvidia/model.py` (~253). Sonden, die über die
  Variable umschalten wollten, verglichen den Fork mit sich selbst.
- **venv ≠ Checkout.** Die venv ist 1.5.0-Wheel + Overlay, der Checkout ist
  `origin/main` + Fixes. Ein Fix im Checkout ist NICHT in der venv.
- **Vor jeder Zahl zwei Nachweise:** welches Modul lädt (Boot-Log) UND welche
  Fixes in der geladenen venv-Datei stehen (Marker). Rezept:
  `tools/mtp-diagnostics/README.md`.
- **Ein Testschalter muss beweisen, dass er feuert** (08.09.). Vier Läufe
  maßen unveränderten Code, weil der Schalter in
  `QwenGatedDeltaNetAttention.forward_cuda` saß — einer Methode, die
  `Qwen3_5GatedDeltaNet` (`models/qwen3_5.py:392`) **überschreibt**. Für den
  27B ist die Basisklassen-Fassung tot, für Flash-Next (Qwen4Exp) dagegen
  lebendig. Vor jedem Bisect: Klasse und überschriebene Methode prüfen.
- **Aus einer getracten Funktion darf man nicht loggen** (08.09.).
  `logger.warning_once` im `forward_cuda` killt den Boot:
  `torch._dynamo.exc.Unsupported: logging.Logger method not supported`. Dafür
  gibt es `_log_runtime_route_once` mit `is_compiling()`-Sperre — die aber
  genau dann schweigt, wenn man sie als Nachweis braucht. Diagnosemarken
  gehören in `__init__`.
- **INFO von Nebenrängen wird gefiltert** (08.09.). Unter PP loggt
  `Worker_PP0_TP0` hunderte Zeilen, die anderen Ränge unter fünfzehn — eine
  fehlende `info_once`-Meldung ist **kein** Beweis, dass der Code nicht lief.
  Für Zustandsnachweise pro Rang in eine Datei schreiben
  (`AIFRED_STATE_FILE`-Muster), nicht loggen.
- **Die Fangfrage heißt „Kuanda-Effekt", nicht „Coandă".** Erledigt 08.09.:
  `qual_longctx.sh` und `flashnext_qual.sh` fragen den absichtlich falsch
  geschriebenen Begriff, mit Kommentar im Skript, der das Zurückändern
  verbietet. **Bewertung (Peuqui, 11.09.):** Bestanden ist die Frage in zwei
  Formen — das Modell erkennt den Verschreiber und erklärt den Coandă-Effekt,
  ODER es weist den Begriff als nicht existent zurück und nennt Coandă als
  möglichen Kandidaten, ohne etwas zu erfinden. Beides ist korrektes
  Verhalten. **Durchgefallen** ist nur, wer ein neues Phänomen erfindet,
  Wissenschaftler halluziniert oder in eine Wiederholungsschleife gerät.
  Die frühere, strengere Fassung („nur Erklären zählt") ist zurückgenommen:
  Halluzination ist das Schlimmere, striktes Festhalten am gestellten
  Begriff ist kein Fehler.
- **Ein einzelner Durchfall beim 180B beweist nichts** (09.09.). Im ersten von
  drei Läufen verfehlte q3 die Coandă-Erkennung und drehte sich im Kreis; in
  zwei weiteren Läufen korrekt. Bei dieser Ausgabelänge reproduziert sich das
  180B nicht einmal mit sich selbst — vor einer Schlussfolgerung wiederholen.
- **`git stash push <datei>` legt bei sauberem Baum KEINEN Stash an** (09.09.),
  und das folgende `git stash pop` nimmt dann den obersten **fremden** Stash.
  Zweimal an einem Tag zugeschnappt: einmal wurde ein fremder Stash verbraucht
  (über `git fsck` zurückgeholt), einmal wäre ein Commit unvollständig
  geworden. Für Vorher/Nachher-Vergleiche `git checkout origin/main -- <datei>`
  und zurück mit `git checkout HEAD -- <datei>`.
- **`python /pfad/skript.py` setzt `sys.path[0]` auf das SKRIPTverzeichnis**,
  nicht auf das Arbeitsverzeichnis (09.09.). Eine Sonde im Scratchpad
  importierte deshalb das `vllm` aus der venv statt aus dem Checkout und maß
  unseren eigenen Patch statt Upstream. Sonden müssen `vllm.__file__` und den
  Quelltext der geprüften Funktion mitprotokollieren.
- **Ein fremdes Profil wörtlich zu übernehmen stellt NICHT seine Bedingungen
  her** (09.09.). 1Cats Aufrufzeile enthält kein
  `--disable-custom-all-reduce`, weil ihre vier V100 in einem Server mit
  funktionierendem P2P stecken. Auf diesem Rechner ist P2P aus (Karten an
  OCuLink/USB4), und der Custom-Allreduce-Kernel setzt direkte
  GPU-zu-GPU-Zugriffe voraus: beide TP-Ränge warten dann im Allreduce, ohne je
  fertig zu werden. Zwei Läufe verloren. Beim Nachbauen fremder Profile die
  eigenen Hardware-Flags NICHT streichen — sie sind keine Geschmacksfrage.
- **`utilization.gpu` ist keine Fortschrittsanzeige** (09.09., Peuqui).
  Ein wartender NCCL- oder Autotune-Kernel meldet 100 % bei ~46 W auf der
  V100 — die Karte dreht sich im Leerlauf. **Leistungsaufnahme ist der
  brauchbare Indikator**, und für „arbeitet oder steht" hilft nur
  `py-spy dump` in mehreren Proben: steht dieselbe Zeile über Minuten und
  fehlt `benchmark_all_configs` im Stack, ist es ein Hänger und kein
  Autotuning.
- **vLLM liefert den Denkblock im Feld `reasoning`, NICHT
  `reasoning_content`** (09.09.). Eine Sonde, die nur `reasoning_content` und
  `content` liest, wirft 1.600 erzeugte Token weg und meldet leere Antworten.
  Steht so auch in AIfred (`aifred/backends/base.py`, SEND_TURN_REASONING mit
  Issue #38488). **Rohantwort immer als JSON mitschreiben**, bevor Felder
  ausgelesen werden.
- **Ein Qualitätsurteil braucht die volle Ausgabelänge** (09.09.). Bei 1.600
  Token wirkte ein QUASAR-Denkblock sauber; mit 6.000 Token zeigte derselbe
  Prompt eine Wiederholungsdegeneration (an jeden Satz derselbe Nachsatz).
  Ein abgeschnittener Text misst nur, wie weit ein Modell kommt, bevor es
  kippt.
- **`torch.utils.cpp_extension` ignoriert `TORCH_CUDA_ARCH_LIST` KOMPLETT,
  sobald in `extra_cuda_cflags` schon ein `-gencode` steht** (09.09.).
  `_get_cuda_arch_flags()` gibt dann `[]` zurück. Der Skinny-Shim übergibt
  genau das (`-gencode=arch=compute_70,code=sm_70`), also ist jeder Build
  sm_70 — auch auf der RTX 8000, die ihn über die Cubin-Kompatibilität
  derselben Major-Version ausführt. `cuobjdump -lelf` auf der gebauten `.so`
  zeigt eine einzige ELF, `sm_70`, kein PTX. Wer die Architektur einer
  Extension prüfen will, fragt die `.so`, nicht die Umgebungsvariable.
  Praktisch kostet es wenig: ein echter sm_75-Build bringt bei M≤4 ein bis
  sechs Prozent und bei M=8 **nichts** (gemessen).
- **Synthetische Zufallsgewichte sprengen den Bitvergleich** (09.09.). Codes
  aus `randint(0,256)` mit zufälligen e4m3-Skalen und `gscale=1.0` laufen
  durch `gscale*16384` in den fp16-Überlauf; der Vergleich liest dann `NaN`
  zurück und sagt über keinen der beiden Kernel etwas. Für Äquivalenztests
  Skalen auf exakt 1,0 (e4m3 `0x38`) und `gscale=1/16384` setzen, und die
  Endlichkeit der Referenz mitprüfen.
- **Ein „flexibler" Kernel-Parameter kann teurer sein als zwei
  Instanziierungen** (09.09.). Das Aktivierungs-Layout von `qpn2` als
  Laufzeit-Strides zu übergeben machte aus dem Gruppen-Offset (`g*16`, ein
  Shift) ein IMAD im Innenloop und kostete die V100 1–5 % — messbar daran,
  dass Zellen, die denselben Pfad fahren wie vorher, plötzlich 0,95× statt
  1,00× standen. Als Template-Parameter ist es wieder exakt 1,00×. **Wenn ein
  unveränderter Pfad nicht exakt 1,00× misst, ist die Änderung nicht so
  neutral, wie sie aussieht.**
- **`speed_dflash.sh` bricht ab, wenn das eigene Aufrufkommando das Wort
  `api_server` enthält** (09.09.). Die Sicherung ist
  `pgrep -af 'api_server' | grep -v $$`, und `$$` schließt nur die Subshell
  aus, nicht die aufrufende Kommandozeile. Ein `echo`, das den Namen erwähnt,
  reicht für „ABBRUCH: api_server laeuft" bei völlig freien Karten.
- **`import vllm` mit cwd im 1Cat-Checkout findet das LOKALE Verzeichnis**
  (10.09.). Eine Prüfung „welche venv hat vLLM wo" meldete für alle drei
  venvs den Checkout — das war nur cwd. Aus neutralem Verzeichnis zeigt jede
  venv auf ihr eigenes site-packages. Der Checkout trägt einen Symlink
  `vllm/_C.abi3.so` → `.venv-sm70-150/.../vllm/_C.abi3.so`; mit cwd im
  Checkout prüft pytest deshalb wirklich den **Checkout-Code** mit den
  fertigen Extensions. Genau das ist für 1Cat-PRs gewollt — man muss es nur
  wissen, statt es zufällig richtig zu machen.
- **Einen Helfer nie zwischen `@support_torch_compile` und die Klasse
  schieben** (10.09.). Ein per Skript „vor `class DFlashQwen3Model`"
  eingefügter Helfer landete unter dem Dekorator; der lag dann auf der
  Funktion, und die Tests brachen schon beim Sammeln ab
  (`assert isinstance(cls, type)`). Vor dem Einfügen die Zeile ÜBER dem Anker
  ansehen.
- **Eine auffällig HOHE Annahmelänge ist ein Warnsignal, kein Erfolg**
  (09.09.). Eine Wiederholungsschleife ist trivial vorhersagbar, also nimmt
  der Verifizierer fast jeden Entwurf an: gemessen 5,569 von 8 möglichen — bei
  völlig degeneriertem Text. Gesund sieht anders aus: die
  Per-Position-Annahmeraten fallen ab (0,822 / 0,644 / 0,550 / … / 0,246).
  Sind alle Positionen gleichmäßig hoch, zuerst den Text lesen.
- **1Cat-Quellbau nur mit `TORCH_CUDA_ARCH_LIST=7.0`** (10.09.). Die
  Produktion war immer reines sm_70 (cuobjdump; die RTX fährt es per
  Binärkompatibilität). Bei `7.0;7.5` lässt 1Cats CMake SM70-Marlin samt MoE
  ganz weg (`MARLIN_SM70_ARCHS AND NOT MARLIN_OTHER_ARCHS`) — die V100 hätte
  kein Marlin, die RTX andere Kernel.
- **Das nvcc-Deb hat keine CCCL-Header** (10.09.). Ohne `CPATH` auf
  `nvidia/cuda_cccl/include` der venv greift Ubuntus libcu++ 1.9 aus
  `/usr/include`, und `grouped_topk_kernels.cu` bricht an
  `cuda::std::isfinite`.
- **1Cats editable Bau ist dreifach kaputt** (10.09.; 1Cat selbst baut nur
  Wheels): fünf `WITH_SOABI`-Module ohne `USE_SABI` scheitern erst NACH dem
  Vollbau am abi3-Namen (Fix `f03a7102`); `flash_attn_v100` fällt aus dem
  editable Mapping (absoluter `package_dir`); die GDN-Erweiterung landet nur
  im temporären `build-lib`. pip löscht bei jedem Fehlschlag das Build-Temp —
  Nachlaufschritte vorher einzeln prüfen.
- **FA2-Bibliotheken nie beim Import laden** (10.09.). Der Worker hat sein
  Gerät dann noch nicht gewählt — ein verstecktes Gerät-0-Gate. zhinianqins
  V100-FA und unsere sm75-FA gehen nicht in eine Bibliothek: die V100-FA ist
  eine Neufassung mit festen `SM70_8x8x4`-Atomen, die sm75-FA nutzt die
  stabile ABI (1Cats v37-`register.cpp` bricht dort mit `#error`), und beide
  beanspruchen `TORCH_LIBRARY(_vllm_fa2_C)`.
- **Keine Kommentarzeile in eine Backslash-Kette** (10.09.). In
  `flashnext_qual.sh` beendete ein `#` die Env-Präfix-Kette; `K`,
  `ENV_PREFIX` und Co. kamen nie an, der Server lief mit alter venv und ohne
  MTP. `bash -n` war grün, aufgefallen nur über `speculative_config=None`.
  Übergabe per Trockenlauf und `/proc/<pid>/cmdline` belegen.
- **Der Tool-Call-Parser folgt dem Chat-Template, nicht der Modellfamilie**
  (11.09.). Qwen3.8 (27B wie Flash-Next) schreibt Aufrufe als XML
  (`<tool_call><function=NAME><parameter=P>…`) und braucht
  `--tool-call-parser qwen3_coder --reasoning-parser qwen3`. Alle
  vLLM-Einträge und die Startskripte hier fuhren `hermes` (JSON): vLLM
  verschluckte damit jeden Aufruf im Streaming STILL (`finish_reason=
  tool_calls` ohne Aufruf, keine Logzeile). Kein vLLM-Eintrag hat je ein
  Werkzeug aufgerufen; aufgefallen erst am Gebetsauftrag, weil alle
  vLLM-Tests reine Textfragen waren. Behoben 11.09.: llama-swap-Einträge,
  `serve-qwen38-mini.sh`/`-native.sh`; AIfreds Kalibration leitet beide
  Parser jetzt aus dem Template ab (`data/vllm_runtime.yaml`,
  `*_by_template_marker`). Belegt mit Streaming-Proben (Denken an/aus ×
  Werkzeug an/aus) auf 27B-MTP, 27B-DFlash2 und Flash-Next. Zweiter Fund
  dabei: AIfred las bei vLLM das Denkfeld `reasoning` nicht — Flash-Next
  lief seit 07.09. ohne sichtbaren Denkblock. **Ein neues Profil ist erst
  abgenommen, wenn es einmal ein Werkzeug aufgerufen hat.**

---

## Offene Punkte

1. ~~MTP-Beschleunigung blockiert~~ — **ERLEDIGT 08.09.** Der MTPQ-Transplant
   lädt auf 1.5.0 fehlerfrei; der Boot-Abbruch
   (`Qwen4Exp MTP routed-expert checkpoint weights were not loaded`) tritt in
   vier Läufen am 08.09. **nicht mehr** auf. `K=4` ist gemessen und liefert
   Annahmelänge **3,030** bei 57,7–60,7 tok/s (300 Token, greedy). Die frühere
   Anweisung „solange `K=0` fahren" ist überholt.
2. **GDN-Anteil am Prefill messen**, bevor über einen FlashQLA-Turing-Port
   entschieden wird.
3. **DeepSeek-V4 nach dem GDN-Umstieg nicht nachgemessen** (09.09.).
   Strukturell nicht betroffen: DSV4 baut keine Qwen-GDN-Schicht, und keine
   `deepseek_v4_*`/`dsv4_*`-Datei referenziert den entfernten Apparat. Die
   beiden generischen Dateien (`gpu_model_runner.py`, `mamba_hybrid_state.py`)
   verlieren nur einen Aufruf, der seit dem Umstieg immer `False` lieferte.
   **Argument, keine Messung.** Ein Gegentest wäre billig und aussagekräftig,
   weil DSV4 byteidentisch reproduzierbar ist: `scripts/serve-deepseek-het-graphs.sh`
   (PP5 über alle fünf Karten), Referenz Essay 21,3 / Code 26,7 tok/s, 8/8
   Kohärenz.
5. ~~Braucht es den fork-eigenen sm75-GDN-Backend?~~ — **ERLEDIGT 09.09.,
   Antwort: nein.** Der Sonderzweig ist entfernt, Turing baut den gemeinsamen
   `QwenGatedDeltaNetAttention` (c86fc8d), der tote Apparat samt beider
   sm75-Dateien ist raus (ac0b0ae). Belegt für 27B **und** Flash-Next, siehe
   „Messstand 09.09." oben. Kosten 1,9 % beim 27B, dafür kein Sonderpfad und
   keine Unterdeklaration mehr.

   **Der Restabstand von 1,74 % ist wieder UNLOKALISIERT.** Die frühere
   Zuordnung zu `_sm70_compile_graph_slice_dim` war ein Denkfehler: wenige
   Zeilen unter dem Split stehen `z.contiguous()`, `b.contiguous()`,
   `a.contiguous()` — nach `index_select` No-ops, mit einem View kopieren sie
   dort erst recht. Die Kopie verschwindet nie, sie wandert nur. Vier
   Messungen, jede Sparmaßnahme langsamer; `a` als View ist korrekt, kostet
   aber 0,4 %. **Nicht dort weitersuchen** — die übrigen Unterschiede des
   entfernten sm75-Pfads (eigene Kern-Op, Faltungsbehandlung, Norm-Fusion)
   wären die Kandidaten. Herleitung: Gedächtnisnotiz
   `project_z_slice_materialization_closed`, Kasten in `HANDOVER.md`.

6. **PLE-Überlaufkaskade — vier Stufen statt zwei.** Zielbild (Peuqui, 06.09.,
   bekräftigt 08.09.): die PLE-Tabelle läuft über wie ein Glas —
   **VRAM der Rechenkarten → VRAM überschüssiger Karten → gepinnter Host-RAM →
   SSD**, jede Stufe bis zu ihrem gemessenen Budget, hardware-agnostisch. Ob
   die freie Karte oder der Host-RAM die zweite Stufe wird, ist offen: der Host
   ist ein Hop weniger, die freie Karte hat mehr Platz. **Der Vergleich, der
   zählt, ist freie GPU gegen Platte** — 30 GB Host-RAM reichen für 50,7 GiB
   PLE nicht.

   **Warum nötig:** Flash-Next-PLE = 50,7 GiB (ein Tensor, 128 Shards). Bei
   TP2×PP2 liegt GPU 4 (V100, 32 GB) brach. Der MTP-Betriebspunkt hängt bei
   MML 16384, weil PLE die RTX-Stufe füllt. Die Kaskade erlaubt MTP **und**
   großen Kontext — und macht künftige, noch größere Modelle unterbringbar.

   **Wo im Code (geprüft 08.09.):**
   - Die eigentliche Arbeit liegt im Platzierungsplaner
     `plan_ple_placement`/`PLEPlacement` (`common/ple.py`) und im Gather in
     `Qwen4ExpPinnedHostEmbedding` (`nvidia/ple_layer.py`), auf dem
     #528-Code — von zwei auf vier Stufen erweitern, Budgets aus
     `mem_get_info` je Gerät.
   - **Block A** in `config/vllm.py` (`if sm70_flash_v100_baseline:`) ist die
     Stelle, an der die Stufen scharfgeschaltet werden:
     `_apply_sm70_qwen38_hybrid_ple_defaults` setzt `VLLM_SM70_QWEN38_HYBRID_PLE`,
     `VLLM_PLE_CPU_OFFLOAD` und `VLLM_PLE_DISK_OFFLOAD`. Dort käme die Vorgabe
     für eine neue Stufe hin. **Achtung:** der Aufruf hängt an
     `_is_sm70_qwen38_nomtp_dual_compile_contract` — er greift nur **ohne MTP**
     und feuert bei unseren k=4-Läufen gar nicht; dort kommt die Platzierung
     über `PLE_HOST_GIB` von der Kommandozeile.
   - **Block B** (`if sm70_flash_0dot3_compile_graph:`) ist **nicht** beteiligt,
     der setzt nur Compile- und Broadcast-Vorgaben.
   - Stufen jenseits des Rechen-VRAM sind nicht graph-capturable (kein P2P):
     Zeilen für die bekannten nächsten Token-IDs vor dem Decode-Schritt in
     einen Puffer auf der Rechenkarte vorabholen, Prefill eager.
   - Zeilen nach Token-ID aufteilen (niedrige IDs = häufige Token bei BPE),
     dann hält die schnellste Stufe automatisch die heißen Zeilen.

   **Reihenfolge — ausdrücklich festgelegt (Peuqui, 08.09.):** Die Kaskade wird
   erst angegangen, wenn die laufende Turing-Arbeit **maximiert, optimiert und
   als Pull Request veröffentlicht** ist. Vorher nicht anfangen.

7. **QUASAR-QAT ist in diesem Stack unbrauchbar — Ursache offen** (09.09.).
   1Cats DFlash2-Referenzcheckpoint `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4`
   (ModelScope-Empfehlung in ihrer `RELEASE.md`) degeneriert bei uns:
   an jeden Satz wird derselbe Nachsatz angehängt, danach bricht das Modell
   ohne Antwort ab. **Ausgeschlossen sind** DFlash2 (tritt ohne jede
   Spekulation auf), das Chat-Template (tritt über `/v1/chat/completions` mit
   `enable_thinking` genauso auf), die Kontextlänge (32k wie 256k) und das
   Sampling (greedy in beiden Fällen). Der verbleibende Verdacht ist der
   Quantisierungspfad: QUASAR kommt über `compressed-tensors`, RadixArk über
   `modelopt`. Dazu passt, dass QUASAR auf Turing gar nicht erst lädt —
   `gptq_marlin_repack` verlangt Ausgabebreiten als Vielfache von 64, eine
   Schicht hat 8.240. 1Cat baut vLLM selbst und weist für den Checkpoint
   Qualitätsgates nach (MBPP 32/32); wir fahren Wheel + Overlay.
   Nachfahrskript: `tools/mtp-diagnostics/quasar_1cat.sh`.
   **Folge: gemessen wird auf RadixArk.**

8. **Turing-Tempolücke bei DFlash2 — GESCHLOSSEN** (09.09. abends).
   69,22 → **72,72 tok/s** auf 2× RTX 8000, Text-SHA unverändert
   `0106659946c064b1`, Annahmelänge unverändert 3,353. Der Abstand zur V100
   (74,02) ist von 6,7 % auf **1,8 %** gefallen. Die V100 selbst bleibt
   unverändert (74,09 → 74,02).

   **Die Ursache war NICHT die MMA-Form.** Der Auftrag aus der vorigen
   Übergabe — sm75-Variante der Skinny-Kernel, weil Turing auf `m16n8k8`
   ausgelegt ist und `m8n8k4` nur ausführt — ist gemessen widerlegt. Bei
   gleicher FLOP-Zahl auf der RTX 8000, Registeroperanden, kein Speicher im
   Innenloop:

   | Form | 72×4 Warps | 144×4 | 288×8 |
   |---|---:|---:|---:|
   | `m8n8k4` | 37,6 | 44,2 | **45,2** TFLOPS |
   | `m16n8k8` | 39,9 | 43,6 | **45,0** TFLOPS |

   Ununterscheidbar — und der `m8n8k4`-Arm ist dabei sogar benachteiligt (eine
   abhängige Akkumulatorkette gegen zwei unabhängige). Turing führt Voltas MMA
   mit voller Tensorkern-Rate aus. Die absoluten 45 TFLOPS sind latenzbegrenzt
   und keine Dachlinie; für den Formvergleich unter identischen Bedingungen
   taugen sie. Werkzeug: `mma_probe.cu` (verifiziert beide Fragment-Layouts
   gegen eine CPU-Referenz, max|err| = 0) — der Ersatz für das verschollene
   `mma8_probe.cu`.

   **Die Ursache war die Zeilenstreuung der Aktivierungen.** Die
   A-Fragment-Abbildung von `m8n8k4` gibt Lane L die Zeile
   `(L&3)+((L&16)?4:0)`, ein Warp braucht also acht Aktivierungszeilen je
   16-k-Gruppe. Aus `x[M][K]` gelesen liegen die `K*2` Byte auseinander:
   **acht 128-B-Zeilen für 256 verschiedene Bytes**, während die Gewichte
   daneben in einem zusammenhängenden Zug kommen. Der L1 zahlt je berührter
   Zeile, nicht je gewünschtem Byte — und die Kosten wachsen mit M. Das ist
   die gesamte M=1→M=8-Steigung.

   Belegt mit einer Sonde, die dieselbe Quelle zweimal baut und nur die acht
   Zeilenzeiger auf Zeile 0 zusammenlegt (Ergebnis absichtlich falsch, nur die
   Zeit zählt):

   | M=8 | V100 | RTX 8000 |
   |---|---:|---:|
   | `1536,5120` | 1,08× | 1,66× |
   | `5120,3584` | 1,19× | **2,00×** |
   | `5120,62080` | 1,00× | 1,30× |

   Bei M=1 ändert die Sonde nichts (1,00×) — die Kontrolle, die sagt, dass sie
   die Streuung misst und nicht sich selbst. Turing leidet drei- bis fünfmal
   stärker als Volta, weil sein unified L1 96 KB hat und Voltas 128 KB.

   **Mit Zählern bestätigt (10.09., nach `ncu`-Freischaltung).** Die
   indirekte Herleitung ist damit nicht mehr nötig — `skinny_nvfp4_qpn2` auf
   der RTX 8000, `5120,4096`, alles gleich außer der Datenlage:

   | | ungepackt M=4 | ungepackt M=8 | gepackt M=8 |
   |---|---:|---:|---:|
   | Lade-Anfragen | 163.840 | 163.840 | 163.840 |
   | DRAM gelesen | 11,85 MB | 11,89 MB | 11,89 MB |
   | L1-Sektoren | 696.310 | 1.022.952 | 1.024.000 |
   | **L1-Wavefronts** | 779.710 | **1.447.064** | **451.847** |
   | Laufzeit | 28,3 µs | 46,1 µs | **26,4 µs** |

   Gleiche Befehlszahl, gleiche DRAM-Bytes, **gleiche Sektorzahl** — die
   einzige Größe, die sich bewegt, sind die Wavefronts, und die Zeit folgt
   ihnen. Der Pack drückt sie um Faktor 3,2. **Sektoren sind die falsche
   Währung**, Wavefronts sind die richtige: der L1 zahlt je zusätzlich
   berührter 128-B-Zeile und Ladebefehl, nicht je 32-B-Sektor. Mit 451.847
   Wavefronts auf 163.840 Anfragen liegt der Kernel bei 2,76 je Befehl, der
   bauartbedingte Boden sind 2.

   Auf der V100 dieselbe Bewegung, dieselbe Ursache, viel kleinere Wirkung:
   780.409 → 1.437.501 Wavefronts, aber nur +9,3 % Zeit (128 KB L1).

   **Fix: Block-Pack der Aktivierungen** (`skinny_pack_x8` in
   `kernels/skinny_kernels.cu`). x wird in `xb[K/16][8][16]` umgelegt — je
   k-Gruppe ein zusammenhängender 256-B-Block mit allen acht Zeilen. Der Warp
   berührt dann zwei Zeilen statt acht. Reine Datenlage: dieselben Werte,
   dieselbe Reihenfolge, dieselben Register, dieselbe fp32-Akkumulation.
   **Bitgleich** — im Kernel-A/B über sieben Formen × M 1..8 null Abweichungen,
   und end-zu-end derselbe Text-SHA auf beiden Kartentypen.

   Kernel-Gewinn unter Graph-Replay (Serving-Regime), Trunk-Summe bei M=8:
   **1,45× auf der RTX** (sm75-Build) bzw. **1,39×** mit dem sm_70-Build, den
   die Produktion tatsächlich fährt; **1,04×** auf der V100. Einzelne Formen
   bis 1,62×. Die M-Steigung ist weg: `5120,4096` steht jetzt über M=1..8
   durchgehend bei 90–91 % der Leseobergrenze statt 99 % → 61 %.

   **Schwelle `qpn2_pack_min_m()`:** sm75 ab M=5, sm70 ab M=8. Darunter kostet
   der eigene Start des Packs mehr, als die Streuung dort wert ist (gemessen:
   4–13 % Verlust). Die beiden Zahlen unterscheiden sich, weil die beiden L1
   sich unterscheiden — es ist eine Schwelle, keine Kernel-Gabelung: ein
   Kernel, ein Layout, eine Verzweigungsstelle. Das Layout ist
   Template-Parameter (`PACKED`), nicht Laufzeit-Stride: Strides als Argumente
   machten aus dem Gruppen-Offset ein IMAD im Innenloop und kosteten die V100
   1–5 %.

   **`qpn8` wurde geprüft und bewusst NICHT umgestellt.** FP8 liest doppelt so
   viele Bytes je Gewicht, ist also viel stärker DRAM-gebunden und lag mit
   81–90 % der Dachlinie schon fast oben. Der Pack brachte 1,05× auf der RTX
   und **kostete 4 % auf der V100** — der Start ist dort nicht bezahlt.

   **Fünf Erklärungen sind damit gemessen widerlegt** — nicht erneut
   verfolgen: QPN8-Rerank (kein topk-Kernel im Profil), AllReduce (auf beiden
   Karten Grundlast), die Compile-Vorgaben (+0,5 %), Marlin als Skinny-Ersatz
   (37 % langsamer), und jetzt die MMA-Form.

   **Was von den 435 ms übrig ist:** `skinny_fp8_qpn8` (217 ms) ist
   Bandbreite, kein Rechenwerk — die RTX schöpft dort 86 % ihrer
   Leseobergrenze aus, die V100 nur 81 % ihrer eigenen (Abschnitt Hardware).
   Der Rest ist geholt.

9. **Kurze K-Formen: strukturell begrenzt, Hebel durch Bitgleichheit
   gesperrt** (10.09.). `1536,5120` steht bei 54 % der Leseobergrenze,
   `5120,2048` bei 74 % — schon bei M=1, also unabhängig von der
   Zeilenstreuung. Zähler bei M=8, gepackt:

   | | `1536,5120` | `5120,2048` | `5120,4096` |
   |---|---:|---:|---:|
   | Grid / Block | 160 / 512 | 64 / 1024 | 128 / 512 |
   | Wellen je SM | 1,11 | 0,89 | 0,89 |
   | belegte Warps | 92,7 % | 96,7 % | 88,1 % |
   | DRAM-Durchsatz | 47,5 % | 59,5 % | 74,8 % |

   **Nicht die Occupancy** (88–97 %). Kein Gitter füllt die Maschine auch nur
   einmal: bei `5120,2048` bekommen acht der 72 SMs überhaupt nichts, bei
   `1536,5120` kostet ein Rest von 16 CTAs die volle Zeit einer zweiten Welle.
   Dazu das schlechte Verhältnis von Arbeit zu Barriere — bei K=1536 und
   SPLITK=16 dreht ein Warp sechs Durchläufe und zahlt dann eine
   `__syncthreads()` plus 16-fache Smem-Reduktion.

   Der Hebel wäre SPLITK=8 (gemessen 1,18× bzw. 1,16× auf den beiden Formen),
   **aber SPLITK ändert, welche Teilsummen wo gebildet werden — die
   Bitgleichheit fällt.** Wert: die beiden Formen sind ~26 % der
   Gewichtsbytes, bei ~16 % Gewinn also ~3,6 % der `qpn2`-Zeit und
   **rund 0,5 % end-zu-end**. **Entscheidung Peuqui, 10.09.: nicht machen** —
   der Verifikationsanker ist mehr wert. Ein bitgleicher Ersatzweg existiert
   nicht: mehr Arbeit je CTA halbiert das Gitter und verschlimmert die
   Wellen-Quantisierung.

10. **Quantisierter DFlash2-Entwurfskopf — ERLEDIGT, +6,1 % auf der RTX**
   (10.09.). Der ausgelieferte Entwurfskopf `incoai/Qwen3.8-27B-DFlash2` ist
   **unquantisiert** (keine `quantization_config`, 3,6 GB fp16). Im
   Decode-Profil kostete er `turing_fp16_s1688gemm` mit **272,6 ms / 600
   Aufrufen** — fünf je Vorwärtsschritt, passend zu den fünf
   `target_layer_ids`. Der Gegentest mit MTP statt DFlash2 zeigt den Kernel
   **überhaupt nicht**, damit ist die Zuordnung bewiesen.

   Gefahren wird jetzt **`maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal`**
   (1,44 GiB, modelopt, aus genau diesem Kopf abgeleitet, identische
   `dflash_config`). Umschaltbar über `DRAFT=` in `speed_dflash.sh` und
   `prof_dflash.sh`.

   | Karten | fp16-Kopf | NVFP4-Kopf | |
   |---|---:|---:|---|
   | 2× RTX 8000 | 72,72 | **77,13** | +6,1 % |
   | 2× V100 | 74,02 | **76,33** | +3,1 % |

   Text-SHA in allen Läufen unverändert `0106659946c064b1`, Annahmelänge
   3,353 → 3,325 (−0,8 %). **Ein quantisierter Entwurfskopf kann die Ausgabe
   nicht verändern:** angenommen wird ausschließlich, was das Zielmodell bei
   greedy ohnehin erzeugt hätte. Er kostet Annahmerate, nie Korrektheit.

   **Nötig war ein Patch**, sonst bootet es nicht:
   `DFlashQwen3Model._build_context_kv_buffers` legt die K/V-Projektionen
   aller Schichten in eine Matrix zusammen und schneidet dafür Zeilen aus dem
   **rohen** `qkv_proj.weight` — am `quant_method` vorbei. Bei NVFP4 sind das
   die gepackten Codes (`[N, K/2]`), die Fusion kommt halb so breit heraus:
   `mat1 and mat2 shapes cannot be multiplied (2048x5120 and 2560x5120)`.
   Der Override in `fork_patches_150/qwen3_dflash2.py` baut die Fusion beim
   **ersten Gebrauch** über `quant_method.apply` mit einer Einheitsmatrix neu
   (packformat-unabhängig, wie beim Ziel-LM-Head). Faul, weil
   `_build_fused_kv_buffers` am Ende von `load_weights` läuft — also bevor
   `process_weights_after_loading` fertig ist. Kosten: rund 52 MB je Rang und
   fünf GEMMs einmalig. Der Decode-Pfad bleibt quantisiert, dort sitzt der
   Gewinn; die Kontextvorberechnung ist Prefill-Arbeit.

   **Als 1Cat-PR #592 eröffnet** (10.09.) — gegen `origin/main` `0a0d4d67`,
   dort unverändert vorhanden. Der Upstream-Fix sitzt in der Basisklasse
   (`vllm/model_executor/models/qwen3_dflash.py`), nicht als Override, mit
   neuer Testdatei `test_dflash2_context_kv_quantized.py` und Gegentest.
   https://github.com/1CatAI/1Cat-vLLM/pull/592 — Entwurf mit allen Belegen:
   `upstream-contrib/03-1cat-issues/pr-dflash-quantized-draft-context-kv.md`.
   Solange #592 nicht gemergt ist, bleibt der Override in unserem Overlay
   nötig; nach dem Merge kann er raus.

11. **`prof_prefill.sh` ist auf dieser nsys-Version nicht lauffähig** (09.09.).
   Es übergibt `--output` an `nsys launch`; nsys 2022.4.2 nimmt die Option nur
   bei `nsys start` („unrecognised option"). In `prof_dflash.sh` ist es
   korrigiert, in `prof_prefill.sh` noch nicht — STAND.md empfiehlt das Skript
   an mehreren Stellen.


12. **DFlash2 steht seit 10.09. abends in llama-swap** als
    `Qwen3.8-27B-NVFP4-DFlash2-vllm` und bootet unter 256K mit Prefix-Caching
    (Tabelle „Produktive llama-swap-Einträge"). Offen bleiben die Messungen
    unten. Der bisherige Eintrag `Qwen3.8-27B-NVFP4-vllm` fährt weiter MTP. Der
    vLLM-Eintrag `Qwen3.8-27B-NVFP4-vllm` in
    `~/.config/llama-swap/config.yaml` fährt **MTP k=3** mit 256K Kontext
    und Prefix-Caching; der Hauptpfad für den 27B ist ohnehin **llama.cpp**
    (`Qwen3.8-27B-MTP-UD-Q8_K_XL.gguf`, `--spec-type draft-mtp`). Bevor
    DFlash2 dort eingetragen wird, zwei Messungen unter
    Produktionsbedingungen: (a) langer Kontext mit Prefix-Caching — 1Cat hat
    zu DFlash2 bei 256K das offene Issue #467 (Allocator-Reset,
    KV-Überschätzung); (b) der Vergleich gegen den produktiven llama.cpp-Pfad
    auf denselben Karten. Der Maßstab steht in der Gedächtnisnotiz
    `project_vllm_vs_llamacpp_criterion`: vLLM muss llama.cpp **schlagen**.
    Die llama-swap-Konfiguration ist Peuquis Datei — händisch oder mit
    Sicherung und Freigabe, nie per Skript.

13. **Decode-Profil nach Block-Pack und Entwurfskopf** (10.09., RTX-Paar,
    `prof_dflash.sh packed 0,2`, ein Rang, 5.395 ms GPU-Zeit im Fenster):

    | Posten | ms | Anteil | Stand |
    |---|---:|---:|---|
    | `ncclDevKernel_AllReduce` | 1.714 | 31,8 % | **größter Posten, nie untersucht** |
    | `skinny_nvfp4_qpn2` | 1.303 | 24,2 % | fertig, 87–98 % der Dachlinie |
    | `skinny_fp8_qpn8` | 889 | 16,5 % | DRAM-gebunden, nicht anfassen |
    | `cutlass_75_wmma…f16_16x16` | 291 | 5,4 % | zugeordnet 11.09., siehe unten |
    | `turing_fp16_s1688gemm` | 273 | 5,1 % | war der fp16-Entwurfskopf, erledigt |
    | `fused_sigmoid_gating_delta_rule` | 180 | 3,3 % | |
    | `ncclDevKernel_AllGather` | 178 | 3,3 % | |
    | `skinny_pack_x8` | 29 | 0,5 % | der Pack: spart 310, kostet 29 |

    Das Profil ist vor dem Wechsel des Entwurfskopfs aufgenommen; mit dem
    NVFP4-Kopf fällt die `turing_fp16_s1688gemm`-Zeile, der Rest steht.

    **AllReduce:** 102 µs je Aufruf für 80 KB Nutzlast ist latenz-, nicht
    bandbreitendominiert; ohne P2P läuft es über Host-Staging. Unter MTP
    kostet dasselbe AllReduce 58,7 µs, weil k=3 die halbe Nutzlast bedeutet.
    Billigster erster Versuch: NCCL-Umgebungsschalter (`NCCL_ALGO`,
    `NCCL_PROTO`, Puffergrößen) — reine Env-Experimente.

    **Cutlass-fp16-GEMM — zugeordnet (11.09., aus den vorhandenen
    nsys-Berichten, Aufrufzahlen je Grid):** Es ist kein einzelner GEMM,
    sondern eine Kernelfamilie für alles, was fp16 bleibt. Der Großteil ist
    der **unquantisierte Entwurfskopf** — bei DFlash2 der damalige fp16-Kopf
    (Gruppen mit 5, 7 und 10 Aufrufen je Verify-Schritt), unter MTP der
    MTP-Block, den der RadixArk-Checkpoint von der Quantisierung ausnimmt
    (`exclude_modules: mtp*`; fünf Gruppen mit je ~408 = 136 Schritte × 3
    Entwürfe). Im **Zielmodell** liegt nur `in_proj_ba` (die GDN-Gating-
    Projektionen a/b, im Checkpoint unquantisiert): 48 Aufrufe je Schritt
    (5.715 = 48 × 119), 5,5 µs je Aufruf, rund 0,6 %. Die fusionierte
    1Cat-Route (`fp8_qpn8_gemm_ba_split_sm70_out`) greift nur bei genau einem
    Token, im Spekulations-Verify also nie. Die frühere Annahme „im
    Zielmodell, weil auch unter MTP" war falsch — auch der MTP-Block ist fp16.
    Folge: Mit dem NVFP4-Entwurfskopf sollte der große Teil bei DFlash2 schon
    weg sein (Nachmessung mit `prof_dflash.sh` und `DRAFT=maurienne` steht
    aus). Für 27B-MTP wäre ein quantisierter MTP-Block der Hebel, wie bei
    Flash-Next (MTPQ).

14. **Block-Pack auf 1Cats eigene QPN2-Kernel übertragen** (10.09. geprüft).
    1Cats QPN-Kernel (`csrc/sm70_turbomind/ops/nvfp4_qpn2_sm70.cu` u. a.)
    sind eine **Übernahme aus v100-skinny** (1Cat-PR #403, 29.08., Lizenz
    `LICENSE.v100-skinny`), Stand Ende August. Deren Aktivierungszugriff hat **exakt
    dieselbe Zeilenstreuung**: Zeilen 184–186 und 295–297 laden
    `input + row * k + group * 16` — dasselbe Muster, das wir behoben haben.
    Ein PR wäre also eine Portierung auf deren Kernel samt eigener Messung
    auf Turing; unser Diff gilt nicht wörtlich. **Geklärt 10.09.: auf Turing
    fährt 1Cat ihn nicht** — die TurboMind-Weiche (`sm70_turbomind.py`) prüft
    exakt SM70. Ein Angebot müsste den Turing-Pfad mitbringen; auf der V100
    brachte der Pack nur 1,04×.

15. **PR-Pakete** (10.09. abends, Freigaben Peuqui: 8, 9, 10, 11, 12, 13).
    Entwürfe in `upstream-contrib/03-1cat-issues/`: editable-Bau
    (`pr-editable-soabi-modules.md`, Worktree `1Cat-vLLM-editable-pr`,
    Belegbau auf frischem main grün) und DFlash2 unter SM80
    (`pr-dflash2-pre-sm80-worker-device.md`, Worktree `1Cat-vLLM-pr-dflash2`,
    CPU-Test samt Gegentest grün, GPU-Messung auf main + #572 steht aus).
    Beide nicht gesendet. Frisches main ist per `PYTHONPATH` auf den
    Belegbau-Worktree lauffähig — damit lassen sich PRs vorher/nachher auf der
    echten Hardware messen. Ursprüngliche Einordnung: Aus 1Cat plus
    unseren offenen PRs entsteht NICHT unser System: der Overlay (60 Dateien,
    +5.067 Zeilen) steckt in keinem PR; Skinny-Quelltext, sm75-FA und
    TileLang-Fix liegen ganz außerhalb von 1Cat. 1Cat selbst ist ein
    eigenständiger vLLM-Schnappschuss (Historie ab 29.08., Upstream wird von
    Hand portiert) — niemand muss vLLM dazupicken. Kandidaten:
    (a) editable-Bau-Fix (`f03a7102`, klein, plus sechs mypy-Befunde in
    `setup.py`); (b) Routenzähler (`VLLM_SKINNY_ROUTE_COUNT_FILE`) in den
    Produktionskonfigurationen → Skinny-Angebot an 1Cat: Turing-Pfad,
    Block-Pack und MoE-Backend in deren kompilierte QPN-Kopie; (c) TileLang-Pin
    0.1.10 → ≥ 0.1.12 — der Geräte-Fix ist dort seit v0.1.12 upstream, vorher
    auf V100 und RTX testen (GDN und mHC); (d) Paket gemischte Hardware:
    Gates pro Gerät, Turing-Zweig, FA2-Lader, sm75-FA als eingebundener Fork
    (so wie 1Cat zhinianqins FA einbindet — Upstream-FA2 ist nur Ampere+,
    flash-attention #190 seit 01.09. unbeantwortet); (e) Overlay-Inventur.
    Skinny-Nutzen gemessen: Marlin als Ersatz auf der RTX 37 % langsamer
    (Punkt 8); auf der V100 rechnet beim 27B TurboMind, bei DeepSeek laufen
    die MoE-Experten auf allen Karten über Skinny.

16. **Befunde vom 10.09.:**
    - ~~`flashnext_qual.sh`/`flashnext_ab.sh` beenden jeden vLLM-Worker der
      Maschine per `pgrep`~~ — ERLEDIGT, nur noch die eigene Prozessgruppe.
    - `VLLM_SKINNY_*` stehen nicht in `envs.py` → Startwarnung. Wird im
      Skinny-PR mit angemeldet (Peuqui 10.09.).
    - ~~Werkzeuge mit fester `.venv-sm70-130`~~ — ERLEDIGT.
    - Overlay-Inventur: `upstream-contrib/OVERLAY-INVENTUR.md`. Neun
      Merge-Reste, darunter ein echter Bug (`_custom_ops.py`: MLA-Modelle
      stürzen beim Chunked-Context-Prefill ab), eine abgeschaltete
      Upstream-Optimierung (`speculative.py`, Qwen4Exp-MTP index_share) und der
      E5-Cache mit Vorgabe an. Freigabe Peuqui 10./11.09.: Reste
      bereinigen, index_share per A/B auf Flash-Next.
    - **Aufräumen (Befunde 1, 3–8) ABGENOMMEN und COMMITTET 11.09. abends**
      (`43ccb9b8`, Tag `verified-2026-09-11`, gepusht). Abnahme mit und
      ohne Aufräumen am selben Abend, gleiche Skripte: 27B-MTP Text gleich
      der Referenz; 27B DFlash2 SHA `0106659946c064b1` auf RTX und V100,
      Annahmelänge 3,325 (die 3,353 vom Vormittag war ein Einzelausreißer,
      drei weitere V100-Läufe mit und ohne Aufräumen: 3,325); Flash-Next je
      drei Läufe, q1/q2 sechsmal sauber, q3 2/3 mit gegen 1/3 ohne
      Aufräumen (Rohtext-Sonde, siehe Fallstrick unten); DeepSeek PP5 8/8
      byteidentisch. Offen bleibt Befund 2 (index_share, A/B).
    - **Befund 9, E5-Cache: GEMESSEN 11.09. abends** (`handover/2026-09-11/
      scripts/abnahme2/e5_ab.sh`, exakter Produktionsbefehl 27B-MTP, V1-Runner,
      RTX-Paar, 6 Anfragen à 600 Token, Median der Läufe 2–6, Decode vom
      ersten bis zum letzten Token): **E5 aus 70,17 tok/s, E5 an 73,11 tok/s,
      +4,2 %**, Text in allen zwölf Läufen byteidentisch (`dc8d4f97f6910e01`).
      Nachweis, dass der Cache feuerte: beide Ränge `[e5-cache] captured:
      groups=4` und `[e5-v2] persistent prepare active`, zurückgewiesen nur
      drei Aufwärmrunden. Boot mit E5 an 410 s statt 195 s (anderer
      Compile-Cache-Schlüssel, #536). Einordnung: E5 wirkt nur im V1-Runner;
      der schnellste 27B-Pfad ist DFlash2 im V2-Runner (77 tok/s auf dem
      RTX-Paar) und nutzt ihn nicht. **Entscheidung Peuqui 11.09. nachts:
      RAUS** („statt MTP lieber DFlash2, dann schmeiß es raus"). Ausgebaut:
      die zehn E5-Hunks in `gpu_model_runner.py` rückwärts angewendet
      (`handover/2026-09-11/scripts/abnahme2/e5_remove.diff`), 14.357 →
      13.267 Zeilen, keine E5-Reste, ruff auf der Datei jetzt sauber (die
      zwei alten Fehler steckten in E5). Gegen main bleiben 27 Hunks,
      +353/−41: #574-Trim, PP-Draft-Broadcast, Mamba-Kopierfunktionen, PLE,
      `only_gids`, und die drei lokalen Haken `STAGED_PREP_SPEC_FORCE`,
      `GDN_SLOT_DEBUG`, `MTP_THINK_ONLY`. E5 war quantisierungsunabhängig,
      aber an V1 + MTP + Single-Stream gebunden und auf QSA inkompatibel. Der
      längere Boot im A/B (410 s) war der Compile-Cache-Schlüssel (#536), kein
      E5-Preis. `VLLM_SM70_E5_CACHE=0` aus den 8 llama-swap-Einträgen entfernt
      (11.09. 21:31, Freigabe Peuqui, Sicherung `backups/config.yaml.bak-
      2026-09-11-vor-e5`; YAML geprüft, 25 Modelle) — jeder vLLM-Eintrag
      bootet beim nächsten Laden einmal kalt (Compile-Cache-Schlüssel, #536).
      Committet: work-main `433dfa10` = Tag `verified-2026-09-11b`.
    - **Befund 2 (index_share) ANGEWENDET 11.09. nachts**: `speculative.py`
      = Upstream + DSV4-Guard, Qwen4Exp-MTP fährt `index_share_for_mtp_
      iteration=True` wie Upstream. Flash-Next, Rohtext-Sonde, drei Läufe:
      q1/q2 sauber, **q3 3 von 3** (heute Abend mit `False`: 3 von 6),
      Annahmelänge 2,70–3,12, Decode 27,6–31,8 tok/s — gleiches Band wie mit
      `False`. Kein Beweis für besser, kein Hinweis auf schlechter, Upstream-
      Vorgabe bleibt.
    - **Abnahme E5-Ausbau + Befund 2 (driver2.sh, 11.09. 20:42–21:25):**
      27B-MTP Text gleich der Referenz, 0 E5-Zeilen im Log, Boot 116 s;
      DFlash2 V100 76,27 tok/s, SHA `0106659946c064b1`, Annahme 3,325;
      Flash-Next 3/3; DeepSeek PP5 8/8 byteidentisch. Committen auf Ansage.
    - **Die Rohtext-Sonde `flashnext_qual.sh` misst nicht den
      Produktionspfad** (11.09.). Sie schickt Kontext + Frage roh an
      `/v1/completions`, ohne Chat-Template und ohne `enable_thinking`. Das
      Modell muss selbst entscheiden, ob es `<think>` öffnet; in den
      Ausfällen tat es das nicht (oder erfand vorher eine weitere
      Nutzeranweisung) und rutschte ohne Denkblock in Schleife oder
      Erfindung. Die q3-Ausfallquote der Sonde ist deshalb NICHT die
      Quote in AIfred. Chat-Variante mit Template, `chat_template_kwargs`
      und Reasoning-Parser wie im llama-swap-Eintrag:
      `handover/2026-09-11/scripts/abnahme2/flashnext_qual_chat.sh`;
      wandert nach Bewährung neben die alte nach `tools/mtp-diagnostics/`.

      **Ergebnis der Chat-Sonde (11.09. abends, drei Läufe, Aufräum-Stand,
      `MAXTOK=3300` — mehr passt bei 13.053 Prompt-Token nicht in MML 16384,
      mit 8.000 kam HTTP 400):**

      | Frage | Lauf 1 | Lauf 2 | Lauf 3 |
      |---|---|---|---|
      | q1 Quantenphysik | 30 Sätze, sauber, Denkblock 7,1k Zeichen | sauber, 7,4k | sauber, 6,4k |
      | q2 Regenbogen | 30 Sätze, sauber, 7,1k | sauber, 5,0k | sauber, 4,5k |
      | q3 Kuanda | **kein Inhalt**, Denkblock 12,9k Zeichen bis zum Limit | kein Inhalt, 13,4k | kein Inhalt, 12,5k |

      Decode 37–44 tok/s, Annahmelänge 2,8–3,6, Reasoning-Parser `qwen3`
      aktiv (Denkblock im Feld `reasoning`). q3 ist damit auf dem
      Produktionspfad **nie beantwortet**: Das Modell denkt auf Deutsch
      („Kuanda-Effekt? Kenne ich nicht… vielleicht erfunden…"), entwirft
      dreißig Sätze über einen undefinierten Begriff, zählt sie nach und
      verwirft sie wieder, bis das Tokenlimit greift; Coandă kommt in keinem
      der drei Denkblöcke vor (im Rohtext-Modus dachte es auf Englisch und
      erkannte Coandă sofort). **Befund über den Betriebspunkt, nicht über
      das Aufräumen:** Flash-Next mit Denken unter MML 16384 hat bei 13k
      Kontext keinen Platz für eine grübelnde Antwort — in AIfred mit
      Tool-Schemata und History ist das der Alltag. Offen: q3-Verhalten mit
      größerem Ausgabebudget (braucht MML > 16384, also PLE-Kaskade oder
      kleineren Kontext) und ob `enable_thinking=false` für solche Fälle die
      bessere Produktionswahl ist.

17. **DeepSeek-Eintrag läuft als PP5** (10.09.): TP1 PP5 über alle fünf
    Karten, `CUDA_VISIBLE_DEVICES=0,1,4,3,2`, Partition 11,8,8,8,8,
    `--kv-cache-dtype fp8`, DSpark k=5. Abnahme über llama-swap bestanden;
    13k-Vorkontext: 15,1 / 19,1 / 15,4 tok/s, Prefill ~120 tok/s.
    **Kontextgrenze: 65.536** (11.09., in llama-swap eingetragen, Sicherung
    `backups/config.yaml.20260911-031418-vor-dsv4-ctx64k`): 600 Blöcke
    (KV 93.622 Token), `VLLM_SM70_INDEXER_PREFILL_TILE_MB=64`. Belegt mit
    einem 64.349-Token-Prompt: TTFT 541 s, Decode 12,3 tok/s, Antwort
    kohärent, V100-Spitze 32.200 von 32.492 MiB nutzbar (292 MiB Luft).
    Die V100-Stufen sind die Grenze, nutzbar sind dort 31,73 GiB. Der Weg „zu hohe max-model-len, Wert aus dem Fehler"
    trägt hier nicht — FlashMLA-Sparse reserviert beim Start
    5 × max-model-len × 576 × 2 Byte (`flashmla_sparse.py:235`), bei 1M
    ~6 GB, der Profillauf stirbt vor der KV-Rechnung. 131.072 bootet
    (KV 232k Token), stirbt aber bei ~100k Kontext im Indexer-Prefill (OOM,
    156 MiB Kachel). Leerlauf wächst ~22 MiB je 1.000 Token max-model-len.
    Hebel: `VLLM_SM70_INDEXER_PREFILL_TILE_MB` (Vorgabe 192; unter ~32k
    Kontext wirkungslos, Referenz-Hashes unberührt).

18. ~~**Drei DFlash2-PRs von 1Cat fehlen in work-main**~~ — ERLEDIGT 10.09.:
    Merge 82301e6b (#586 #587 #589), SHA auf beiden Kartenpaaren gleich,
    76,28 (V100) / 76,65 (RTX) tok/s; auf dem RTX-Pfad neutral, weil die PRs
    SM70-Attention ändern. Getaggt `verified-2026-09-10b`.
