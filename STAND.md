# Betriebsstand v100-skinny

**Stand 2026-09-11 abends, Punkt 6 aktualisiert 2026-09-15 spät.** Dieses Dokument beschreibt, WIE der Stack heute
läuft. Warum er so läuft, steht in `docs/journal/` — jede Zeile hier trägt einen
Verweis. Übergabeaufträge stehen in `HANDOVER.md`, Upstream-Beiträge in
`upstream-contrib/`.

Reihenfolge für eine neue Instanz: dieses Dokument, dann `HANDOVER.md`. Die
Logbücher im Journal sind chronologisch und groß — sie beantworten „warum ist es
so", nicht „wie ist es".

---

## Laufzeitumgebung (seit 10.09. abends; Nachträge 13.09. und 14.09.)

**Nachtrag 14.09.:** work-main `ebc5dc52` (Tag `verified-2026-09-14b`) = 1Cat main
`80c88e8d` (Merge `1d3439f2`, Tag `verified-2026-09-14`) + Overlay (77 Dateien,
+4.771/−375 gegen main) inkl. #636-Fassung von `qwen3_5_mtp.py`; PP5-
Warteschlangen-Deckel und #572-Rest entfernt. Neubau 14.09. mit `MAX_JOBS=3`
(Skript `handover/2026-09-14/rebuild_work_main.sh`, 38 min). Abnahme: 27B DFlash2
SHA `0106659946c064b1` RTX 76,72 / V100 76,48; 27B MTP TP2 73,34 (SHA gleich),
PP2 61,05; DeepSeek PP5 8/8 (neue Referenz `ds_merge0914`, alte `ds_tl014` beim
Plattenaufräumen gelöscht); Flash-Next q1/q2 sauber, Kuanda uneinheitlich.
llama-swap-Einträge: `Qwen3.8-27B-NVFP4-vllm` (MTP) wieder aktiv, DeepSeek ohne
`VLLM_SM70_ASYNC_SCHEDULING_QUEUE_DEPTH`. Details `HANDOVER.md` Nachtrag 14.09.

**Nachtrag 13.09. abends:** work-main `f6c42de7` auf fork/work-main = 1Cat main
`dfef3342` + Overlay + Paket C + PLE-Gather-Fix; Produktions-venv
`.venv-sm70-main` mit torch-Backport #173556 (`tools/torch_patches/apply.sh`,
Hash-geprüft, Original als `.orig-2.10.0` daneben); sm75-FA2 als Datei
`vllm/vllm_flash_attn/_vllm_fa2_C_sm75.abi3.so` (Bau 13.09. per ExternalProject,
Drop-in vom 03.09. als `.drop-in-0903`); Compile-Cache AN, AOT-Artefakte am
13.09. geleert und neu geschrieben. Abschnitt "Paket C im Fork und zwei
Compile-Cache-Fehler" unten. Der Rest dieses Abschnitts ist der Stand vom 10.09.

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

## Abnahme work-main-Merge auf main dfef3342 (12./13.09. nachts, Peuqui: "vorwärts fixen, nicht zurücksetzen")

**Was passiert ist:** `git merge origin/main` in work-main (`34f3f340`), ein Konflikt
(unser Erklärkommentar in `eagle/utils.py`, main hat den Fix selbst übernommen),
veralteter Overlay-Kommentar in `config/vllm.py` entfernt (`f381618a`). Die Overlay-
Teile von #572/#573 sind durch den Merge automatisch verschwunden (Dateien identisch
zu main). Editable-Neubau in `.venv-sm70-main` nach dem Rezept, 45 min, Exit 0:
`_C` 41, `_moe_C` 15, `_vllm_fa2_C` 9 Cubins, alle sm_70; sm75-FA2 und GDN-.so
unverändert daneben.

**Alt gegen neu am selben Abend** (alter Stand 911c259f in Worktree
`1Cat-vLLM-old-prod`, venv `.venv-sm70-old` = Kopie der tltest-venv + editable, keine
neuen Pakete). Alle DFlash2-Läufe Text-SHA `0106659946c064b1`:

| DFlash2 27B, greedy, 400 Token | alt | neu |
|---|---|---|
| RTX-Paar, **Produktionskopf** (maurienne RTNcal) | 76,92 (Annahme 3,325) | **76,88** (3,325) |
| V100-Paar, Produktionskopf | – | **76,42** (3,325); Referenz 10.09. 76,30 |
| RTX-Paar, incoai-Kopf (Skript-Vorgabe) | 72,68 | 73,15 / 72,77 / 72,98 |
| V100-Paar, incoai-Kopf | 74,08 | 75,09 (Tail-Cudagraphs an, main-Default) / 74,21 (aus) |

Der erste Schreck "RTX 73 statt 77" war der Entwurfskopf: `speed_dflash.sh` nimmt
als Vorgabe den unquantisierten incoai-Kopf, die Produktion und alle 77er-Läufe vom
11.09. fahren den quantisierten maurienne-Kopf. Kein Merge-Effekt. mains neue
Tail-Cudagraphs bringen auf der V100 +1,2 %; **Turing bekommt sie nicht**
(`is_device_capability((7, 0))`, exakt Volta und Gerät 0 - Kandidat für einen PR).

**Weitere Schritte:** DeepSeek PP5 Kohärenz 8/8, zwei Läufe identisch in Text,
Denkblock und Tokenzahl, identisch zur Referenz `ds_tl014`, gleiches Tempo.
Flash-Next Chat k=4: q1/q2 sauber, 37-48 tok/s; **Kuanda alt gegen neu je 3x
dieselbe Anfrage: beide 2x Zurückweisung, 1x Erfindung** - das 180B ist trotz greedy
und Seed nicht deterministisch, das Erfinden ist Modelleigenschaft, kein Merge-Effekt.
Vier produktive llama-swap-Einträge kalt (exakter Befehl) und warm (llama-swap):
alle ohne Traceback, Antworten je Eintrag in beiden Phasen identisch, 27B-Einträge
laden die sm75-FA2 auf beiden Workern; Warmstarts 122 / 149 / 301 / 519 s
(`handover/2026-09-12/prod/`). Erster Durchlauf hatte einen Skriptfehler auf meiner
Seite (relativer Ausgabepfad, prod_accept.sh wechselt nach /tmp), im llama-swap-
Journal waren die Boots trotzdem als 200 OK belegt.

**Neu von main, das uns betrifft:** `VLLM_SM70_DFLASH2_TAIL_CUDAGRAPHS` Default an
(gemessen, s. o.); Qwen3.8-NVFP4-Defaults ohne MTP (#579, greift nur ohne
Spekulation und nur bei lauter 7.0-Karten); QSA-E4M3-Skalen unkalibriert = Warnung
statt Abbruch; V100-Long-Context-Layout (#609/#610) im FA2-Target.
Skripte: `handover/2026-09-12/abnahme_*.sh`, `rebuild_work_main.sh`,
`flashnext_q3_rate.sh`, `prod_accept4.sh`. Rohdaten `~/.cache/mtp-diagnostics/qual_rb_*`,
`ds_rebuild`, `fnqc_rb`, `fnq3_q3_{old,new}`.

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

## Paket C im Fork und zwei Compile-Cache-Fehler (13.09. abends)

- **Laufzeitumgebung:** work-main `08d62424` (Paket C: FA2 für Turing als
  eigene Bibliothek `_vllm_fa2_C_sm75`, Lader pro Gerät, Volta-Ladefix) plus
  uncommitted: PLE-Gather-Fix, Volta-Testkorrektur, `tools/torch_patches/`.
  Produktions-venv `.venv-sm70-main` mit torch-Backport #173556 gepatcht
  (`tools/torch_patches/apply.sh`, `rebuild_work_main.sh` ruft es). Compile-
  Cache AN (adf3e5bd), AOT-Artefakte am 13.09. einmal geleert.
- **Abnahme Fork (prod_accept4, Phase 1 kalt):** 27B, DFlash2, Flash-Next,
  DeepSeek bestanden; Flash-Next lädt `_vllm_fa2_C` (V100-Stufe) und
  `_vllm_fa2_C_sm75` (RTX-Stufe) in einem Modell. Phase 2 warm: 27B, DFlash2,
  DeepSeek bestanden, Flash-Next ABGESTÜRZT → Fehler 1.
- **Fehler 1 (behoben):** PLE-Gather-Op nahm den Tabellenzeiger als int, Inductor
  backte ihn ins Artefakt; erster Warmstart mit Cache → illegal memory access auf
  Stufe 0. Fix: Layer-Name + `use_host_table`, Zeiger zur Laufzeit im Op.
  Beleg Flash-Next: kalt 340 s, warm1 312 s, warm2 343 s, je 6 Artefakte.
- **Fehler 2 (behoben):** torch 2.10.0 ohne Kernel-Tabelle im Artefakt → erster
  Warmstart verpuffte. Backport belegt am 27B: kalt 476 s, warm1 85 s (0
  Ladefehler), warm2 80 s. Text byteidentisch (`a3dffc7c5e9b417a`).
- **Bootzeiten RTX-Paar 27B ohne Cache (Vergleich):** kalt 451 s, warm 175 s
  (davon 104 s torch.compile, weil Cache aus); mit Cache siehe oben.
- **Turing-Tests:** `test_sm70_flash_v100_policy.py` (81, eine Erwartung für 7.5
  angepasst) + `test_sm70_e4m3_scalar_fp32.py`: 144 grün in PR-Branch und Fork;
  `test_ple.py`: 60 grün; `test_attention_backends.py` (55) und
  `test_flash_attn.py` (640) auf der RTX: siehe HANDOVER-Nachtrag.

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

### DeepSeek-V4-Flash-NVFP4 + DSpark — Produktion, Stand 20.09. früh

PP5 über alle fünf Karten (`CUDA_VISIBLE_DEVICES=0,1,4,3,2`, Partition 11,8,8,8,8;
die letzte Stufe trägt zusätzlich die drei MoE-Schichten des Drafters, real also
11/8/8/8/11 Schicht-Äquivalente), `--dtype half`, fp8-KV, 65k Kontext,
`--max-num-seqs 1`, `--max-num-batched-tokens 256`, `--num-gpu-blocks-override 900`,
DSpark K=5 (gierig), CUDA-Graph-Größe 6. Zweiter llama-swap-Eintrag
`…-Coding-K7-vllm` mit K=7 und Graph-Größe 8, sonst identisch.

| Messpunkt (temp 1.0, top_k 40) | 18.09. | 20.09. früh |
|---|---|---|
| kalter „18k"-Prefill (TTFT; der Prompt hat 21.857 Tokens) | 83 s = 263 tok/s | 16,7–18,3 s = 1.190–1.310 tok/s |
| 62k-Prefill (61.719 Tokens) | — | 49 s = 1.260 tok/s |
| TTFT auf gecachtem 18k-Präfix | 1,3 s | 0,8 s |
| Decode Prosa 18k | 210 ms/Schritt, 14–15 tok/s | 81 ms, 35–40 tok/s |
| Decode Code 18k | 22–24 tok/s | 84 ms, 50–62 tok/s |
| Decode kurz | — | 81 ms |
| 61,7k Prompt + 3.000 erzeugte Tokens | — | 132 s |

Woher das kam: Sparse-MLA als Matrixprodukt (Decode 210 → 92 ms), Präfix-Cache-
Backport vllm#44082, mHC-fp16-Wertebereich (NaN-Absturz), 128er- statt 64er-Häppchen
(83 → 46 s), gebündelter MoE-Kernel `moe_qpn` auch im Prefill (46 → 19 s), 256er-
Häppchen (19,2 → 18,3 s; 62k 55 → 49 s). Der
Prefill war zuvor praktisch nur die Per-Experten-Schleife: ~0,13 ms Kernel-Starts je
aktivem Experten, 26–32 ms je Schicht, Takt = RTX-Stufe mit elf Schichten. Profil
danach (PP0): GPU ausgelastet, MoE 48 %, dichte FP8-Linears 21 %, Sparse-Attention
~10 %, Indexer ~6 %, mHC 3 %.

Abnahme: Nadel-Test 30k und 62k je 4/4, Stresstest 120/120, Qualität bei temp 1.0
25/25, Härtetest 61,7k Prompt + 3.000 erzeugte Tokens (64,7k Gesamtkontext), V100
32.436 MiB stabil (Reserve 332 MiB). Die 256er-Häppchen liefen erst, nachdem der
Attention-Workspace des bmm-Prefills auf 128 Query-Tokens gedeckelt war (vorher
echter OOM auf den V100; der KV-Cache ist mit ~130 MiB je Stufe nicht der Hebel,
NCCL_BUFFSIZE gibt nichts frei, je RTX passt nur eine weitere Schicht; die
Blockzahl muss mit der Häppchengröße wachsen, weil der SWA-Bedarf
`sliding_window - 1 + max_in_flight_tokens` ist). Gemessen und verworfen: K=7 für
Prosa (−8 %, darum der eigene Coding-Eintrag), probabilistisches Entwurfs-Sampling
(kein Unterschied).

Decode-Profil (PP0, 17k Kontext): MoE ~0,75 ms je Schicht und Schritt (33 %), der
Indexer-Kernel `_paged_index_logits_relu_kernel` 0,91 ms je Kompression-4-Schicht
(18 %, wächst linear mit dem Kontext: ~19 ms von 91 ms bei 18k, bei 62k der
Hauptposten, Schritt dort ~180 ms). Behoben am 20.09. (Fork f22c87c9,
`VLLM_SM70_INDEXER_DECODE_CUBLAS=1`): 1Cats cuBLAS-Weg griff bei Spec-Decode mit
K > 1 nie, weil der Metadaten-Bauer den Decode dann auf eine Blocktabellen-Zeile je
Token ausflacht und der Weg genau eine Zeile verlangte; dazu ein Gerät-0-Gate. Jetzt
je Anfrage ein Gather + ein GEMM, konstant ~0,13 ms je Schicht statt 0,04–3,3 ms:
Schritt bei 18k 91 → 81 ms, bis 18k praktisch kontextunabhängig, 62k-Härtetest
175 → 132 s, kurz +2 ms (Festpreis des GEMM über die Graph-Breite).

Wohin die 81 ms je Decode-Schritt gehen (20.09., `VLLM_SM70_ASYNC_CPU_TRACE=1`, 18k):
GPU-Pipeline aller fünf Stufen ~52 ms (davon ~30 ms MoE), Entwurf 5,7 ms,
Eingabevorbereitung auf der CPU ~6,5 ms, Rest (Prozesskommunikation, zwei
gloo-Broadcasts, Warteschlangen) ~12–16 ms. Die vorderen Stufen warten 66–78 ms auf
das Ergebnis der letzten — inhärent bei Spec-Decode mit einer Anfrage.
CPU-Profil (py-spy, 15 s Decode, `benchmarks/pyspy-dsv4-decode-18k-2026-09-20.raw`):
EngineCore 99,4 % aktives Warten im Shared-Memory-Kanal, Scheduling < 1 %; letzte Stufe
79 % in der Host-Synchronisation `reference_rows.any()` von `apply_top_k_top_p_triton`
(Warten auf die GPU, dahinter nur ~0,5 ms CPU-Arbeit bis zur ohnehin nötigen
Synchronisation → kein Hebel); übrige Stufen je ~4 ms echte CPU-Arbeit je Schritt.
Der MoE-Decode-Kernel `moe_qpn` ruft die feste 8×8×4-Tensor-Kern-Kachel
(`mma…m8n8k4`) auf; die 8 Zeilen je Durchgang sind Hardware, nicht einstellbar, und er
liegt ~1,5–2× über der Bandbreiten-Untergrenze. K: je Zusatzposition ~3 ms je Schritt,
K=3/4/5 liegen für Prosa gleichauf. Profil-Tabellen: `benchmarks/torchprof-dsv4-*`.
py-spy: `~/.venv/pyspy`, Anhängen nur mit dem ptrace-Haken unter
`~/.venv/pyspy/ptrace_hook` (Dienst hat PrivateTmp, Pfade nie unter /tmp übergeben).


### Qwen3.8-Flash-Next (180B, Qwen4Exp) — geprüft 07.09., nachverifiziert 09.09.

```bash
cd /home/mp/Projekte/vllm-research/v100-skinny
VLLM_SM70_E5_CACHE=0 CUDA_VISIBLE_DEVICES=0,2,1,3 \
TURBOMIND=1 QUANT_BACKEND=turbomind \
ENV_PREFIX=/home/mp/vllm/venv \
TP=2 PP=2 K=4 GMU=0.95 MML=262144 PP_PARTITION=24,24 PLE_HOST_GIB=6 \
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

6. **PLE-Überlaufkaskade — Paket 1 (Durchstich) gebootet 15.09., Paket 2 als
   Nächstes.** Reihenfolge der Stufen VRAM → Host → freie GPU → SSD, Budgets
   konfigurierbar und dynamisch; Entwurf, Entscheidungen und Messungen in
   `docs/PLE-KASKADE-ENTWURF.md` (Abschnitte 9 und 10), Werkzeuge in
   `handover/2026-09-15/`. Vorgezogen am 15.09., weil der Mini nach jedem
   Flash-Next-Start zäh ist (12 GiB PLE gepinnt, MemAvailable ~2 GiB, Swap
   ~10 GiB).

   - **Stand Code** (1Cat-Fork, Branch `qwen4exp-ple-tier-cascade`): Schalter
     `VLLM_QWEN4EXP_PLE_STORE_DEVICE=<sichtbarer Index>` startet den
     vorhandenen PLE-Offload-Worker neben den residenten Tabellen. Die Ränge
     gathern VRAM- und Host-Stufe wie bisher, warten per `ple_offload_wait` im
     Graphen und führen die Worker-Zeilen vor dem TP-All-Reduce per `where`
     zusammen. Der Worker hält die Checkpoint-Shards nur als mmap und liefert
     in Paket 1 Nullen; meldet ein Rang Zeilen jenseits seiner residenten
     Stufen, bricht er mit „store tier is not built yet“ ab.
   - **Belegt (15.09.):** läuft mit MTP k=4, TP2×PP2, async, volle Graphen,
     0 Tracebacks; ohne Schalter 3× bitgleich zur Produktion; mit Schalter
     2–5 % langsamer, ein Prompt kippt an einem Beinahe-Gleichstand (Token 76).
     Das Zusammenführen ist unter Inductor bitgleich, die Drift kommt aus der
     Neu-Bündelung des PLE-Graphstücks.
   - **Abnahme (Peuqui 15.09.):** Bitgleiche Ausgaben sind bei Flash-Next über
     Graph-Änderungen nicht zu verlangen. Kriterium: unveränderter Pfad
     bitgleich, Kaskade fehlerfrei, Abweichungen nur an Beinahe-Gleichständen,
     Tempo dokumentiert.
   - **Fakten, die bleiben:** Tabelle hash-adressiert (jede belegte Stufe wird
     bei jedem Schritt anteilig getroffen, keine heißen Zeilen); TP-geteilt,
     je Rang 160.000.768 Zeilen; SSD-Stufe auf dem Mini planmäßig leer; TP4 ist
     keine Motivation; GPU-4-Budget = Rest nach VLM/TTS-Spitzenbedarf.

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

    **Env-Sweep GEMESSEN (12.09. nachts, Punkt 3 des Plans, RTX-Paar `0,2`,
    `handover/2026-09-11/scripts/nccl_sweep.sh`, 27B DFlash2 mit
    maurienne-NVFP4-Kopf, je Variante fünf Läufe à 400 Token, greedy; Rohdaten
    `handover/2026-09-11/ergebnisse/nccl_sweep_rtx*.out`):**

    | Variante | Median tok/s | Spanne | Boot |
    |---|---:|---|---|
    | Basis | 76,65 / 77,04 | 76,64–76,83 / 76,89–77,16 | Sweep 1 / Sweep 2 |
    | `NCCL_PROTO=Simple` | 75,86 | 75,73–75,88 | 1 |
    | `NCCL_ALGO=Ring` | 76,88 | 76,86–77,00 | 1 |
    | `NCCL_PROTO=LL` | 71,25 | 71,21–71,42 | 1 |
    | `NCCL_PROTO=LL128` | 71,33 | 71,25–71,45 | 1 |
    | `NCCL_ALGO=Tree` | — | bootet nicht: „no algorithm/protocol available for AllGather" | 1 |
    | `NCCL_BUFFSIZE=524288` | 77,77 | 77,60–77,85 | 2 |
    | `NCCL_BUFFSIZE=1048576` | 77,57 / 77,73 | 77,47–77,82 / 77,53–77,97 | 1 / 2 |
    | `NCCL_BUFFSIZE=2097152` | 76,65 | 76,61–76,84 | 2 |

    Annahmelänge überall 3,325, SHA überall `0106659946c064b1` — kein Schalter
    ändert den Text. **Befund:** Ring und Simple sind bereits die Vorgabe
    (Wechsel = Rauschen), LL/LL128 kosten 7 %, Tree gibt es für AllGather
    nicht. Einzig ein kleinerer NCCL-Puffer (512 KiB–1 MiB statt 4 MiB
    Vorgabe) bringt in zwei unabhängigen Boots reproduzierbar **+0,9 bis
    +1,2 %** bei nicht überlappenden Spannen; 2 MiB liegt auf der Basis. Die
    Boot-zu-Boot-Streuung der Basis beträgt 0,5 %. **Schluss: der AllReduce
    ist über die Umgebung nicht zu heben; die 31,8 % sind der Preis des
    Host-Stagings ohne P2P und nur strukturell zu senken (weniger AllReduces
    je Schritt).** Vorschlag für die Produktion (Entscheidung Peuqui):
    `NCCL_BUFFSIZE=1048576` in die vLLM-TP-Einträge von llama-swap; +1 % ist
    wenig, kostet aber nichts. Punkt 3 des Plans damit ERLEDIGT.

    **Sweep-Skript-Falle (12.09.):** `speed_dflash.sh` bricht ab, wenn die
    Karten nach dem Abbau der Vorgänger-Variante noch über 500 MiB belegt sind
    oder ein `api_server` nachläuft — vier Varianten des ersten Laufs blieben
    so ohne Ergebnis, die Meldung ging im `/dev/null` verloren. Das Skript
    wartet jetzt auf freie Karten und schreibt je Variante ein Log.

    **Paket G (Punkt 5 des Plans) VORBEREITET, nicht eröffnet (12.09.
    nachts):** Worktrees `1Cat-vLLM-pr-timeout` (Branch
    `nccl-subgroup-timeout`) und `1Cat-vLLM-pr-compilecache` (Branch
    `drop-forced-compile-cache-off`) auf `origin/main`, beide uncommitted,
    pre-commit + mypy-3.10 grün, Entwürfe in `upstream-contrib/03-1cat-issues/
    pr-nccl-subgroup-timeout.md` und `pr-drop-forced-compile-cache-off.md`.
    - Timeout: Beleg am Code, torch 2.10 `_new_group_with_tag` nimmt bei
      `timeout=None` die 600-s-Konstante, nie den Wert der Weltgruppe; neuer
      CPU-Test (3 passed). Der Wachhund-Fall vom 06.09. im Memory betraf die
      Weltgruppe; ein Log eines Untergruppen-Abbruchs liegt nicht vor.
    - Compile-Cache: die Erzwingung sitzt upstream ZWEIMAL (`config/vllm.py`
      und `envs.py` `disable_compile_cache()`); der Fork trägt die zweite
      noch, deshalb ist der Cache auf dem 0DOT3-Pfad auch in Produktion aus.
      Kalt/Warm-Beleg (`handover/2026-09-11/scripts/cache_coldwarm.sh`, RTX-
      Paar, DFlash2): kalt 447 s, warm 121 s mit 4× „Directly load AOT",
      erzwungen-aus 127 s; SHA `0106659946c064b1` und Annahme 3,325 in allen
      drei. **Kein Drift, aber auch kein Bootzeit-Gewinn** bei warmem
      Inductor-Cache; der Kalt-Export kostet einmal ~5 min. Der PR ist eine
      Bereinigung, Entscheidung Peuqui. Overlay-Folge unabhängig vom PR:
      envs.py-Default im Fork zurücknehmen (siehe OVERLAY-INVENTUR).

    **Punkt 15, Skinny-Build pro Architektur — GEMESSEN 12.09. nachts (Punkt 4
    des Plans, `handover/2026-09-11/scripts/p15_chain.sh`, Patch
    `patches/punkt15_skinny_per_arch.diff` auf work-main ANGEWENDET,
    uncommitted):** Extension heißt jetzt `skinny_nvfp4_v11_sm{cc}` und baut
    mit `-gencode` der eigenen Karte; cuobjdump zeigt sm_75 bzw. sm_70.

    | Paar | Median tok/s | Referenz | SHA | Annahme |
    |---|---:|---:|---|---:|
    | RTX 8000 (sm75-Build) | 77,25 (77,10–77,32) | 76,65 / 77,04 (sm70-Build, zwei Boots) | `0106659946c064b1` | 3,325 |
    | V100 (sm70-Build) | 76,28 (76,26–76,59) | 76,27 | `0106659946c064b1` | 3,325 |

    Bitgleich auf beiden Paaren. Der RTX-Zuwachs von +0,3 bis +0,8 % liegt
    innerhalb der Boot-Streuung (0,5 %); die 1–6 % aus dem Mikrobenchmark
    bei M ≤ 4 kommen im Decode nicht an, weil der Skinny-GEMM nur 24 % des
    Schritts ist und der Rest AllReduce/QPN8 bleibt. **Behalten oder
    zurücknehmen — Entscheidung Peuqui:** korrekt ist der Bau für die eigene
    Architektur allemal (Paket E braucht ihn für die Turing-Kopie), ein
    Tempo-Argument gibt es nicht. Rückweg: `git apply -R` des Diffs.

    **Punkt 14, Decode-Profil MIT NVFP4-Entwurfskopf — GEMESSEN 12.09. nachts
    (Punkt 6 des Plans, `prof_dflash.sh nvfp4head 0,2`, `DRAFT=maurienne`,
    sm75-Skinny-Build aus Punkt 15, Rohdaten
    `handover/2026-09-11/ergebnisse/prof_nvfp4head_rtx.out`):** Rang 0
    **5.194 ms** GPU-Zeit im Fenster (10.09. mit fp16-Kopf: 5.395 ms, −3,7 %).

    | Posten | ms | Anteil | gegenüber 10.09. |
    |---|---:|---:|---|
    | `ncclDevKernel_AllReduce_Sum_f16_RING_LL` | 1.786 | 34,4 % | 1.714 → gleich; NCCL wählt LL von selbst |
    | `skinny_nvfp4_qpn2` | 1.399 | 26,9 % | 1.303 → gleich (Boot-Streuung) |
    | `skinny_fp8_qpn8` | 912 | 17,6 % | 889 → gleich |
    | `fused_sigmoid_gating_delta_rule` | 182 | 3,5 % | 180 |
    | `ncclDevKernel_AllGather_RING_LL` | 182 | 3,5 % | 178 |
    | `Kernel2` (neu, 7.387 Aufrufe) | 138 | 2,7 % | vermutlich der NVFP4-Kopf selbst |
    | `turing_fp16_s1688gemm` | — | — | 273 → **weg** (fp16-Kopf) |
    | `cutlass_75_wmma…f16` | — | — | 291 → **weg aus den Top 14** |

    **Befund:** Nach Block-Pack und Kopfwechsel bleiben drei Posten mit
    79 % des Schritts: AllReduce (Umgebung ausgereizt, siehe Env-Sweep),
    qpn2 (87–98 % Dachlinie) und qpn8 (DRAM-gebunden). Der Kopfwechsel hat
    564 ms fp16-GEMM gegen 138 ms `Kernel2` getauscht. **Weitere Hebel sind
    nur strukturell: weniger AllReduces je Schritt.** Punkt 6 des Plans
    ERLEDIGT.

    **Punkt 10, TileLang-Pin 0.1.14 — Kernel-Tests GRÜN (12.09. nachts, Punkt 7
    des Plans, `handover/2026-09-11/scripts/p7_tilelang_chain.sh`, venv
    `.venv-sm70-tltest`, tilelang 0.1.14):**

    | Karte | `test_mhc_kernels.py` | `test_mhc_sm70_fp16.py` |
    |---|---|---|
    | V100 (Karte 1) | 43 passed, 8 skipped | 24 passed, 1 skipped |
    | RTX 8000 (Karte 0) | 43 passed, 8 skipped | 10 passed, 15 skipped |

    Belegt am Code: tilelang 0.1.14 `cuda/target.py` liest die Architektur
    über `torch.cuda.current_device()`, exakt unser Overlay-Patch
    `fork_patches_150/tilelang_target.py` (0.1.10 nahm Gerät 0). Der Patch
    wird mit dem Pin überflüssig. Modellboots (DeepSeek PP5, 27B-GDN) mit der
    venv: siehe Folgeeintrag. **Falle:** ohne `CUDA_HOME=/home/mp/vllm/cuda`
    (und dessen `bin` im PATH) übersetzt tilelang mit `/usr/bin/nvcc` (CUDA
    12.0), das den System-g++ 13 ablehnt — ein erster Lauf scheiterte so
    komplett; `speed_dflash.sh` setzt beides, jede pytest-Kette muss es auch.

    **Modellboots mit 0.1.14 — BESTANDEN (12.09. nachts,
    `handover/2026-09-11/scripts/p7_p10_chain.sh`):** DeepSeek-V4-Flash PP5
    über `ds_accept.sh tl014` (Boot 710 s): acht Prompts zweimal im selben
    Prozess **8/8 byteidentisch**, Code 26,5 tok/s (Referenz 26,7), Prosa
    23,6. Qwen3.8-27B (GDN) DFlash2 auf dem RTX-Paar: **77,14 tok/s, SHA
    `0106659946c064b1`, Annahme 3,325** — wie mit 0.1.10. Pin-Vorschlag an
    1Cat als Entwurf: `upstream-contrib/03-1cat-issues/comment-tilelang-pin-
    0114.md` (nicht gepostet, Freigabe Peuqui; vorher `gh issue list --search
    tilelang`). Danach: Overlay-Patch `tilelang_target.py` und die 11-GB-venv
    `.venv-sm70-tltest` löschen, sobald der Pin in work-main gezogen ist.
    Punkt 7 des Plans ERLEDIGT.

    **DeepSeek-Eintrag: Tool-Call und kurze Chat-Prompts — BEFUND (12.09.
    nachts, Punkt 10 des Plans, `handover/2026-09-11/scripts/abnahme2/
    toolcall_probe.py`, Ergebnisse `ergebnisse/toolcall_deepseek*/`):**
    - **Runde 1 (Tool-Call) BESTANDEN:** 323 Prompt-Token mit `get_weather`-
      Schema, `finish_reason=tool_calls`, Parser `deepseek_v4` liefert Name
      und Argumente `{"city":"Hamburg","unit":"celsius"}` korrekt.
    - **Runde 2 (Werkzeugergebnis als `role=tool` zurück) HÄNGT:** 420
      Token berechnet, dann Decode ohne Fortschritt, nach 300 s `RPC call to
      sample_tokens timed out`, EngineCore tot, HTTP 500. Die fünf Worker
      überleben den API-Server-Tod als Waisen mit **180 GB VRAM** (das
      `cmdStop`-Skript greift nur beim Entladen, nicht beim Selbsttod) —
      zweimal von Hand per PID beendet.
    - **Boot-Prompt „Guten Tag." (8 Token nach Template, max_tokens 8) STÜRZT
      AB:** `AssertionError: topk_indices is not None` in
      `deepseek_v4/amd/rocm.py:788` `_forward_prefill` (Worker PP0). Der
      C128A-Builder setzt `c128a_prefill_topk_indices` nur bei
      `num_prefill_tokens > 0` mit Schwelle `1 + num_spec = 6`; der SWA-Builder
      (`sparse_swa.py:294`) hat Klassenschwelle 1 — Verdacht auf eine
      Decode/Prefill-Einstufung, die zwischen beiden Buildern auseinanderläuft.
      Die Abnahme vom 11.09. hatte nur 13k-Vorkontext; die acht Kohärenz-
      Prompts (16–32 Token) laufen über das Serve-Skript **ohne**
      `--enable-prefix-caching`, das der llama-swap-Eintrag setzt. A/B dazu:
      `abnahme2/ds_short_prompt_ab.sh` — **ERGEBNIS: beide stürzen ab.** (A)
      Serve-Skript ohne Prefix-Cache, up nach 500 s, „Guten Tag." → dieselbe
      Assertion in Worker PP0, HTTP 500 nach 300 s; (B) llama-swap-Eintrag,
      identisch. **Prefix-Caching ist nicht die Ursache; der Auslöser ist
      der kurze Prompt (8 Token). 16 Token liefen in derselben Nacht.** Die
      Logs: `ergebnisse/ds_short_prompt_ab/`.
    - **GRENZE AUSGEMESSEN und URSACHE GEFUNDEN (12.09. ~04:00,
      `abnahme2/ds_short_prompt_boundary.sh`, zwei Boots über das Serve-
      Skript, Prompts mit exakt 4–16 Token nach Template):** 4, 5, 6 laufen;
      **7, 8, 11 stürzen ab** (7 und 11 gemessen, 8 dreimal zuvor); **12, 13,
      14, 15, 16 laufen.** Das Fenster ist exakt (6, 11].
      **Ursache:** drei Metadaten-Bauer, zwei Schwellen. `flashmla_sparse`
      (C128A) holt seine Decode-Schwelle über `_init_reorder_batch_threshold(1,
      supports_spec_as_decode=True)` = `1 + 2·k` unter parallelem Drafting
      (`backend.py:617`), DSpark setzt `parallel_drafting=True`
      (`config/speculative.py:951`) → **11**. `sparse_swa.py:304` rechnet
      `1 + k` → **6** (der Indexer ebenso, `indexer.py:447`). Eine Anfrage mit
      7–11 Query-Token ist für C128A Decode (`treat_short_extends_as_decodes`,
      also keine Prefill-Topk-Indizes), für SWA Prefill → `_forward_prefill`
      → `assert topk_indices is not None`. **Die Werkzeug-Rückrunde vom Abend
      war derselbe Fehler:** Prefix-Cache-Treffer ließ 8 neue Token übrig
      (`num_scheduled_tokens: 8`, Assertion im `crash_journal.txt`); der
      „Hänger" war nur der 300-s-RPC-Timeout nach dem Worker-Tod. Upstream
      identisch (`origin/main` sparse_swa.py:304).
      **Fix:** SWA-Builder ruft denselben Helfer (`_init_reorder_batch_threshold`),
      Worktree `1Cat-vLLM-pr-swathreshold` (Branch `sparse-swa-spec-threshold`,
      pre-commit + mypy grün), als Overlay auf work-main ANGEWENDET
      (`handover/2026-09-11/patches/sparse_swa_spec_threshold.diff`).
      **Verifikation `abnahme2/ds_fix_verify.sh` (12.09. ~04:45, ein Boot mit
      Fix, Serve-Skript): alle 13 Längen 4–16 OK, null Assertions im Log.**
      Tool-Call dort nicht prüfbar (Serve-Skript ohne `--enable-auto-tool-
      choice` → HTTP 400 ist Anfrage-Ablehnung, kein Engine-Fehler); Tool-
      Runden gegen den llama-swap-Eintrag mit Fix (`ergebnisse/
      toolcall_deepseek_fixed/`): **Runde 1 `get_weather` mit Argumenten
      BESTANDEN, Runde 2 (Werkzeugergebnis zurück, der Absturzfall vom
      Abend) BESTANDEN in 5,3 s**, Antwort nennt 17 °C und bewölkt.
      **Punkt 10 des Plans damit ERLEDIGT: Fehler gefunden, behoben, belegt.**
      Committet: work-main `911c259f`, **PR #603** (12.09. früh).

18. **Paket E gestartet (12.09. vormittags, Freigabe Peuqui):** Plan und
    Befunde in `upstream-contrib/03-1cat-issues/paket-e-plan.md`, Gate-Karte
    in `paket-e-gates.md`. Kern: 1Cat trägt unsere Kernel kompiliert
    (`csrc/sm70_turbomind/ops/`, MIT-Vermerk), aber nur Volta; Turing ist im
    TurboMind-C++ per `Arch<700, 750>` ausgeschlossen. Routenkampagne: unser
    Skinny-Pfad ist in Produktion nur beim 27B auf der RTX aktiv (qpn2/qpn
    Decode, Dequant+cuBLAS Prefill); Flash-Next (Marlin-MoE) und DeepSeek
    (Marlin) laden ihn nicht → E-3 entfällt ohne Beleg. Vorschlag E-1: unseren
    Turing-Pfad in 1Cats NVFP4-Op (Entscheidung Peuqui offen). Quellbau
    reines main → `.venv-pr-turing` (`handover/2026-09-12/
    build_pr_turing.sh`, Worktree `1Cat-vLLM-pr-turing-ops`).

    **E-1 UMGESETZT und BELEGT (12.09. nachmittags):** Turing-Pfad für
    modelopt NVFP4 (qpn2-Decode + Dequant-Prefill, neuer Op
    `utils/nvfp4_qpn2_dequant.py`) und FP8 (per-Tensor → QPN8 mit Kanal-
    skalen, ihr `fp8_qpn8_dispatch`), Gates pro Worker-Gerät.
    **Ergebnis 27B MTP k=3, Triton-Attention, fp16-KV: RTX 8000 mit E-1
    71,0 tok/s, V100 auf main 63,4 tok/s, SHA beider `38848c08a44405ae`
    identisch, Annahme 3,000.** Reines main lädt den 27B auf Turing gar
    nicht (Mixed-Config 89; darunter `orig_dtype`-Absturz im Marlin-FP8).
    Zwei Turing-Fallen außerhalb des Linear-Pfads, im PR nur vermerkt:
    FlashInfer (Default auf Turing) scheitert in `BatchPrefillWithPagedKVCache`
    → `--attention-backend TRITON_ATTN`; fp8-KV lässt Inductor `fp8e4nv`
    casten, das Triton auf sm75 ablehnt → `--kv-cache-dtype float16`.
    Lehre: ein Python-Zweig auf M im Forward wird von torch.compile beim
    Warmup-M eingefroren (17 tok/s, Dequant 64 % des Fensters) — der Split
    gehört in den Op (wie 1Cats C++-Dispatcher). Profil RTX: qpn2 31 %
    (ohne Block-Pack, → E-2), AllReduce 21 %, qpn8 18 %, Triton-Attn 6,5 %.
    Tests: 13 CPU, 11 GPU auf V100 und RTX, bestehende sm70-Testdateien
    grün (Mixed-Min-Cap-Test auf Pre-Ampere umgestellt). Entwurf
    `upstream-contrib/03-1cat-issues/pr-turing-nvfp4-fp8-linear.md`.
    Committet `78a78f5e` auf fork/sm70-ops-on-turing, **PR #604** (12.09.
    nachmittags, Freigabe Peuqui). Nächste Schritte in Peuquis Reihenfolge:
    E-2 Block-Pack, dann der Dreizeiler gegen FlashInfer auf Turing, dann
    Paket C (sm75-FA2), dann der fp8-KV-Cast.

    **E-2 Block-Pack — IN ARBEIT (12.09. nachmittags), Branch
    `sm70-qpn2-block-pack` auf E-1 im selben Worktree:** Pack-Kernel
    `[k/16][rows][16]`, `Packed`-Template auf GEMM und gated GEMM, Schwelle
    sm75 ≥ M5 / sm70 ≥ M8, Schalter `VLLM_SM70_NVFP4_QPN2_PACK` (registriert,
    aus den Compile-Faktoren genommen). **Kernel bitgleich 58/58 auf V100 und
    RTX.** Bau in `.venv-pr-turing` (`handover/2026-09-12/build_e2.log`).
    **Befund nebenbei, wichtig für alle Abnahmen:** auf dem 0DOT3-Pfad ist
    jeder frische Compile eine Münze (Combo-Kernel-Benchmark), der AOT-Pfad
    friert die Wahl pro Env-Hash ein — 400-Token-SHA ist dort kein
    Korrektheitskriterium, A/B nur bei gleichem Env-Hash. 12 Boots belegt,
    Details `paket-e-plan.md`, Memory `reference_aot_inductor_cache_env_hash_drift`.
    DFlash2 auf main+Turing gesperrt (FA_V100 exakt SM70) → Tempo-Beleg über
    MTP mit k=3..7. **ERGEBNIS (RTX, gleicher Env-Hash, je 5×400 Token):**
    M=4 68,8/68,9 (ungepackt), M=5 69,7/69,4, M=6 71,4/71,6, M=7 66,5/**67,8
    (+1,9 %)**, M=8 64,4/**66,7 (+3,7 %)**; SHA und Annahme je Paar gleich.
    **Turing-Schwelle auf 7 gesetzt** (statt 5 aus dem Mikrobenchmark),
    Volta bleibt 8. PR-Worktree `1Cat-vLLM-pr-blockpack` (Branch
    `sm70-qpn2-block-pack-pr` auf main): `nvfp4_qpn2_sm70.cu`, `envs.py`,
    neuer Test. Entwurf `upstream-contrib/03-1cat-issues/pr-qpn2-block-pack.md`.
    Committet `82ae4e0e` auf fork/sm70-qpn2-block-pack-pr, **PR #611**
    (12.09. nachmittags, Freigabe Peuqui). Nächster Schritt: **Paket C**
    (sm75-FA2 für 1Cat), Reihenfolge Peuqui.
    - Passt zu 1Cat-Issue #597 (delubee, DSML-Tool-Calls auf 8× V100). Ob und
      was dort gemeldet wird: Entscheidung Peuqui.
    - **Betriebsrisiko für AIfred:** ein toter Engine-Kern hinterlässt 180 GB
      belegtes VRAM; llama-swap meldet „running: []", der nächste Boot scheitert
      am Speicher. Vorschlag: `vllm-swap-stop` um eine Prüfung auf verwaiste
      `VLLM::Worker` derselben Prozessgruppe ergänzen, oder ein Watchdog.

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
      erkannte Coandă sofort). **Befund über das Abnahme-Skript, nicht über
      den Betriebspunkt:** die 16384 standen nur in `serve-qwen38-flash-next.sh`-
      Aufrufen; der Produktionseintrag fährt MML 262144.

      **Nachmessung gegen den llama-swap-Produktionseintrag (11.09. spät,
      `handover/2026-09-11/scripts/abnahme2/chat_ask_prod.py`, Vorkontext ×3
      = 38.995 Prompt-Token, `MAXTOK=16000`, Denken an, Ergebnisse in
      `handover/2026-09-11/ergebnisse/flashnext_prod_kuanda/`):**

      | Frage | Ausgabe-Token | Denkblock | Antwort | Ende |
      |---|---|---|---|---|
      | q1 Quantenphysik | 3.073 | 8,4k Zeichen | 30 Sätze, sauber | stop |
      | q2 Regenbogen | 1.597 | 4,7k | 30 Sätze, sauber | stop |
      | q3 Kuanda | 4.172 | 15,3k | 30 Sätze, Zurückweisung | stop |

      Decode 56–58 tok/s (q1 26,5 inkl. Kaltstart). q3: „nicht als Fachbegriff
      bekannt", Tippfehler vermutet, **Kunda-Effekt** (motiviertes
      Schlussfolgern, Ziva Kunda) angeboten, ausdrücklich keine erfundenen
      Namen/Zahlen/Experimente. Coandă kommt nicht vor. Nach der Kuanda-Regel
      kein Durchfall (Zurückweisen ohne Erfindung); Einordnung Peuqui offen.
      **Folge: Denken bleibt an, die Denkfrage aus `HANDOVER.md` ist
      aufgelöst, die PLE-Kaskade ist dafür nicht nötig.** Abnahme-Skripte
      (`flashnext_qual.sh`, `flashnext_ab.sh`, `flashnext_qual_chat.sh`) und
      Betriebspunkt-Angaben stehen seitdem auf `MML=262144`.

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

19. **MXFP4-Port für Volta und Turing — NÄCHSTER ARBEITSBLOCK** (Auftrag
    Peuqui 20.09.2026). Ziel: MXFP4-Kernel, jeweils für sm70 und sm75
    optimiert. Ausgangslage: 1Cat hat `mxfp4_qpn_m1_sm70.cu` und
    `mxfp4_sm70_moe.py`, aber „m1" deutet auf einen reinen M=1-Kernel —
    gebraucht werden die breiteren Bänder (6 Zeilen für den DSpark-Verifier,
    256 für den Prefill-Chunk). Der Unterschied zu NVFP4 liegt im
    Entpack-Pfad: eine Skala je 32 statt je 16 Werte, E8M0 statt FP8.
    **Warum es sich doppelt lohnt:** Der MXFP4-Checkpoint ist rund 8 GB
    kleiner (157,2 GiB gegen 165 GiB), weil MXFP4 laut NVIDIAs
    `cast_mxfp4_to_nvfp4.log` das native Format ist und NVFP4 nur teurere
    Skalen darauf packt (lossless=100 %). Auf den V100-Stufen sind das je
    etwa 1,5 GB mehr freier Speicher — und der entspannt **beide** Posten,
    die den Kontext begrenzen: KV-Pool und Indexer-Reserve (siehe Punkt 20
    und den gescheiterten 128k-Versuch vom 20.09.). Checkpoint-Kandidaten:
    `haanjack/…-MXFP4` (157,2 GB, einziger mit quantisiertem Head, MTP
    quantisiert, aber kein README und keine Qualitätszahlen) und
    `amd/…-MXFP4` (159,1 GB, saubere Modelkarte, GSM8K 99,9 %, ROCm-Ziel).
    Danach erst: Kontextfenster neu ausmessen.
    **BEFUND 20.09. — KEIN DOWNLOAD UND KEINE KONVERSION NÖTIG.** Unser
    NVFP4-Checkpoint IST MXFP4, nur in NVFP4-Verpackung. Nachgemessen an
    `layers.0.ffn.experts.0.w1`: 262.144 von 262.144 Skalenpaaren sind
    identisch, alle Skalen sind exakte Zweierpotenzen, und der globale Faktor
    `weight_scale_2` ist 2^-13 — genau das `m=-13`, das
    `cast_mxfp4_to_nvfp4.log` für diese Schicht nennt. Die gepackten 4-Bit-
    Gewichte sind bereits die MXFP4-Gewichte, Byte für Byte; beim Hinweg wurde
    lediglich jede Skala je 32 Werte auf zwei Skalen je 16 Werte dupliziert.
    Genau das sind die ~8 GB Unterschied zu den fremden MXFP4-Checkpoints.
    ⇒ Die Redundanz wird BEIM LADEN aufgelöst (Skalen halbiert als E8M0
    halten), nicht durch einen zweiten Checkpoint auf der Platte. Spart 157 GB
    Plattenplatz, den Download und die Qualitätsfrage bei `haanjack` (kein
    README, keine Messwerte).
    **VORGABE PEUQUI 20.09.:** generisch bauen, NICHT an TP-Stufen festmachen
    („manchmal fahren wir auch TP2") und nicht auf eine Architektur
    festklopfen. 1Cats vorhandener Code macht genau das Gegenteil:
    `_mxfp4_qpn_m1_decode_contract` (mxfp4_sm70_moe.py) lässt den schnellen
    Pfad nur zu bei `tp_size == 4`, 256 Experten und exakt passenden
    Tensorformen — Kommentar dort: „Admit only the measured TP4 six-route
    W13/W2 tensor pair." Für sm75 gibt es bei 1Cat überhaupt nichts; ihre
    beiden Entwurfsdokumente zielen auf TP8 mit acht V100 und TurboMind
    (Marlin ausdrücklich „out of scope").
    **KERNEL-SEITE FERTIG UND BELEGT (20.09., UNCOMMITTED).** Beide Formate
    laufen durch DENSELBEN Kernel, der Modus ist ein Template-Argument:
    `e8m0_to_half2` (MXFP4-Skala ist ein reiner Exponent — fp16-Bitmuster per
    Shift, kein Multiplizieren), `group_scale<MODE>` kapselt den
    Indexunterschied (eine Skala je 16 gegen je 32), `skinny_nvfp4_moe_qpn`
    nimmt `SCALE_MODE`, der Einstiegspunkt einen optionalen Parameter
    `scale_mode` (Vorgabe 0 = NVFP4, bestehende Aufrufer unverändert).
    `_qpn_prepack(codes, scales, scale_group=16)` in marlin.py trennt jetzt
    Code-Gruppen (immer 16) von Skalen-Gruppen (16 oder 32).
    Rund 25 Zeilen im Kernel, kein zweiter Kernel, keine Verzweigung im heißen
    Pfad — und **architekturunabhängig**, weil die Änderung vor den
    Tensorkern-Instruktionen sitzt.
    BELEGE (scratchpad moe_regression.py, mxfp4_equivalence.py,
    mxfp4_real_weights.py, je splitk 8/16/32 x nacc 1/2):
     - NVFP4-Pfad gegen die Fassung VOR dem Umbau: bitgleich, maxdiff 0.
     - MXFP4 gegen NVFP4 auf gleichwertigen Daten: bitgleich, Skalentabelle
       8192 -> 4096 Bytes.
     - MIT ECHTEN GEWICHTEN (layers.0.ffn.experts.0-3.w1, 4 x [2048, 4096]):
       bitgleich, Skalenspeicher 2,00 -> 1,00 MiB, und das Code-Layout ist
       für beide Formate IDENTISCH (die 146 GiB Gewichte werden beim
       Umschalten nicht angefasst).
    **LÄUFT IM SYSTEM (20.09. 16:14, Fork b81c503e).** Die Faltung passiert
    beim Laden: Erkennung per Byte-Arithmetik, Raster auf eine E8M0-Skala je
    32 Codes gefaltet, die vollen Parameter freigegeben, Kernel liest die
    kurze Tabelle. Schalter `VLLM_SKINNY_MXFP4_SCALES=0` behält das
    ausgelieferte Raster.
    GEMESSEN (2x RTX 8000 + 3x V100, PP5):
     - 8,7 GiB frei geworden (vorhergesagt 8,62), Modell 164,0 → 155,3 GiB,
     - Schrittzeit 79 ms gegen 81 ms vorher (2. Lauf; der erste nach dem Boot
       zeigt 88 ms Aufwärmeffekt — nicht als Regression lesen),
     - **Fenster 65.536 → 131.072, Pool 71.493 → ~219.000 Token (1,67x)**,
     - Nadeln 30k/62k/**125.511** je 4 von 4 — der 120k-Prompt hatte mittags
       noch die Engine getötet.
    ZWEI FALLEN AUF DEM WEG (beide gemessen, nicht geraten):
     1. Die Erkennung darf das Raster NICHT nach float wandeln: 256 Experten je
        Schicht ⇒ 2 GiB Transienten ⇒ OOM mitten im Laden. Byte-Arithmetik:
        Zweierpotenz = Mantissenbits null (`b & 0x87`), E8M0 = `(b >> 3) + 120`.
     2. Die eigene Referenz umzubiegen reicht nicht — der Layer hält die
        Parameter weiter, die gefaltete Kopie liegt dann NEBEN dem Original
        (gemessen: 1,9 GiB je Stufe VERLOREN statt gewonnen). Freigabe über den
        Storage-Zeiger, nicht über Attributnamen.
    INDEXER-RESERVE: `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` im DSv4-Eintrag von 64
    auf 256 angehoben. Die 64 waren der Grund für den OOM am Mittag; mit dem
    gewonnenen Speicher ist die ehrliche Reserve jetzt bezahlbar.
    OFFEN: für FREMDE MXFP4-Checkpoints die Bestimmung eines globalen Faktors —
    reine E8M0-Exponenten überschreiten fp16, unser Checkpoint löst das mit
    `weight_scale_2` = 2^-13 daneben. Und der Turing-Teil (siehe unten).
    **TURING-HEBEL, nachgemessen (20.09.):** Gebaut wird ausschließlich
    `-gencode=arch=compute_70,code=sm_70`. `cuobjdump` am fertigen Modul zeigt
    genau EINE Cubin (sm_70) und **kein PTX**. Auf der RTX 8000 läuft das nur
    über CUDAs Binärkompatibilität aufwärts innerhalb derselben Hauptversion
    (7.0 → 7.5) — die Karte bekommt also keine einzige Turing-Instruktion zu
    sehen. Der Code selbst ruft `mma.m8n8k4` als Inline-PTX, also die
    Volta-Form; Turing könnte `mma.m16n8k8`. Ein zusätzliches
    `-gencode=arch=compute_75,code=sm_75` bringt daher zunächst nur besseres
    Scheduling und Registerverteilung, nicht die breitere MMA-Form. Für „auf
    Turing optimiert" braucht es beides: den sm75-Build UND einen Kernelpfad
    mit der Turing-MMA. Unabhängig von MXFP4 und vermutlich der größere Posten.

20. **KV-Auslagerung in den Hauptspeicher, generisch — DANACH** (Auftrag
    Peuqui 20.09.2026, ausdrücklich „nach Möglichkeit generisch, sodass da
    viele Modelle von profitieren"). **Es ist ein Prefill-Vermeider, kein
    Decode-Beschleuniger** (Peuqui 20.09.: „Prefill ist genau das, was den
    User warten lässt, beim Decode kann er sowieso nicht so schnell
    mitlesen"). Überlebt der KV im Hauptspeicher, wird ein langer Kontext
    EINMAL gerechnet und nie wieder — das ist der Gewinn, nicht die
    Millisekunden je Token. Messlatte aus dem Offload-Betrieb: 22k-Prompt
    nach einer 34k-Zwischenanfrage 1,5 s statt 14,3 s. Beweggrund: Bei quantisierten Modellen
    ist der KV-Cache ab etwa 35.000 Token der GRÖSSERE Posten gegenüber den
    Gewichten (Llama-8B Q4: 4,5 GB Gewichte gegen 128 KiB je Token), er ist
    also meist der eigentliche Grund, warum überhaupt ausgelagert werden
    muss. Wer stattdessen Schichten auf die CPU legt, lagert ausgerechnet
    das aus, was jedes Token vollständig braucht.
    **Zwei Wege, beide prüfen:**
    (a) Nachladen nach der Top-k-Auswahl des Indexers — so die vLLM-RFCs
        #33980 und #48203 (beide OHNE Code, Konzeptphase). Bandbreite bei uns
        (PCIe Gen3 x4, ~3,5 GB/s): 1M Kontext 132 ms/Token nur für
        Indexer + Top-2048 gegen 721 ms bei naivem Nachladen; bei 128k nur
        17 ms, also hinter den 81 ms Rechenzeit versteckbar.
    (b) Aufmerksamkeit aufteilen statt Daten bewegen (HGCA, arxiv 2507.03153)
        — GPU rechnet über die Blöcke im VRAM, CPU über die im Hauptspeicher,
        zusammengeführt über die Flash-Attention-Statistiken. **Für Modelle
        MIT Indexer verworfen** (Peuquis Einwand 20.09., nachgerechnet): Die
        fünf PCIe-Verbindungen addieren sich auf 17,5 GB/s, die CPU hat ihre
        ~50 GB/s nur einmal und teilt sie mit Scheduler und fünf Workern. Vor
        allem aber verschwindet der große Posten, wenn man ihn gar nicht erst
        auslagert: Der Indexer-Cache ist bei 1M Kontext nur 453 MiB je Stufe
        und bleibt im VRAM (so auch der vLLM-RFC). Dann bleiben je Schritt
        24 MiB Auswahl = 1,2 ms Transport — gegen eine Synchronisation über
        43 Schichten und eine CPU, die Attention ein bis zwei Größenordnungen
        langsamer rechnet. **HGCA bleibt richtig für DICHTE Modelle**, wo der
        ganze KV je Token gelesen werden muss und 50 GB/s gegen 17,5 GB/s
        gewinnen. Dafür ist das Papier geschrieben.
    **Generisch heißt:** Die Speicherverwaltung ist modellunabhängig; was
    nicht generisch ist, ist die Frage, WELCHE Blöcke gebraucht werden. Bei
    DSv4 und GLM-5.2 beantwortet sie das Modell selbst, bei dichten Modellen
    bräuchte es einen Schätzer (ShadowKV, InfiniGen, FreeKV — Forschung,
    nicht Portierung). Schnittstelle also: „Modell liefert Indizes, Backend
    besorgt die Blöcke."
    **Vorarbeit bei uns:** `common/ops/sparse_decode_bmm.py` macht den
    indizierten Entpack-Gather der ausgewählten Blöcke bereits — dort
    müssten fehlende Blöcke vorher aus dem Host geholt bzw. dorthin
    ausgelagert berechnet werden.
