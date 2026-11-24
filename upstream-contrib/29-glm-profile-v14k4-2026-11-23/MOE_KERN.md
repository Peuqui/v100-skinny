# GLM-5.3-Flash Decode: der skinny NVFP4-MoE-Kern bei 21–32 aktiven Experten (23.11.2026)

Nur Analyse: keine Änderung an Baum, GPU, Bau oder Git. Gelesen wurden `1Cat-vLLM-v14` (Produktion), die nsys-Datei `~/.cache/nsys-runs/glm-v14k4-decode16k-1123/*.1.sqlite` (Capture 1, 38 Decode-Runden), die SASS der ausgelieferten `vllm/_C.abi3.so` (`cuobjdump`, schreibgeschützt) und die Zählerläufe `~/.cache/bench-scripts/expert_count_2026-11-23/rank*.json`.
Kennzeichnung: **[gemessen]** direkt aus nsys/cubin, **[Quelle]** aus dem Quelltext (Datei:Zeile), **[abgeleitet]** Rechnung aus Gemessenem, **[vermutet]** Schluss ohne direkten Beleg. Einheiten stehen in jeder Tabelle; Richtung: Zeit weniger = besser, GB/s mehr = besser.

Datei- und Zeilenangaben (Baum `/home/mp/Projekte/vllm-research/1Cat-vLLM-v14`):
- Kern: `csrc/sm70_turbomind/ops/skinny_moe_qpn_sm70.cu` (im Folgenden `kern.cu`)
- Dispatch Python: `vllm/model_executor/layers/fused_moe/experts/skinny_sm70_moe.py` (`dispatch.py`)
- Routing: `vllm/model_executor/layers/fused_moe/sm70/routing.py`
- Fünf-Bit-Skalen: `vllm/model_executor/layers/quantization/utils/s5_scales.py`

---

## 1. Kurzfassung

1. **Keine Unterbelegung der SMs.** Das Gitter ist 64 × 40 CTAs; bei 21–36 aktiven Experten laufen 1376–2304 CTAs mit Arbeit, die V100 fasst 240 gleichzeitig, die RTX 144 (S5-Kern). Das sind 5,7–16 Wellen [abgeleitet]. „Mehr CTAs je Experte“ (Split-N) ändert daran nichts.
2. **Bandbreite, neu gerechnet mit den gemessenen Expertenzahlen je Schicht und den echten Bytes (13,574 MB je Experte, nicht 13,44):** RTX S0 548 GB/s (82 %), RTX S4 495 GB/s (74 %), V100 S1–S3 600/605/600 GB/s (67 %). Das weicht von den 523–537 GB/s (V100) und 436–485 GB/s (RTX) der Auftragsvorgabe ab (Abschnitt 4: ratio-of-sums mit Zählerlauf-Mittelwerten je Schicht; eine Regression über alle V100-Schichten gibt 510–530 GB/s Grenzrate, also 57–59 %). Die V100 liegt je nach Rechenweg bei **57–67 %**, die RTX bei **74–82 %**.
3. **Die V100-Stufen sind die Lücke** (11,06 ms je Runde bei 57–67 %); die RTX-S0 ist fast an der 85-%-Decke, die RTX-S4 hat ein eigenes Problem (neun csf-Launches je Runde mit Expander-Kern).
4. **Hauptverdacht V100 [abgeleitet aus SASS + Messung, nicht per ncu bestätigt]:** Der Dekodier-Pfad `RB == 1` des m8n8k4-Kerns hat keinen expliziten Vorlade-Ring wie der m16n8k8-Kern; der Compiler holt die Codes nur etwa eine Gruppe voraus und leert die Pipeline an jeder Schleifenkante (unroll 4). Pro Warp-Iteration vergehen 2400 Takte bei 165 Befehlen: der Scheduler ist zu 59 % leer, die Warps warten (Speicherlatenz), sie konkurrieren nicht um Issue-Slots.
5. **Auffällig:** Beide Karten liefern **gleich viel je SM**: V100 600 GB/s / 80 SM = 7,5 GB/s je SM; RTX 548 GB/s / 72 SM = 7,6 GB/s je SM [abgeleitet]. Das spricht für eine Grenze je SM (ausstehende Ladeanforderungen / Latenzkette), nicht für DRAM oder Rechenleistung. Ein Test „nur Laden“ (E1, Abschnitt 6) entscheidet das, bevor irgendetwas umgebaut wird.
6. **Größter belegter Posten ohne Unsicherheit:** RTX S4, neun Projektionen je Runde behalten ihre csf-Planes (Fünf-Bit reicht nicht) und laufen über `csf32_expand_kernel` + Raster-Kern: **0,55 ms je Runde gemessen** nur für den Expander.
7. **Preis bei 85 % für alle Stufen: 3,4 ms je Runde (5,6 % der 60,7 ms), mit der niedrigeren V100-Rechnung bis 4,6 ms** [abgeleitet]. Realistisch erreichbar nach Rangliste (Abschnitt 5): etwa 1,2–2,4 ms (V100-Vorlade-Umbau) + 0,55–1,05 ms (S4-Skalen) + kleinere Posten.

---

## 2. Aufbau: Dispatch, Gitter, CTA, Skalen

### 2.1 Aufruf je Schicht [Quelle + gemessen]

`_apply_grouped` (`dispatch.py:805–880`): `group_routing(topk_ids)` (Triton, ein Block: `_group_routing_kernel`, `routing.py:82–120`), dann je nach Projektion (nur wenn kein S5-Plane vorliegt) `expand_csf32_experts`, dann `skinny_moe_qpn_sm70` für W13 (`up`), `self.activation` (Triton `triton_poi_fused_clamp_copy_`), `skinny_moe_qpn_sm70` für W2 (`down`), `_mtp_weighted_reduce`. Das sind je Schicht **sechs Kerne** (sieben mit csf-Expander): `grouped_topk_fused_small_expert_count_kernel`, `_group_routing_kernel`, [csf32_expand], `up`, `act`, `down`, `_mtp_weighted_reduce_kernel` [gemessen, Kernliste der nsys-Datei]. Up, Act und Down sind getrennte Launches; zwischen Up-Ende und Down-Start liegen 4,9–7,2 µs im Mittel, darin der Act-Kern mit 3,1–4,5 µs [gemessen].

Aufruf des Kerns: `_grouped_projection` (`dispatch.py:883–935`), Splitk aus `grouped_splitk` (`dispatch.py:81–97`): W13 (K 4096) und W2 (K 2048) bekommen beide **8 Warps** (m16n8k8: Präferenz (8,4) für K ≥ 2048; m8n8k4: (8,16,10)). Je Warp 32 (W13) bzw. 16 (W2) Gruppen zu 16 K.

### 2.2 Gitter und Arbeitsverteilung [Quelle]

`kern.cu:790–793`: `kColumnTiles = 2`, `grid = (N / 64, min(S, E))`. Hier N = 4096 für beide Projektionen (W13: 2·2048, W2: Hidden 4096), S = 5·8 = 40, E = 288 → **Gitter (64, 40, 1), Block 256 Threads**, so in der nsys-Datei für alle MoE-Kerne [gemessen]. Eine CTA = **ein Experte × zwei benachbarte 32-Spalten-Kacheln (64 Ausgabespalten) × volles K**, das K liegt in 8 Warps zerlegt (jeder Warp eine zusammenhängende K-Scheibe), die Warps summieren am Ende über Shared Memory in fester Reihenfolge `v += cs[w]` (`kern.cu:466` bzw. `:671–672`).

Das Routing kommt kompakt (`kern.cu:206–215`): Gruppenzeile `grp = blockIdx.y` hat `goff[grp+1]-goff[grp]` Slots; Zeilen hinter der letzten aktiven Gruppe haben `cnt = 0` und laufen nach dem Lesen von `goff` sofort ab (`kern.cu:256` qpn, `:522` m16n8k8). Das ist die Regel „Gitter an der Expertenzahl begrenzen“ (Commit 79e07bfef7, 21.10.2026: Gitter-y von Slotzahl auf `min(S, E)`; bit-gleich; im Prefill 3072 auf 256 Zeilen, **Decode unverändert**, weil S < E).

Folge im Decode [abgeleitet]: es werden immer 40 Gruppenzeilen gestartet, aktiv sind 21–36 (Mittel V100 etwa 24); also **10–47 % der CTAs sind leer** (bei n = 24,4: 998 von 2560 = 39 %). Sie werden nach den aktiven eingeplant (x läuft am schnellsten, y am langsamsten), kosten Startzeit plus zwei Leseverzögerungen von `goff`, geschätzt 4–6 µs je Kern auf der V100 [vermutet], also 1–3 %.

### 2.3 Was eine CTA je Gruppe tut [Quelle + SASS]

m16n8k8 (RTX, `kern.cu:502–705`): Gewichte sind der A-Operand (16 Zeilen = Ausgabespalten), die Zeilen des Experten der B-Operand (8 Spalten). Eine Warp-Iteration lädt je Kachel `uint2` Codes je Lane (256 B je Warp, zusammenhängend, `__ldcs`), ein `uint32` Skalenwort und, im S5-Pfad, ein Nibble- und ein Hi-Wort; die Aktivierung liest jede Lane als 8 B aus ihrer Zeile (`kern.cu:586–606`), Zeilen jenseits `rows` laden nichts und gehen als Nullen in die MMA. Die nächste Gruppe wird **explizit vorgeladen** (`load_group(g + 1)` zu Beginn der Iteration, `kern.cu:606, 621`); der Kommentar `kern.cu:483`/`:826` hält fest, dass tiefere Ringe auf der RTX langsamer waren.

m8n8k4 (V100, `kern.cu:221–445`, Dekodierpfad `RB == 1` ab `:308`): Der Code ist **nicht** vorgeladen; die Schleife `#pragma unroll 4` (`:312`) verlässt sich auf den Compiler. Der Kommentar `:315–318` hält fest, dass das Vertauschen der Reihenfolge (Gewichte vor Aktivierung) „18 % langsamer bei Decode auf V100“ war: der Pfad ist laut Autor lastempfindlich auf die Lade-Anordnung.

**Skalen [Quelle + gemessen]:** Drei Wege.
- **S5 (Fünf-Bit-Planes, in der Schleife dekodiert):** alle Schichten auf S0–S3 und die meisten auf S4. Je Kachel und Gruppe: `scale = base + nibble + 16·high` (`s5_scales.py:1–30`, `kern.cu:490–498`). Bytes je Kachel: Basis 32 B + Nibble (G/2)·32 B + Hi G·4 B.
- **Raster (vorab expandiert):** nur wo `scale_spread > S5_MAX_SPREAD = 31` (`s5_scales.py:33`, `dispatch.py:587`): die letzten Schichten von GLM-5.3 (Schicht 43 und 44, einzelne Zeilen in 36 und 40–42). Der Kern bekommt dann das Raster, vorher läuft `expand_csf32_experts` (`dispatch.py:821–850`). In nsys: 9 Launches von `csf32_expand_kernel` je Runde, **nur auf Gerät 4**, 61,1 µs im Mittel (22–137), **0,55 ms je Runde**; dazu Kernvariante mit 80 statt 91 Registern [gemessen].
- **csf direkt:** gibt es nicht (die „Ausnahmen“ lassen sich in der Schleife nicht nachschlagen, `s5_scales.py:8–13`).

### 2.4 Register, Shared Memory, Belegung [gemessen: nsys `registersPerThread`/`staticSharedMemory`; abgeleitet: Belegung]

| Karte | Kern | Reg. je Thread | Shared | CTA-Threads | CTAs je SM (Register) | gleichzeitige CTAs | CTAs mit Arbeit bei n = 22 / 36 | Wellen bei n = 22 / 36 |
|---|---|---|---|---|---|---|---|---|
| | | | B | | | (SM·CTAs) | | |
| V100 (80 SM) | `qpn<8,0,1,2,S5>` | 77 (belegt 80) | 16 480 | 256 | 3 (3·20 480 = 61 440 von 65 536) | 240 | 1408 / 2304 | 5,9 / 9,6 |
| RTX (72 SM) | `m16n8k8<8,0,1,2,S5>` | 91 (belegt 96) | 18 528 | 256 | **2** (2·24 576; ein drittes bräuchte ≤ 85 → 80 Reg.) | 144 | 1408 / 2304 | 9,8 / 16 |
| RTX (72 SM) | `m16n8k8<…,Raster>` | 80 | 18 528 | 256 | 3 (3·20 480) | 216 | | |

Die S5-Variante auf der RTX lässt 16 384 von 65 536 Registern je SM ungenutzt [abgeleitet]; die Raster-Variante (80 Reg.) zeigt, dass 3 CTAs je SM erreichbar sind. Rundung der Registerbelegung auf Vielfache von 8 je Thread (Zuteilungseinheit 256 je Warp) angenommen [abgeleitet; Gerätedokumentation nicht in diesem Lauf geprüft].

**Bytes je Experte [Header/abgeleitet, S5-Pfad]:** W13: 8 388 608 B Codes + 128 Kacheln · 5152 B Planes = 9 048 064 B; W2: 4 194 304 B + 128 · 2592 B = 4 526 080 B; zusammen **13,574 MB** (Codes 12,583 MB = 92,7 %; Planes 0,991 MB = 7,3 %). Im Raster-Pfad 14,147 MB (+1,573 MB). Die 13,436 MB der ANALYSE.md sind die csf-Rohgröße (ohne Planes-Umbau); ändert die Bandbreiten um etwa 1 %.

---

## 3. Zeitaufteilung je Schicht und Stufe [gemessen, 38 Decode-Runden, nsys-Capture 1]

Fenster = Beginn `grouped_topk…` bis Ende `_mtp_weighted_reduce`. Aux = Fenster − Up − Down (Topk, Routing, Act, Reduce, Lücken, auf S4 auch csf-Expander).

| Stufe | Karte | MoE-Schichten je Runde | Up | Down | Up : Down | Fenster | Aux | Up+Down je Runde |
|---|---|---|---|---|---|---|---|---|
| | | | µs | µs | – | µs | µs | ms |
| S0 | RTX | 11 | 526 | 266 | 1,98 | 810 | 18 | 8,71 |
| S1 | V100 | 7 | 367 | 184 | 1,99 | 578 | 27 | 3,86 |
| S2 | V100 | 7 | 353 | 176 | 2,00 | 558 | 29 | 3,70 |
| S3 | V100 | 7 | 333 | 166 | 2,01 | 527 | 28 | 3,50 |
| S4 | RTX | 10 | 393 | 196 | 2,00 | 661 | 72 | 5,89 |
| Summe | | 42 | | | | | | **25,66** |

Up : Down = 2,0 entspricht dem Byteverhältnis 9,048 : 4,526 (= 2,00); es gibt **keinen festen Zeitanteil je Launch** (die Anpassung Up = a + b·Down in der ANALYSE.md ergibt a im Rauschen). Die Zeit ist also fast ganz bytegebunden bzw. proportional zu den aktiven CTAs, nicht launch- oder rampengebunden.

Aux-Kerne je Schicht [gemessen, µs]: `grouped_topk…` 5,8 (RTX) / 10,0 (V100); `_group_routing` 2,9 / 4,6–5,1; Act (`triton_poi_fused_clamp_copy`) 3,1–4,5; `_mtp_weighted_reduce` 2,2–3,4; Lücke Up→Down 4,9–7,2 (enthält Act). Zusammen 18–29 µs je Schicht auf S0–S3, das sind 2,3–5 % des Fensters.

Alle diese Kerne laufen auf **einem** Stream (Stream 20, kein Zeitüberlapp mit anderen Streams während der skinny-Kerne: 0,0 %, geprüft an Gerät 1) [gemessen].

CTA-Größe: Eine Up-CTA liest 141,4 kB (2 Kacheln · (65 536 + 5152 B)), lebt auf der V100 etwa 56 µs (367 µs / 6,5 Wellen), eine Down-CTA 70,7 kB und etwa 28 µs [abgeleitet].

---

## 4. Bandbreite mit den gemessenen verschiedenen Experten [abgeleitet]

Rechnung je Schicht L: GB/s = n(L) · 13,574 MB / (Up(L) + Down(L)), n(L) = Mittel der verschiedenen Experten aus `rank*.json` (5-Zeilen-Schritte, 4302 Schritte), Up/Down(L) = Mittel der nsys-Runden. Wichtig: **Zählerlauf und nsys-Lauf sind verschiedene Läufe/Texte**; je Schicht rauscht das, über eine Stufe mittelt es sich. Auswertung in `eval_expert_count.py` (liefert die Schichtmittel), Rechnung hier nachvollziehbar aus den Tabellen unten.

| Stufe | Karte | Spitze | Mittel n | Up+Down je Schicht | GB/s | % Spitze | Spannweite je Schicht % |
|---|---|---|---|---|---|---|---|
| | | GB/s | | µs | GB/s | % | % |
| S0 | RTX | 672 | 32,0 | 791 | 548 | 82 | 75–86 |
| S1 | V100 | 900 | 24,4 | 552 | 600 | 67 | 65–70 |
| S2 | V100 | 900 | 23,6 | 529 | 605 | 67 | 62–73 |
| S3 | V100 | 900 | 22,0 | 500 | 600 | 67 | 64–71 |
| S4 | RTX | 672 | 21,5 | 589 | 495 | 74 | 60–85 |

Einzelwerte S0: Schicht 3 (n = 36,1): 874 µs, 561 GB/s (83 %); Schicht 13 (n = 26,7): 685 µs, 529 GB/s (79 %). S4: S5-Schichten 35/36/37/39: 569/558/544/543 GB/s (81–85 %); die Schichten mit csf-Projektionen 40–44 liegen bei 404–492 GB/s (60–73 %).

**Abweichung zur Vorgabe:** die Auftragszahlen (V100 523–537, RTX 436–485 GB/s) reproduziere ich als Verhältnis der Summen nicht. Die Regression Zeit gegen n über die 21 V100-Schichten gibt tatsächlich Grenzraten 532 (Up) / 510 (Down) GB/s und einen Achsenabschnitt von etwa −40 µs (kein fester Anteil); sie stützt die „523–537“. Für die RTX-S0 gibt dieselbe Regression Grenzraten über der Spitze (755–775 GB/s) und ist unbrauchbar: die Schichttiefe korreliert mit n und mit den Zeiten, die Läufe unterscheiden sich. **Belastbar ist die Spanne: V100 57–67 %, RTX S0 ≤ 82 %, RTX S4 60–85 % je nach Schicht; für Stufenentscheidungen die niedrigen Zahlen nehmen.** Eine exakte Zahl braucht Zähler und nsys im selben Lauf (siehe Abschnitt 6, Punkt 0).

Preis bis 85 % (V100 765, RTX 571 GB/s), Zeit neu = Zeit · (GB/s jetzt / Ziel): S0 0,31, S1 0,83, S2 0,77, S3 0,75, S4 0,78 → **3,4 ms je Runde**; mit 520 GB/s statt 600 auf den V100-Stufen 4,6 ms. [abgeleitet]

---

## 5. Wohin geht die Zeit bei etwa 25 Experten × 1,6 Zeilen? Urteil je Hypothese

| Hypothese | Beleg | Urteil |
|---|---|---|
| SM-Unterbelegung (CTAs gegen 80/72 SM) | 1376–2304 aktive CTAs, 240/144 gleichzeitig: 5,7–16 Wellen (Abschnitt 2.4); Zeit ∝ n ohne festen Anteil (Abschnitt 3) | **widerlegt** |
| Schwänze (letzte Welle) | CTA-Lebensdauer V100 Up 56 µs, Down 28 µs; erwarteter Leerstand etwa eine halbe CTA-Zeit: 28 µs von 367 (7,6 %) auf V100, RTX S4 etwa 5 % (9,8 Wellen), RTX S0 etwa 3,6 % [abgeleitet; hängt davon ab, dass eine einzelne CTA nicht schneller läuft, wenn die anderen fertig sind] | **ja, 4–8 %** |
| Leere CTAs (Gitter 40, aktiv 24) | 10–47 % leere CTAs (Abschnitt 2.2) | **ja, 1–3 %** [vermutet] |
| Skalenbehandlung S0–S3 | S5-Planes in der Schleife; 7,3 % der Bytes; 3 Skalenladungen je Kachel/Gruppe (Codes, Nibble, Hi: 6 `LDG` je Gruppe laut SASS) | Bytes ok, Ladezahl **klein, nicht Hauptursache** |
| Skalenbehandlung S4 | 9 csf-Projektionen je Runde: Expander 0,55 ms [gemessen], Raster-Kern +5 % Bytes, bis +40 % Zeit an Schichten 43/44, aber nicht von n trennbar | **ja, 0,55 ms sicher** |
| Launch-Zahl (Up, Act, Down getrennt) | Aux 18–29 µs je Schicht (Abschnitt 3), davon Lücken etwa 3 µs je Schicht | **klein, 0,3–0,6 ms in Summe** |
| Ladebreite (8 B je Lane) | `LDG.E.EF.64` je Gruppe und Kachel: 256 B zusammenhängend je Anfrage; Warp-Strom zusammenhängend 8 kB (Up) | Zugriffsmuster **sauber, vollständig koaleszent**; breiter nur über die Speicherordnung zu gewinnen [vermutet, 0–?] |
| Speicherlatenz nicht verdeckt (Vorlade-Tiefe) | V100: Compiler holt Codes etwa eine Gruppe voraus und leert an der Schleifenkante (SASS); 2400 Takte je Warp-Iteration bei 165 Befehlen (IPC je Warp 0,07), Scheduler 59 % leer; gleiche 7,5 GB/s je SM auf beiden Karten | **Hauptverdacht V100** [abgeleitet, ncu fehlt] |
| Rechenaufwand (Entpacken, Skalen, MMA mit leeren Zeilen) | 165 (V100) / 197 (RTX) SASS-Befehle je Gruppe und Warp, davon HMMA 32 / 8; Issue-Auslastung 41 % / 39 % (Rechnung unten); rein rechnerisch Integer-Pipe 31 % ausgelastet | **nicht begrenzend allein**, aber 25–30 % der Befehle sparbar |

### 5.1 Befehlsmischung der Hauptschleife [gemessen aus SASS]

Auszählung der Schleife von `skinny_nvfp4_moe_qpn<8,0,1,2,true>` (sm_70, Schleife 659 Befehle für 4 Gruppen) und `skinny_nvfp4_moe_m16n8k8<8,0,1,2,true>` (sm_75, Schleife 197 Befehle für 1 Gruppe), je Gruppe und Warp (beide Kacheln):

| Befehl | V100 qpn | RTX m16n8k8 |
|---|---|---|
| gesamt | 164,8 | 197 |
| LOP3 | 41,2 | 61 |
| IMAD | 38,0 | 57 |
| HMMA | 32,0 | 8 |
| HMUL2 | 20,0 | 32 |
| SHF | 10,0 | 16 |
| IADD3 | 7,5 | 6 |
| LDG | 6 | 6 |
| LD (A-Zeilen, generisch) | 2 | 1 |

Je Gruppe und Warp: 544–552 B Gewichte+Skalen. Das sind 3,0 (V100) bzw. 3,6 (RTX) Befehle je Byte, rund 5–6 Befehle je Gewichtscode und Lane [abgeleitet]. Die HMMA stehen nicht im Weg: 4 PTX-`mma.m8n8k4` je Kachel und Gruppe, etwa 64 Takte Tensor-Pipe je Warp-Gruppe, gegen 165 Befehle [abgeleitet].

### 5.2 Issue-Auslastung und Iterationszeit [abgeleitet]

- V100: 600 GB/s / 552 B = 1,09 · 10^9 Warp-Gruppen/s, mal 165 = 179 G Befehle/s gegen 441 G/s Spitze (80 SM · 4 · 1,38 GHz, Taktannahme PCIE-Boost) = **41 %**; je SM IPC 1,6 von 4.
- RTX S0: 548 GB/s: 1,0 · 10^9 · 197 = 196 G/s gegen 510 G/s (72 · 4 · 1,77 GHz, Taktannahme) = **39 %**.
- Eine Warp-Iteration dauert V100 1,74 µs (≈ 2400 Takte), RTX 1,16 µs (≈ 2050 Takte). Mit 6 Warps je Scheduler (V100) ist der Scheduler **zu 59 % leer**: die Warps warten, sie werden nicht verdrängt.
- Je SM kommen auf beiden Karten **7,5–7,6 GB/s** an, obwohl die V100 24 und die RTX 16 Warps je SM hält. Gleiche Rate trotz anderer Warpzahl heißt: etwas je SM sättigt (Anzahl ausstehender Ladeanforderungen, LSU/MIO, oder Latenz wächst mit der Warpzahl) [vermutet, Test E1].

---

## 6. Änderungen, Ersparnis, Risiko (Rangliste)

Ersparnisse sind **abgeleitet**: Zeit neu = Zeit alt · (GB/s alt / GB/s neu), nicht additiv. MoE sind 25,66 von 60,74 ms je Runde.

### Rang 1: V100-Dekodierpfad mit explizitem Vorlade-Ring (m8n8k4, RB == 1)
- **Was:** die Struktur des m16n8k8-Kerns (`kern.cu:586–621`) auf `kern.cu:308–345` übertragen: Codes, Skalen und die A-Zeilen der nächsten Gruppe (besser zwei Gruppen) im Voraus laden, die Schleife ohne `unroll 4`-Leerung an der Kante. Die A-Ladung (`LD.E.128`, generisch) liegt laut SASS nur etwa 40 Befehle vor der ersten HMMA (Position 31/34 gegen 77); L2-Treffer-Latenz bleibt sichtbar.
- **Evidenz:** SASS: erste Code-Ladung der Gruppe nur etwa 60 Befehle vor ihrer Verwendung, danach Verdecken über eine Gruppe, Leeren an der Kante; V100 57–67 % gegen RTX 74–82 % bei gleich vielen Experten (Modell-Rechnung Abschnitt 5.2); Autor-Kommentar `kern.cu:315–318` (18 % durch Umsortieren).
- **Rechnung:** Die Iteration dauert jetzt 1,74 µs. Bei Vorlade-Tiefe zwei und etwa 0,5–0,9 µs Rechen- bzw. Latenzanteil je Gruppe wäre die Iteration 0,9–1,2 µs, das entspricht 75–100 % der DRAM-Bandbreite. Realistische Annahme 75–85 %: S1 3,86·(1−600/675…765) = 0,43–0,83 ms; S2 0,39–0,77; S3 0,39–0,75; **Summe 1,2–2,4 ms je Runde**, Mitte etwa 1,7 ms.
- **Risiko:** Register 77 → etwa 95 → 2 CTAs je SM (16 statt 24 Warps); die Wirkung kann den Gewinn auffressen. Gegenmittel: Tiefe 1 mit explizitem Ring (+10 Reg.), `__launch_bounds__(256, 3)`, oder 4-Warp-CTAs. **Bit-gleich**, weil jeder Warp seine K-Scheibe in derselben Reihenfolge summiert und die Warp-Summe `cs[0..7]` unverändert bleibt.

### Rang 2: S4: kein csf-Raster-Pfad mehr
- **Was:** die neun csf-Projektionen (Schicht 36 W2, 40 W2, 41 W13+W2, 42 W2, 43 W13+W2, 44 W13+W2; Zählung: 9 Expander-Launches = 342/38 [gemessen]) ohne Expander bedienen. Billigste Variante: **eine Basis je K-Scheibe statt je Zeile** (acht Basen je Zeile und Kachel). Die Spanne von 31 gilt dann nur innerhalb einer Warp-Scheibe (512 K bei W13, 256 K bei W2); ob 56 darin ≤ 31 wird, ist an den echten Planes **nur lesend auf dem Host** prüfbar (`scale_spread` je Scheibe, Schichten 36, 40–44). Falls nicht: sechs Bit (Nibble + zwei Hi-Bits, 0,75 des Rasters).
- **Rechnung:** Expander 0,55 ms [gemessen] + Mehrbytes und 80-Reg-Kern bis etwa 0,5 ms [vermutet; nicht von n trennbar] = **0,55–1,05 ms auf S4**. VRAM: Sechs-Bit-Planes 1,18 MB je Experte gegen csf 0,85 MB (+0,33 MB · 288 · 6 Projektionssätze ≈ +0,57 GB), das gemeinsame Raster (0,45 GB) entfällt, sobald keine csf-Schicht bleibt: netto etwa +0,1 GB [abgeleitet; auf S4, der engsten Stufe, mit memdiag belegen].
- **Risiko:** niedrig; die Skalenbytes bleiben gleich, die Produkte gleich: **bit-gleich**.

### Rang 3: RTX S5-Kern auf drei CTAs je SM
- **Was:** `__launch_bounds__(256, 3)` für den m16n8k8-S5-Kern bei `RB == 1` (heute Mindestblöcke 0 bei RB = 1, `kern.cu:502`): 91 → 80 Register (11 weniger). Die Raster-Variante belegt mit 80 Registern schon drei.
- **Rechnung:** Warps je SM 16 → 24; ob das Zusatz-MLP trägt, ist offen: 0–4 % der S5-Kernzeit (S0 8,71 ms + S4 etwa 4 ms) = **0–0,5 ms** [vermutet]. Spillgefahr (STACK in Varianten bis 16 B).
- **Risiko:** niedrig, bit-gleich; billiger Versuch.

### Rang 4: Leere CTAs und Schwanz
- **Was:** kein Launch von 40 Zeilen bei 24 aktiven: Gruppenzahl aus `goff` im Kern lesen und CTAs über `blockIdx.y` strided auf 12–16 Zeilen verteilen, oder die Zeilenzahl aus einer Obergrenze bilden, die die Routing-Verteilung hergibt (p99 n ≈ 36 [Zählerlauf: p90 ≈ n+5]). Schwanz: nur über kleinere Arbeitseinheiten oder Stream-K lösbar; Stream-K mit fp32-Teilsummen über Atomics ändert die Summenreihenfolge (nicht bit-gleich, Prüfstufe 1).
- **Rechnung:** leere CTAs 1–3 %, Schwanz 4–8 % der V100-Kernzeit: leere CTAs allein 0,25–0,5 ms; der Schwanz bis 0,9 ms, aber nur mit riskanten Umbauten.
- **Risiko:** mittel, Aufwand eher groß für wenig.

### Rang 5: kleinere Launch-Posten
- Act in das A-Laden des Down-Kerns fassen: spart Act (3,1–4,5 µs) und die Lücke (etwa 3 µs) = etwa 7 µs · 42 Schichten = **etwa 0,3 ms**; bit-gleich nur, wenn die Rechnung von `triton_poi_fused_clamp_copy` (Clamp, Silu, Mul, fp16-Rundung) exakt übernommen wird; sonst Stufe 1. Die Silu wird je CTA neu gerechnet (64-fach Redundanz, Transzendenten klein gegen 141 kB Gewichtsstrom).
- `grouped_topk…` auf V100 10 µs, `_group_routing` 4,6–5,1 µs je Schicht: zusammenlegen oder verkürzen = 0,15–0,2 ms.

### Rang 6: Kernfremd, aber hier relevant: Shared-Experte parallel
Die Shared-Experten-Kerne (`fp8_qpn8`, 30–35 + 19 µs je Schicht, 49–64 % der Spitze) laufen auf demselben Stream hinter dem MoE-Kern (kein Überlapp gemessen). In vLLM gibt es den Aux-Stream-Pfad (`runner/shared_experts.py:73–95`, Schalter `VLLM_DISABLE_SHARED_EXPERTS_STREAM` bzw. Layer-Attribut `_vllm_disable_shared_experts_stream`). Parallel gestartet könnte der Shared-Experte die Schwanz- und Rampenlücken des routed-Kerns und die ungenutzten 33 % der V100-Bandbreite füllen: **0,3–1,0 ms** [vermutet; prüfen, warum nur ein Stream]. Nicht Teil des Kerns, nicht gemessen.

### Rang 7: Spekulativ
- **Breitere Ladungen:** Prepack so, dass eine Lane zwei aufeinanderfolgende Gruppen als 16 B liest (der Prefill-Pfad `XCH = 2` liest ohnehin zwei Gruppen). Halbiert die Ladebefehle, verdoppelt die Anfragegröße. Wirkung unbekannt (0–0,8 ms); bit-gleich, aber Prepack und beide Pfade ändern sich (`qpn_prepack`, `dispatch.py:154`).
- **Entpack-Diät** (PRMT-Tabelle, weniger IMAD-Verschiebungen): 25–30 % weniger Befehle möglich; nützt nur, wenn die Issue-Seite zählt (V100 41 % Auslastung, RTX 39 %: spricht dagegen); erst nach E2.
- **Keine Aussicht:** mehr N-Teilung je Experte (Wellenzahl reicht), Split-K über CTAs für Down (Teilsummen-Reihenfolge ändert sich, Wellen reichen), Up+Act+Down in einem Kern (Gerätesynchronisation je Experte, kein Garantie gleichzeitiger Residenz).

### Übersicht (ms je Runde von 60,74; bit-gleich = ja außer wo vermerkt)

| Rang | Maßnahme | betrifft | Ersparnis | Beleg | Risiko |
|---|---|---|---|---|---|
| 1 | V100-Dekodierpfad mit Vorlade-Ring | S1–S3 | 1,2–2,4 (Mitte 1,7) | SASS + Modell, abgeleitet | mittel (Register) |
| 2 | S4 ohne csf-Expander (Basis je K-Scheibe oder 6 Bit) | S4 | 0,55 sicher, bis 1,05 | Expander gemessen | niedrig |
| 3 | RTX S5-Kern auf 3 CTAs/SM | S0, S4 | 0–0,5 | Register gemessen, Wirkung vermutet | niedrig |
| 4 | Leere CTAs vermeiden | alle | 0,25–0,5 | Rechnung | mittel |
| 5 | Act in Down fassen; Routing kürzen | alle | 0,3 + 0,15–0,2 | gemessen (Lücken) | niedrig bis mittel |
| 6 | Shared-Experte parallel | alle | 0,3–1,0 | vermutet | mittel |
| 7 | 16-B-Ladungen / Entpack-Diät | alle | 0–0,8 / 0–1 | vermutet | mittel |

Summe der gestapelten Rangliste 1–5 bei mittleren Annahmen: etwa 3,0 ms (5 %); die Decke „alles auf 85 %“ liegt bei 3,4–4,6 ms.

---

## 7. Tests und Messungen für die Hauptsitzung

### 7.0 Voraussetzung: Zähler und nsys im selben Lauf
Die Bandbreitenzahlen in Abschnitt 4 vergleichen zwei verschiedene Läufe. Entweder die Expertenzahl in den nsys-Lauf zurückschreiben (Zähler-Zweig `diag-expert-count`, `moe_runner.py`) oder die Auswertung je Runde und Schicht auf denselben Schritten rechnen. Ohne das bleiben V100 57–67 % eine Spanne.

### 7.1 Mikro-Benchmark im CUDA-Graph mit echtem Routing
- **Kern direkt:** `torch.ops._C.skinny_moe_qpn_sm70` mit den Argumenten aus `_grouped_projection` (`dispatch.py:921–935`): `x` fp16 [5, 4096] (Up) bzw. [40, 2048] slot-major (Down), `qcodes`/`qscales`/S5-Planes aus einem echten Prepack (`qpn_prepack`, `dispatch.py:154–222`; echte Schicht 14 für V100, 35 für RTX, damit die Skalenverteilung stimmt), `gscales` [288] fp32, `perm/gids/goff` aus `group_routing` (`routing.py:82`), `topk = 8`, `num_tokens = 5`, `splitk = grouped_splitk(K, order)`.
- **Gewichtsspeicher größer als der L2** und wechselnd: mindestens sieben Schichten hintereinander (V100-Stufe) in einem Graphen, jede mit eigenem `topk_ids` aus den Stichproben unten; der Graph enthält je Schicht Up, Act, Down, Reduce wie in der Produktion. Zeit je Graph / Zahl der Schichten, A/B/A/B nach den Regeln des Projekts, GPU vorher über GPU-Orc halten.
- **`topk_ids` passend zur Verteilung erzeugen:** Je Schicht liefert `rank*.json` ein Histogramm `layers[<name>][4][d]` = Zahl der Schritte mit **d verschiedenen Experten** bei 5 Zeilen (Index 4; `eval_expert_count.py` liest dasselbe). Für jeden Schritt: (a) d aus dem Histogramm der Schicht ziehen; (b) d Experten-IDs zufällig aus 0–287 wählen (die Adresslage ist bei Gewichten über L2-Größe unerheblich); (c) jeden der d Experten zuerst einer zufälligen Zeile zuweisen (mit höchstens 8 je Zeile), dann die restlichen 40 − d Plätze mit Experten füllen, die in der jeweiligen Zeile noch nicht vorkommen; dabei Zipf-gewichtet (Exponent so, dass das mittlere Mehrfachauftreten 40/d beträgt und die obersten Experten 3–5 Zeilen halten, wie im echten Verify-Schritt, wo Anker und vier Entwurfs-Token überlappen). Jede Zeile enthält acht verschiedene Experten.
- **Sweep zusätzlich zur Verteilung:** feste d ∈ {16, 20, 24, 28, 32, 36, 40}, um Zeit gegen d (Steigung = Grenzrate, Achsenabschnitt = fester Anteil, Schwanz) sauber zu bekommen; in nsys-Läufen ist das wegen der Schichttiefe nicht zu trennen.
- **Ausgabe je Aufruf:** µs für Up und Down getrennt, daraus GB/s mit 13,574 MB je Experte (S5) bzw. 14,147 MB (Raster), `memory.used` je Karte (Regel des Projekts).
- Vorhandene Vorlage: `benchmarks/kernels/` in `1Cat-vLLM-v14` ist reich an `benchmark_sm70_*`-Skripten; der Commit 79e07bfef7 beschreibt ein CUDA-Graph-Benchmark desselben Kerns gegen die ausgelieferte `_C` mit synthetischen Gewichten (Aufbau übernehmen, Routing wie oben ersetzen).

### 7.2 Entscheidende Vorversuche (vor jedem Umbau)
- **E1 „nur Laden“:** denselben Kern mit gleicher Schleife, aber Entpacken und MMA durch das Aufsummieren der geladenen Wörter ersetzt (XOR), gleiche Register/Belegung erzwungen. ≥ 85 % → die Last ist die Rechenseite (Befehlszahl, Rang 7 vorziehen); ≈ 7,5 GB/s je SM → Zugriffsstruktur/Vorlade-Tiefe/Ladegröße ist die Grenze (Rang 1, 3, 7a vorziehen).
- **E2 „nur Rechnen“:** Gewichte aus einem L2-großen Fenster (ein Experte, alle CTAs lesen dasselbe), sonst unverändert. Gibt die Issue-Grenze.
- **E3 `ncu` auf einem Up-Launch je Karte:** `sm__inst_executed`/`smsp__issue_active`, `smsp__warp_issue_stalled_long_scoreboard`/`_wait`/`_math_pipe_throttle`/`_lg_throttle`/`_mio_throttle`, `dram__throughput`, `l1tex__t_sectors_pipe_lsu_mem_global_op_ld`, `sm__warps_active.avg.pct_of_peak_sustained_active`. Entscheidet, ob Latenz (long_scoreboard) oder Issue/LSU (lg_throttle) vorn liegt.
- **E4 Gitterzeilen:** derselbe Kern mit `grid.y = aktive Gruppen` (Host kennt sie im Benchmark) gegen 40: misst die Kosten der leeren CTAs (Rang 4).
- **E5 Belegung:** `__launch_bounds__(256, 3)` auf dem m16n8k8-S5-Kern bei RB = 1; Registerzahl, Spill (STACK), Zeit (Rang 3).
- **E6 Skalen-Spanne je K-Scheibe:** auf dem Host aus den echten Planes der Schichten 36, 40–44 `scale_spread` je Warp-Scheibe (nur lesend, keine GPU) → entscheidet zwischen Basis je Scheibe und Sechs-Bit (Rang 2).

### 7.3 Abnahme
- **Bit-gleich** (Rang 1, 2, 3, 4, 7): neuer gegen alten Kern auf identischen Eingaben, byteweiser Vergleich von `y_slots` über mindestens die Verteilungsstichprobe und die Sweep-Werte; dazu `tests/kernels/moe/test_skinny_sm70_moe.py` (77 Tests laut Commit 79e07bfef7) auf V100 **und** RTX 8000. Jede Änderung, die `SPLITK`, `NT` oder die Summenreihenfolge berührt, ist **nicht** bit-gleich → Prüfstufe 1 (A/B, Texte gleich).
- **Prüfstufen für den Einbau:** Stufe 2 (Tempo ist das Ziel, Schwelle 1,5 % für generische Hebel; der V100-Ring fällt unter „generisch“: MoE-Kern aller PP-Stufen und weiterer Modelle: DSv4, Flash-Next). Messen wie im Projekt: A/B/A/B, Decode kurz und bei langem Kontext (16k und nahe Maximum), Boot-für-Boot-Textvergleich, `memory.used` je Karte (S4 für Rang 2 zwingend).
- **Capture-Größe:** Der Entwurf muss vier Token lang sein (5-Zeilen-Schritte), Capture-Größe 5.

---

## 8. Offene Punkte und Grenzen

- Kein ncu, kein Lauf: die Aussage „Latenz/Vorlade-Tiefe“ in Abschnitt 5.2 ist ein Modell aus Zählern, SASS und Zeiten, keine Messung der Stall-Gründe.
- Taktannahmen (V100-PCIE 1,38 GHz, RTX 8000 1,77 GHz) sind Nenn-Boosttakte; unter nsys-Last können sie niedriger liegen. Relativ zueinander (V100 gegen RTX) ändert das die Schlüsse nicht, die absoluten IPC-Zahlen schon.
- Der nsys-Lauf enthält S4-Schichten mit ungewöhnlich hohen Zeiten bei S5-Kernen (Schichten 40 und 42): wahrscheinlich andere Expertenzahl in diesem Lauf als im Zählerlauf, nicht der Kern.
- 38 Runden je Stufe, ein Text: Streuung der Kernzeit je Schicht min…max etwa ±35 % (V100 Up 245–501 µs).
- Nichts im Baum, in `config.yaml` oder in Git wurde angefasst. Hilfsdateien nur im Scratchpad (`f70.sass`, `f75.sass`, `ec_eval.txt`).
