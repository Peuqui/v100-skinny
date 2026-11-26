# GLM skinny MoE: Mikrobenchmark und ncu (23.11.2026)

Werkzeug `benchmarks/kernels/benchmark_glm_skinny_moe_routing.py` (merge-1cat-2d218ab03 2af528531e, Paket-Zweig 341d5120c5): 7 echte Schichten
je Stufe in einem CUDA-Graphen, Routing aus den gemessenen Histogrammen (5-Zeilen-Prüfrunden), 200 Ziehungen. Logs
`~/.cache/bench-scripts/moebench_2026-11-23/`. Alle Varianten byteweise gleich zum Produktionskern (`--compare` gegen die Grunddumps).

## Durchsatz über die verschiedenen Experten (GB/s, mehr ist besser; je zwei Läufe)

| Stufe (Karte) | Produktion | S6 | lb3 (S6 + 3 CTAs RTX) | Ring (S6 + V100-Ring) |
|---|---|---|---|---|
| 1 (V100) | 634,8 / 634,0 | 634,2 / 635,0 | – | 676,8 / 676,4 (+6,6 %) |
| 0 (RTX) | 562,7 / 563,0 | 563,0 / 563,1 | 571,9 / 571,9 (+1,6 %) | – |
| 4 (RTX, csf-Schichten) | 550,3 / 548,7 | 547,7 / 547,7 | 551,4 / 552,2 | – |

µs je Schicht (weniger ist besser): Stufe 0 823,8 → lb3 810,6; Stufe 4 530,6 → S6 529,5 → lb3 526,0. Grenzrate aus dem Sweep d = 16…40:
V100 700–717 GB/s (Ring 739–746), RTX 592–597 GB/s. Der Mikrobenchmark misst den csf32-Expander der Produktion nicht (0,55 ms je Runde,
entfällt mit S6).

## ncu (Version 2022.4.1, `--graph-profiling node`, 6 Launches je Lauf, Zeiten unter ncu-Takten)

| Größe | V100 Produktion | V100 Ring | RTX Produktion |
|---|---|---|---|
| Up-Launch µs (weniger ist besser) | 367 / 316 / 321 | 339 / 309 / 300 | 619 / 520 / 520 |
| dram % der Spitze | 63–67 | 67–71 | 81–87 |
| Issue aktiv % | 50–52 | 72 | 59 |
| Warps aktiv % (Belegung) | 36 (3 CTAs, 77 Register) | 36 (3 CTAs, 80 Register) | 49 (2 CTAs, 91 Register) |
| Stall long_scoreboard (Up, je Warp) | 19 520 | 6 130 | 19 516 |
| Stall math_pipe_throttle | 3 153 | 6 848 | 4 742 |
| Stall not_selected | 3 790 | 10 057 | 4 254 |
| Befehle (Up) | 67,2 M | 91,0 M (+36 %) | 124,5 M |

Folgerung: Auf der V100 war der Kern latenzgebunden (Warten auf DRAM, wie MOE_KERN.md vermutete); der Ring drittelt dieses Warten,
dafür ist er jetzt ausgabegebunden (Issue 72 %, Math-Pipe und not_selected verdoppelt), weil die Schleife ohne Entrollung 36 % mehr
Befehle braucht (Adressrechnung). Nächster Hebel V100: dieselbe Ladereihenfolge mit weniger Befehlen (Schleife zweifach entrollen,
Adressrechnung teilen). RTX: 81–87 % DRAM, wenig Luft; dort nur noch lb3 (3 CTAs).

## Entrollte Ring-Schleife (Kette `chain_moeru_2026-11-23.sh`, 20:21–20:29)

Stufe 1 (V100), GB/s über die verschiedenen Experten (mehr ist besser), je zwei Läufe im Wechsel, alle bytegleich, Tests je 141 grün:
Paket (Ring ohne Entrollung) 674,2 / 673,6; zweifach entrollt (6053809ac3) 654,9 / 656,0 (−2,8 %); vierfach (dafae7a3c5) 643,5 / 640,3
(−4,8 %). Register sm_70 bei Entrollung 80 mit Auslagerung (S6-Instanz STACK 16). Entrollen verschlechtert; der Ring bleibt unentrollt
(Zweige moe-ring-unroll2/4 verworfen). Die Ausgabegrenze aus ncu ist damit nicht über die Adressrechnung zu lösen.

## Stufe-2-A/B Kernpaket (Zweig moe-kernel-pack: S6 + 3 CTAs Turing + V100-Ring)

GLM k = 4 (`ab_mpk_glm_2026-11-23a.log` T1 P2, dann AIfred-Übergabe; Fortsetzung `ab_mpk_glm_2026-11-23b.log` T1 P2, Variante mpk2):

| Größe | T (a / b) | P (a / b) | T gegen P |
|---|---|---|---|
| Runde greedy ms (weniger ist besser) | 55,3 / 55,3 | 57,0 / 57,0 | −1,7 ms |
| AIfred-Sampling alle tok/s | 42,32 / 42,33 | 41,08 / 41,14 | +2,9 % |
| Greedy tok/s | 52,31 / 52,25 | 50,67 / 50,70 | +3,2 % |
| Decode kurz tok/s | 48,73 / 48,86 | 47,28 / 47,34 | +3,1 % |
| GSM8K-Decode tok/s / richtig | 66,68 / 66,65, 29 | 64,50 / 64,54, 29 | +3,3 % |
| Langkontext ms je Schritt 150k / 250k / 516k | 63,9 / 64,9 / 67,3 (b) | 65,6 / 66,8 / 69,0 (a) | −2,6 bis −2,8 % |
| Prefill 8,7k s | 3,88 / 3,89 | 3,86 / 3,86 | +0,6 % |
| GPU-Spitze Karte 0 / 2 MiB | 47 373 / 45 885 (a) | 47 863 / 46 593 (a) | weniger |

Texte 8 von 8 tokengleich zur Produktionskontrolle (beide T-Arme), T-Journal: alle csf-Projektionen als Bitplanes, kein „stay compressed“.

DeepSeek-V4 (`ab_mpk2_dsv4_2026-11-23a.log` T P T P): Decode kurz 61,19 / 55,62 gegen 61,12 / 61,12, GSM8K-Decode 72,13 / 73,72 gegen
72,13 / 72,12, Langkontext ms je Schritt T3 58,0 / 61,9 / 64,3 gegen P2 58,3 / 61,4 / 65,0, GSM8K 28 überall, Texte T1 = P2.
T1 = P bis auf die Nachkommastelle (gleiche Token, gleiche Annahme); T3 erzeugte andere Token (GSM8K-Längen anders, 28 richtig), passend zu
neu geschriebenen Compile-Artefakten nach T1 („Texte folgen dem Compile-Artefaktsatz“). Je Schritt gleich: DeepSeek neutral.

Urteil: GLM +3 % in allen AIfred-Größen, Langkontext −2,7 % je Schritt, Qualität gleich, Speicher weniger; DeepSeek neutral. Ins Bündel v15.

## Shared-Experte auf vLLMs Aux-Stream (24.11.2026): nicht übernommen

Der generische Mechanismus (`shared_experts.py`, Decode ≤ 256 Zeilen) ist im Baum vollständig; die Produktionseinträge schalten ihn mit
`VLLM_DISABLE_SHARED_EXPERTS_STREAM=1` ab. Ohne die Env (`ab_shs_glm_2026-11-24a.log`, `ab_hps_{glm,dsv4}_2026-11-24a.log`): GLM Decode
+1,1 % (Runde 53,1 gegen 53,7 ms), AIfred-Sampling +1,2 %, Texte tokengleich; aber Prefill 8687 Token 3,94 gegen 3,86 s (+2,1 %), erstes Token
150k–516k +1,5 %. Die Env geht in den Compile-Schlüssel ein; die Gegenprobe `ab_frs_glm_2026-11-24a.log` (unveränderte Produktion, leerer
Cache) erzeugt dieselben Artefakte und prefillt 3,85–3,86 s: Der Verlust kommt vom anderen Graphen mit Stream, nicht vom Neukompilieren.
DeepSeek: Decode +0,8 %, Prefill gleich. Peuqui 24.11.: dem 1-%-Rest nicht weiter nachgehen; Env bleibt in den Einträgen.

## MoE-Fixkosten: Aktivierung und Reduktion als Kern-Epiloge (24.11.2026): verworfen

Zweig `moe-fixed-costs` (im Fork, nicht übernommen): SwiGLU-Aktivierung als Epilog des Up-Kerns (Gate- und Up-Kachel N/64 auseinander
gepaart, Inductor-Rechnung als PTX nachgebildet), gewichtete Reduktion als „letzter Ankömmling“-Epilog des Down-Kerns (Atomics). Kette
`chain_moefix_2026-11-24.sh`, Logs `~/.cache/bench-scripts/moefix_2026-11-24/`: 12 von 12 Läufen bytegleich zu den Grunddumps, Tests grün
(V100 190, RTX 292), aber µs je Schicht (weniger ist besser) Stufe 1 V100 491,0/492,1 → 516,3/516,2 (+5,0 %), Stufe 0 RTX 810,3/810,8 →
805,1/804,7 (−0,7 %), Stufe 4 RTX 524,1/523,6 → 518,7/518,6 (−1,0 %). Leere CTAs: laut Analyse wenige µs (steigen nach zwei Loads aus),
nicht umgebaut. Der Achsenabschnitt a (40–63 µs je Schicht) ist überwiegend Rampe und Schwanz der beiden großen Kerne.
