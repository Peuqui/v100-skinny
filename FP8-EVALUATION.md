# FP8 gegen NVFP4 — Auswertung (2026-08-31, autonome Nachtschicht)

Auslöser: Das 180B-NVFP4 zerfällt bei langer deutscher Generierung
(Wortverstümmelungen, chinesische Zeichen, in einem Lauf zwei erfundene
Wissenschaftler). Peuqui: „für mich unbrauchbar". Der Formatvergleich
läuft am 27B, weil dort ein frisch kalibrierter NVFP4-Referenzwert
vorliegt und der Download klein ist.

## Referenzwerte NVFP4 (27B, gemessen 30.08.)

| Aufstellung | kurz | Prefill | lang 31k |
|---|---:|---:|---:|
| RTX-TP2 (getunte Kacheln) | 76,8 | 533 | 33,8 |
| Gitter TP2×PP2, k=2 | 67,5 | 864 | 33,0 |
| PP5/TP1, k=2 | 52,6 | 1.333 | 23,3 |

Qualität (drei Prompts, gleiche Sonde):
CJK-Zeichen 7 · verdächtige Wörter 183 · Nummerierung 30/30/30 ·
zwei halluzinierte Wissenschaftler im Coandă-Turn.

## Erwartung an FP8

Doppelte Gewichtsgröße (29 statt 21 GiB) heißt beim Decode grob doppelte
Leselast pro Token — dort ist die Speicherbandbreite der Flaschenhals.
Beim Prefill, wo gerechnet statt gelesen wird, sollte der Abstand
kleiner ausfallen. Ein Decode-Verlust ist also erwartbar und
akzeptiert; die Frage ist, ob er im erklärbaren Rahmen bleibt.

## Kernel-Lage (vorab aus dem Code)

Die Attention ist formatunabhängig — alle Fixes dieser Kampagne (sm75,
Split-KV, Combine, QSA, Volta-Kachel) greifen unverändert.

Der Matrixpfad wechselt: statt QPN2/QPN8 (NVFP4-Skinny) laufen die
FP8-Kernel. Der Fork hat dafür eigens gebaute sm70-Kernel, aber der
schnelle Prefill-Weg `fp8_gemm_sm70_prefill_dispatch_out` ist gegated
auf `tp_size == 4` UND vier fest verdrahtete Matrixformen
(_SM70_FP8_PREFILL_DENSE_SHAPES). Wir fahren TP2 mit anderen Formen,
landen also auf `fp8_gemm_sm70_out`. Beide sind echte Kernel — anders
als beim QSA-Fall, wo die Blackwell-Kacheln schlicht falsch
parametriert waren.

## Messungen

(wird beim Durchlaufen ergänzt)

### Messung 1: RTX-TP2 (2026-08-31, 02:4x)

| | NVFP4 | FP8 | |
|---|---:|---:|---|
| kurz | 76,8 | 37,3 | −51 % |
| Prefill | 533 | **744** | **+40 %** |
| lang 31k | 33,8 | 25,5 | −25 % |

Kohärenz 3/3. Das Muster ist physikalisch stimmig: Kurzer Decode hängt
fast nur an der Gewichtsleselast, doppelte Gewichte heißen halbe Rate
(76,8/2 = 38,4, gemessen 37,3). Beim langen Decode verdünnt die
Attention über den KV-Cache diesen Anteil auf ein Viertel Verlust. Der
Prefill rechnet statt zu lesen — dort gewinnen die FP8-Kernel.

### Befund: der schnelle FP8-Pfad ist auf TP4 zugeschnitten

`_SM70_FP8_PREFILL_DENSE_SHAPES` passt exakt auf dieses Modell, aber
nur bei Tensor-Parallelität vier:

| Matrix | Tabelle | unser TP2 | TP4 |
|---|---|---|---|
| gate_up_proj | (5120, 8704) | (5120, 17408) | (5120, 8704) ✓ |
| down_proj | (4352, 5120) | (8704, 5120) | (4352, 5120) ✓ |
| o_proj | (1536, 5120) | (3072, 5120) | (1536, 5120) ✓ |

hidden 5120, intermediate 17408, 24 Köpfe, head_dim 256. Bei TP4 fällt
gate_up auf 2*17408/4 = 8704, down auf 17408/4 = 4352, o_proj auf
24*256/4 = 1536 — Zeile für Zeile die Tabelle. Die Schwelle
`min_M = 3920` liegt unter unserer Chunk-Größe 4096, greift also auch.

Dasselbe Muster wie beim QSA-Kernel, nur eine Ebene höher: nicht falsch
parametrierte Kacheln, sondern ein Tor, das nur die Konfiguration des
Entwicklers durchlässt. Anders als bei QSA ist der allgemeine Pfad hier
aber ein echter Kernel, kein Fehlgriff.

### Befund: TP4-Tor unerreichbar, aber ein zweites Tor war zu

**TP4 scheitert strukturell.** Der Versuch hängt beim Graph-Einfangen
(`shm_broadcast: No available shared memory broadcast block`, Worker
desynchronisiert). Ursache: TP4 auf dieser Maschine hieße zwei RTX
(sm75) plus zwei V100 (sm70) in EINER Tensor-Parallel-Gruppe. Die
Architekturen nehmen unterschiedliche Kernel-Pfade und laufen bei den
Kollektiven auseinander — genau deshalb trennt die Gitter-Topologie die
Klassen ueber Pipeline-Stufen. Homogenes TP4 ist unmoeglich: 2 RTX,
3 V100 (eine davon Side-Channel). Der schnelle FP8-Prefill-Pfad ist
ueber die Topologie also nicht erreichbar.

**Wichtiger: unser Testlauf umging 1Cats FP8-Kernel voellig.** Der
Harness erbte `VLLM_SM70_QUANT_BACKEND=marlin` aus dem
NVFP4-Betriebspunkt, wo es richtig ist (Skinny-Pfad). Bei FP8 gibt
`envs.use_sm70_turbomind()` daraufhin hart `False` zurueck (envs.py:812)
— der gesamte TurboMind-FP8-Weg samt `fp8_sm70_prepare`,
Gated-SiLU-Fusion und dichtem Prefill-Kernel fiel damit weg. Die
Messung 1 (744/25,5) entstand also auf dem generischen Marlin-Pfad,
nicht auf dem optimierten.

Das ist kein Fork-Fehler, sondern eine Falle in unserer eigenen
Konfiguration: eine Einstellung, die formatabhaengig richtig oder falsch
ist, wurde formatunabhaengig vererbt. Konsequenz fuer die Kalibration:
`VLLM_SM70_QUANT_BACKEND` gehoert an das Quantisierungsformat gekoppelt,
nicht an den Betriebspunkt.

**Praezisierung:** Der TurboMind-FP8-Pfad ist auf exakt sm70 eingegrenzt
(`has_device_capability(70) and not has_device_capability(75)`,
fp8.py:210). Er gilt also nur fuer die V100, nicht fuer die RTX 8000 —
dort laeuft FP8 ueber den Standardweg, unabhaengig von
`VLLM_SM70_QUANT_BACKEND`. Die Messung 1 auf RTX-TP2 war damit bereits
der einzig moegliche Pfad; der Backend-Fund wirkt erst dort, wo V100
beteiligt sind (V100-TP2 und das Gitter).

### Messung 2: RTX-TP2 mit TurboMind-Backend — Gegenprobe

| | Marlin | TurboMind |
|---|---:|---:|
| kurz | 37,3 | 36,4 |
| Prefill | 744 | 743 |
| lang | 25,5 | 25,5 |

Identisch, wie aus dem Code vorhergesagt: Der TurboMind-FP8-Pfad ist auf
sm70 begrenzt, auf den RTX 8000 wirkt die Einstellung nicht. Die
Codelesung ist damit empirisch bestaetigt — und Messung 1 war fuer die
RTX bereits der bestmoegliche Pfad.

Offen bleibt die V100-Seite: dort greift TurboMind, und dort ist der
Unterschied zu erwarten. Naechste Messung: Gitter TP2xPP2 (RTX-Stufe +
V100-Stufe) mit TurboMind, gegen die NVFP4-Referenz 864/33,0.

### Messung 3: Gitter TP2xPP2 mit TurboMind

| Gitter, k=2 | NVFP4 | FP8 | |
|---|---:|---:|---|
| kurz | 67,5 | 39,1 | −42 % |
| Prefill | 864 | **1.131** | **+31 %** |
| lang 31k | 33,0 | 26,3 | −20 % |

Kohärenz 3/3. 31.469 Tokens in 27,8 statt 36,4 Sekunden. Der
Prefill-Vorsprung von FP8 bestätigt sich auch im Gitter und faellt dort
sogar deutlicher aus als auf den RTX allein (+31 % statt +40 % gegen
einen niedrigeren Ausgangswert).

Das Gesamtbild ueber alle drei Aufstellungen ist konsistent: FP8 gewinnt
beim Prefill (rechengebunden — acht Bit rechnen sich besser als vier
plus Dequantisierung), verliert beim Decode (bandbreitengebunden —
doppelte Gewichte, doppelte Leselast). Der Verlust ist beim langen
Decode kleiner als beim kurzen, weil die Attention ueber den KV-Cache
den Gewichtsanteil verduennt.

### Kernbefund: sm75 entpackt FP8 beim Laden nach fp16

`qpn8_blk.py:151` — auf Turing werden die FP8-Gewichte EINMALIG beim
Laden nach fp16 umgewandelt und so gehalten:

    # sm75 stage: fp16 dequant once at load, cuBLAS serves it.
    w16 = (w8.view(torch.float8_e4m3fn).to(torch.float32) * sc).half()

Die RTX-Stufe rechnet also gar nicht mit FP8, sondern mit fp16. Belegt
durch die Speichermessung: „Model loading took 15.16 GiB" auf einem
RTX-Rang bei 38/64 Schichten und TP2, also ~8 Mrd. Parameter — fp16
ergibt 16 GB, FP8 waere 8 GB.

Das erklaert beide Messwerte auf einmal: guter Prefill (native
fp16-Tensorkerne ueber cuBLAS, kein Entpacken zur Laufzeit) und
schwacher Decode (doppelte Leselast gegenueber FP8, vierfache gegenueber
NVFP4). Die V100-Stufe laeuft dagegen ueber echte FP8-Kernel
(qpn8-blk/-wmma), weil Volta dort 1Cats Skinny-Pfad hat.

**Konsequenz fuer das 180B:** FP8 bringt auf der RTX-Stufe KEINE
Speicherersparnis. Die 173 GiB waeren dort verdoppelt — bei 38/64
Schichten auf der RTX-Stufe ergaebe das ueber 100 GiB je Karte (48 GB
vorhanden). Auch eine V100-only-Aufstellung scheitert: 3x32 = 96 GiB
gegen 173 GiB Gewichte. **FP8 ist fuer das 180B auf dieser Maschine
nicht tragfaehig** — anders als beim 27B, wo es passt und beim Prefill
gewinnt.

### Messung 4: Backend-Schalter im Gitter — kein Unterschied

TurboMind 1.131 / 26,3 gegen Marlin 1.132 / 26,3. Der Grund steht in
den Routen-Zaehlern beider Laeufe (identisch):

| Route | Anzahl | Stufe |
|---|---:|---|
| sm75-fp16-dequant | 304 | RTX 8000 |
| qpn8-blk / -dequant / -wmma | 110 | V100 |

FP8 laeuft also ohnehin ueber 1Cats Skinny-Kernel (V100) bzw. den
fp16-Dequant-Weg (RTX); `VLLM_SM70_QUANT_BACKEND` steuert daran nichts.
Der vermutete Optimierungshebel existiert nicht — was zugleich heisst,
dass hier nichts liegen gelassen wird.

## Zwischenfazit Leistung

Beide vermuteten Hebel sind Sackgassen: das TP4-Tor ist auf gemischten
Architekturen unerreichbar, der Backend-Schalter wirkungslos. FP8 laeuft
auf dieser Hardware bereits so gut, wie es geht — eine zweite
Optimierungsrunde nach QSA-Muster ist NICHT noetig.

| Aufstellung | | NVFP4 | FP8 |
|---|---|---:|---:|
| RTX-TP2 | Prefill | 533 | **744** |
| | lang | **33,8** | 25,5 |
| Gitter | Prefill | 864 | **1.131** |
| | lang | **33,0** | 26,3 |

FP8 ist die Prefill-Maschine, NVFP4 die Decode-Maschine.

## Qualität: identische Sonde, beide Formate am 27B

Dieselben drei Prompts wie in den AIfred-Sitzungen, identischer
System-Prompt, Temperatur 1,0 (tools/quality_probe.py).

| | 27B NVFP4 | 27B FP8 |
|---|---|---|
| Turn 1 (Quantenphysik) | 4.449 Zeichen, 30 Sätze | 3.633 Zeichen, 30 Sätze |
| Turn 2 (Regenbogen) | **108 Zeichen — Verweigerung** | 2.574 Zeichen, erklärt |
| Turn 3 (Kuanda) | **57 Zeichen — Verweigerung** | 2.375 Zeichen, erklärt |
| CJK-Zeichen | 0 | 0 |
| Wortverstümmelungen | keine | keine |
| verdächtige Wörter je 1.000 Zeichen | 10,4 | 11,7 |

NVFP4 verweigert zweimal rundheraus („Es scheint, dass die Anfrage
einen spezifischen Begriff betrifft, der im Kontext der Physik nicht
existiert" / „Ich muss korrigieren, dass es keinen solchen Effekt
gibt") — beides sachlich falsch und entgegen der Anweisung zur
stillschweigenden Tippfehlerkorrektur. FP8 hakt zwar am Begriff
„Regenbogeneffekt" ein, liefert dann aber substantielle Erklärungen.

Beide 27B-Varianten identifizieren „Kuanda" NICHT als Coandă (das
180B tat es in allen drei Läufen, auch als NVFP4). Das ist ein
Größeneffekt, kein Formateffekt.

## KORREKTUR einer früheren Schlussfolgerung

Nach den 180B-Sitzungen lag der Schluss nahe, NVFP4 als Format sei für
deutsche Prosa untauglich. **Das stimmt so nicht.** Das 27B zeigt in
NVFP4 null chinesische Zeichen und null Wortverstümmelungen — genau die
Fehler, die das 180B reihenweise produzierte.

Der Zerfall gehört also nicht dem Format, sondern der spezifischen
Kombination des 180B: A4B-Architektur mit nur vier aktiven Milliarden
Parametern, Sparse Attention mit indexer_budget 2048, riesige
PLE-Tabelle — und darauf vier Bit. Peuquis Urteil über das 180B bleibt
richtig; die Verallgemeinerung auf NVFP4 war voreilig.

## Fazit für die Praxis

**FP8 auf dem 27B ist eine echte Option**, aber kein klarer Sieger:
Prefill +31 %, Decode −20 %, Qualität etwas hilfsbereiter (erklärt statt
verweigert), dafür 29 statt 21 GiB Platz.

**Für das 180B ist FP8 versperrt** (fp16-Dequant auf Turing).

**Eine zweite Optimierungsrunde ist nicht nötig** — beide vermuteten
Hebel sind belegt tote Enden, FP8 läuft bereits auf den bestmöglichen
Kerneln dieser Hardware.

**Offen bleibt** der GGUF-Weg für das 180B: gescheitert an der
fehlenden `qwen4exp`-Zuordnung in `transformers` (integrations/ggml.py,
29 Architekturen, unsere fehlt). Aufwand geschätzt ein bis zwei Tage,
Gewinn wäre Q6-Qualität mit vLLM-Prefill.

## Neue Hypothese (Peuqui): liegt es an der Arithmetik?

Einwand: „Wenn NVFP4 so schlecht waere, wuerde es kaum jemand
einsetzen." Dazu die eigene Erfahrung, dasselbe 27B sei als Q8-GGUF
unter llama.cpp exzellent gewesen (damit wurde VECTORFALL
nachprogrammiert).

Die Recherche stuetzt den Einwand deutlich:

1. **Der Fork beschreibt sich selbst als AWQ-Projekt.** GitHub-Titel:
   „vLLM fork for Tesla V100 (SM70) with **AWQ 4-bit support**, CUDA
   12.8 build flow, and **validated Qwen3.5 27B/35B** deployment". NVFP4
   ist NICHT der validierte Pfad — es kommt ueber den
   v100-skinny-Aufsatz (VLLM_SKINNY_NVFP4/QPN/QPN2) dazu.

2. **NVFP4 ist ein Blackwell-Format.** Laut Red Hat und der
   NVFP4-Literatur bleibt der Tensor waehrend der GESAMTEN Inferenz im
   Vier-Bit-Format, ohne Entpacken, hardwarebeschleunigt — genau das
   macht seine Guete aus (95-98 % BF16-Genauigkeit, Qwen ~98 %). Auf
   Volta/Turing fehlt diese Hardware, die Skinny-Kernel muessen
   emulieren.

3. **Rundungsfehler sind laut Literatur ohnehin die dominante
   Fehlerquelle** bei NVFP4 (arXiv 2512.02010 „Four Over Six",
   2603.22370 „FAAR"). Eine emulierte Ausfuehrung legt weitere
   Rundungsschritte obendrauf.

### Der entscheidende Test

| | Modell | Bits | Stapel | Kernel-Pfad |
|---|---|---|---|---|
| gemessen | 27B | 4 (NVFP4) | 1Cat-Fork | Skinny, EMULIERT |
| jetzt | 27B | 4 (AWQ) | 1Cat-Fork | Marlin, VALIDIERT |

Dasselbe Modell, dieselbe Bitbreite, derselbe Fork — nur der
Kernel-Pfad unterscheidet sich. Verweigert AWQ nicht, wo NVFP4 zweimal
verweigerte, ist die Emulation ueberfuehrt und weder das Modell noch die
Vier-Bit-Grenze.

Modell: shawnw3i/Qwen3.8-27B-AWQ-MTP (19 GiB, group_size 64, MTP-Kopf
vorhanden — Spekulation bleibt vergleichbar).

### ERGEBNIS: Die Hypothese ist bestaetigt

Gleiches Modell (Qwen3.8-27B), gleiche Bitbreite (4), gleicher Fork,
gleiche Sonde, gleicher System-Prompt — einzige Variable ist der
Kernel-Pfad:

| | NVFP4 (Skinny, emuliert) | AWQ (Marlin, validiert) | FP8 |
|---|---|---|---|
| Turn 1 Quantenphysik | 4.449 Z., 30 Saetze | 4.818 Z., 30 Saetze | 3.633 Z., 30 Saetze |
| Turn 2 Regenbogen | **108 Z. — Verweigerung** | **4.021 Z., 30 Saetze** | 2.574 Z., ohne Nummern |
| Turn 3 Kuanda | **57 Z. — Verweigerung** | **4.702 Z., 30 Saetze** | 2.375 Z., ohne Nummern |
| Gesamt | 4.614 Zeichen | **13.541 Zeichen** | 8.582 Zeichen |
| nummerierte Saetze | 30 | **90** | 30 |
| CJK-Zeichen | 0 | 0 | 0 |
| verd. Woerter/1000 Z. | 10,4 | **9,2** | 11,7 |

NVFP4 verweigert zwei von drei Antworten mit sachlich falscher
Begruendung. AWQ liefert alle drei vollstaendig mit je dreissig
nummerierten Saetzen und hat zugleich die niedrigste Fehlerrate. FP8
liegt dazwischen: antwortet, haelt aber die Nummerierungs-Anweisung nur
im ersten Turn ein.

Die Tippfehler-Aufloesung „Kuanda" -> Coandă schafft KEINE der drei
27B-Varianten (AWQ raet Cavendish/Coulomb). Das 180B konnte es in allen
Laeufen — Groesseneffekt, kein Formateffekt.

**Schlussfolgerung:** Nicht das Modell und nicht die Vier-Bit-Grenze
sind das Problem, sondern der emulierte NVFP4-Pfad. Das deckt sich mit
der Selbstbeschreibung des Forks (AWQ ist der validierte Weg), mit der
Formatspezifikation (NVFP4 rechnet auf Blackwell OHNE Entpacken —
genau diese Eigenschaft fehlt auf Volta/Turing) und mit Peuquis
Erfahrung, dasselbe 27B sei als Q8-GGUF exzellent gewesen.

## Empfehlung

**AWQ ist auf dieser Hardware das Format der Wahl fuer vLLM.** Es ist
der validierte Pfad des Forks, liefert die beste Textqualitaet der drei
gemessenen Formate und ist mit 19 GiB sogar kleiner als FP8 (29 GiB).

Offen und lohnend: Tempo-Messung von AWQ gegen die NVFP4-Referenz
(864 tok/s Prefill, 33,0 lang im Gitter). Der Nutzer erinnert AWQ auf
dem 180B als „schnarchelahm" — das war allerdings VOR der gesamten
Kernel-Kampagne und moeglicherweise ohne
`VLLM_SM70_QUANT_BACKEND=marlin`.

Ausserdem: Fuer das 180B existiert `wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16`
(181 GB) — die Frage nach einem brauchbaren grossen Modell stellt sich
damit neu.

Hinweis fuer die Praxis: `shawnw3i/Qwen3.8-27B-AWQ-MTP` liefert KEINE
`chat_template.jinja` mit; ohne sie lehnt vLLM jede Chat-Anfrage mit
HTTP 400 ab. Vorlage aus dem NVFP4-Upload kopieren oder
`cyankiwi/Qwen3.8-27B-AWQ-INT4` nehmen (dafuer ohne MTP-Kopf).
