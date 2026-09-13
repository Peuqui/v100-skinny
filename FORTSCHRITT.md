# Fortschritt v100-skinny / 1Cat-vLLM — 20.08. bis 13.09.2026

Kurzbilanz für Peuqui. Zahlen sind Decode-Tokens pro Sekunde, greedy, aus den
jeweils genannten Messungen; Details und Belege in STAND.md, Memory und
`handover/`. Vergleiche gelten nur innerhalb einer Zeile: kurzer Prompt,
langer Kontext und vorhersagbarer Text sind verschiedene Maßstäbe.

## Ausgangslage (24.08.)

| Was | Wert |
|---|---|
| llama.cpp Produktion, 27B Q8 mit MTP n=3, RTX 8000 | 32 / 26 / 35 tok/s |
| vLLM upstream 0.27, 27B, RTX 8000, ohne Spekulation | 27,5–28,5 tok/s |
| v100-skinny (dnv2003) auf 2× V100 TP2, k=7, vorhersagbarer Text | 88 / 59 / 86 tok/s |
| DeepSeek-V4-Flash | llama.cpp mit dspark: 40 tok/s kurz, Prosa 21, Code 38; unter vLLM auf Volta lief er nicht |
| Qwen3.8-Flash-Next 180B unter vLLM auf Volta | lief nicht |
| Turing (RTX 8000) im Fork | unbrauchbar, kein korrekter Prefill |

Das Kriterium „vLLM muss llama.cpp mit MTP schlagen" war beim 27B am ersten
Tag erfüllt und ist es seitdem geblieben.

## Tempo heute (13.09., Produktionsstand work-main auf 1Cat main dfef3342)

| Modell, Betriebspunkt | Wert | Weg dorthin |
|---|---|---|
| **27B DFlash2**, 2× RTX 8000 TP2, 400 Token | **77,1** | 65 (10.09. MTP) → 69,1 (DFlash2) → 77,1 (Block-Pack + quantisierter Entwurfskopf, 10.09.) |
| **27B DFlash2**, 2× V100 TP2 | **76,4** | die RTX hat die V100 am 10.09. überholt; beide bitgleich |
| 27B MTP k=3, RTX, reines 1Cat-main + unsere PRs | 74,6 | 63,4 (V100) / 71,0 (RTX, E-1) → 74,6 (Split-Graphen, #618) |
| **Flash-Next 180B**, TP2×PP2 heterogen, 13k Vorkontext, Chat mit Denken | **37–48** | 33 (27.08., k=0) → 49/67 kurz (28.08., NVFP4-Entwurfskopf) → 20–21 bei 9–10k echtem Text (07.09.) → 37–48 bei 13k (12./13.09.) |
| Flash-Next, kurzer Prompt | 56 | Prefill 413–466 tok/s |
| **DeepSeek-V4-Flash**, TP1×PP5 alle fünf Karten, 13k Kontext | **15–19** | 4 (26.08., erster vLLM-Lauf) → 11× durch Per-Experten-MoE → 13–20 (02.09.) → 21/27 (03.09., fp16-mHC) → 25/28 heterogen (06.09., moe_qpn: llama.cpp-Schrittlatenz von 110 ms erreicht, Prosa eingeholt) → PP5-Produktion mit 65k Kontext; llama.cpp-Eintrag seitdem abgelöst |
| Turing-Attention (Paket C, sm75-FA2), 27B, 13k Vorkontext | TTFT 17,8 s statt 36,3, Decode 56 statt 15 | gemessen 12.09., Bauform bei 1Cat angefragt (#612) |

Kontext: alle drei Produktionseinträge fahren ihre volle Länge (27B und
Flash-Next 262.144, DeepSeek 65.536). Die Kuanda-Fangfrage besteht das
27B-Profil, Flash-Next weist sie in zwei von drei Fällen zurück (Modellgrenze,
nicht Stack).

## Stabilität: was gefunden und behoben wurde

| Datum | Befund | Behoben durch |
|---|---|---|
| 24.08. | PP + MTP: Draft-Token nicht auf alle Ränge, Ausgabe nur auf letztem Rang getrimmt | Broadcast + Trim (später PR #574) |
| 28.08. | Flash-Next mit BF16-Entwurfskopf: MTP war ein Verlust (14 statt 49 tok/s) | quantisierter Entwurfskopf (MTPQ-Checkpoint) |
| 04.–07.09. | Flash-Next PP2 + Spekulation verklemmt nach ~180 Token | ungepolsterte Broadcast-Form im V2-Runner (0b8f4cc), MTP stufenlokal (#573 gemergt) |
| 06.09. | Compile-Cache lud fremde Artefakte (KeyError skinny_codes) | Schlüssel kennt alle VLLM_-Schalter (#536 gemergt) |
| 07.09. | Gerätegates fragen Gerät 0 statt eigenes Gerät, gemischte Rigs rechnen falsch | #576 #599 #600 #618 (offen), #514 (gemergt) |
| 08.–09.09. | Turing bootet nicht / rechnet falsch / verliert Fusionen | #572 gemergt |
| 11.09. | Qwen3.8-Tool-Calls still verschluckt (hermes-Parser) | qwen3_coder-Parser in llama-swap |
| 12.09. | DeepSeek stürzt bei 7–11 Query-Token ab (SWA-Schwelle gegen C128A-Builder) | #603 |
| 12.09. | NCCL-Untergruppen ignorierten `--distributed-timeout-seconds` (Wachhund bei kaltem PP-Compile) | #619 |
| 12.09. | Turing bekommt unter `auto` fp8-KV und bootet nicht | #613 |
| 13.09. | Compile-Cache: erster Warmstart je Generation lädt nicht (torch 2.10, #173556 fehlt), danach lädt alles | Zwangsabschaltung entfernt (Fork + PR), Ursache dokumentiert |

Abnahmeprinzip seit 07.09.: Text-SHA über 400 Token, Annahmelänge, tok/s,
je Kartenpaar; DeepSeek 8/8-Kohärenz zweimal je Prozess; Flash-Next drei
Fragen mit 13k Vorkontext über den Produktionspfad. Bekannt seit 12.09.: auf
dem 0DOT3-Pfad ist jeder frische Compile eine Münze (Combo-Kernel-Wahl), der
Text-SHA ist deshalb nur bei gleichem Artefakt ein Korrektheitskriterium.

## Beiträge an 1Cat

11 PRs gemergt, 12 offen (#574 #576 #592 #599 #600 #601 #603 #604 #611 #613
#618 #619, dazu der Compile-Cache-PR in Anlage), Issues #441 #479 #612 #620.
Alle mit Messbeleg, Duplikatsuche und KI-Erklärung nach AGENTS.md.

## Optimierungspotenzial, ehrlich sortiert

1. **AllReduce** frisst bei DFlash2 auf TP2 rund ein Drittel der Zeit (x4-Links,
   kein NVLink). NCCL-Schalter sind ausgemessen: nur `NCCL_BUFFSIZE` +1 %
   (seit 13.09. in der Config). Mehr geht nur über weniger Kommunikation
   (größere Spekulationsblöcke, seltenere Reduktionen), nicht über Schalter.
2. **Turing-Attention** (Paket C): der größte offene Hebel für die RTX bei
   langem Kontext, Faktor 2 bei TTFT und 3,5 bei Decode ab 13k. Wartet auf #612.
3. **Determinismus**: die Compile-Münze macht A/B-Vergleiche über Boots hinweg
   unscharf. Cache an (seit 13.09.) friert die Wahl je Artefakt ein.
4. **PLE-Kaskade** (Flash-Next): kein Tempo, sondern Zwei-Karten-Tauglichkeit.
   Hinten angestellt.
5. **Flash-Next-Attention** läuft über QSA-Triton und GDN; die Kuanda-Quote und
   das 13k-Tempo sind Modell- und Kernelgrenzen, kein bekannter Stack-Fehler.
6. **Kein Hebel mehr**: Punkt 15 (bitgleich, 0 %), sm75-Port der Skinny-Kernel
   (0 %), FA2 mit fp8-KV auf Turing (kein Nutzen ohne fp8-Hardware) — alles
   gemessen und gestrichen.
