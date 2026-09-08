# Übergabe

**Stand 2026-09-08 nachmittags.** Nur der aktuelle Auftrag. Wie der Stack läuft:
`STAND.md`. Warum er so läuft: `docs/journal/`.

---

## Auftrag

Drei Upstream-Defekte für Turing sind gefunden, gefixt und gemessen. **Nichts
ist eingereicht.** Peuqui hat gesperrt: es geht alles **gemeinsam** raus, und
erst wenn der Restabstand geklärt ist. Kein Teilbeitrag, keine Nachbesserung
später. Am 08.09. nachmittags bestätigt: **Zerfall UND Tempo sind Blocker** —
„ich gebe Erreichtes nicht so einfach auf". 3 % Rückstand sind keine
Verhandlungsmasse.

Der Nachmittag des 08.09. hat den Restabstand **eingegrenzt statt gelöst** und
einen vierten Fixversuch **widerlegt**. Siehe den Abschnitt „Nachmittag 08.09.".

---

## Die drei Defekte (alle gemessen, alle upstream)

**1 — Turing startet nicht.**
`_resolve_gdn_prefill_backend` in `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`
meldet FlashQLA für sm75 (`capability.minor in (0, 5)`). Der voreingestellte
TileLang-Prefill fordert 86.016 B dynamisches Shared Memory, Turing gibt 65.536.

```
RuntimeError: Worker failed with error
  'Failed to set the allowed dynamic shared memory size to 86016'
→ Worker proc VllmWorker-3 died, Engine core initialization failed
```

Fix: `is_sm70_or_sm75` in `is_sm70`/`is_sm75` trennen, Rückfall auf Triton/FLA,
mit begründeter `logger.warning_once`. 17 Zeilen.

**2 — Turing rechnet falsch.**
Die pre-Ampere-Grundabstimmung in `config/vllm.py` hängt an
`_any_visible_device_has_capability((7, 0))` — verlangt also eine *Volta*-Karte.
Ein reines Turing-System läuft unkonfiguriert und liefert Müll:

```
Frage 1:  "Er is een nieuwe versie van de app beschikbaar."  ×30
Frage 2:  ᭡᭢᭣᭤᭥᭦᭧᭨᭩᭪᭬…
```

Bisektiert von 13 Schaltern auf einen: `VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH=1`
allein trägt die gesamte Korrektheit. Fix: Tor auf pre-Ampere ausweiten, 1 Zeile.

Der Kommentar über der Stelle stammt aus unserem eigenen PR #514 — wir haben
damals das Prüfen aller Karten eingeführt, aber die Fähigkeitsstufe auf Volta
stehen lassen.

**Warum Flash-Next nie betroffen war:** Es fährt `CUDA_VISIBLE_DEVICES=0,2,1,3`,
also RTX **und** V100. Eine Volta ist sichtbar, die Abstimmung greift, die
Turing-Stufe erbt sie. Nur ein Turing-only-Aufbau (`0,2`) fällt durch.

**3 — Unfusionierter GDN-Kern im Spekulationszweig.**
Upstream ruft `fused_gdn_gating` + `fused_recurrent_gated_delta_rule` (zwei
Kernel), obwohl es `fused_sigmoid_gating_delta_rule_update` bereits importiert
und im DFlash2-Zweig direkt darüber nutzt. Fix: die fusionierte Variante auch
dort. Wirkung: GDN-Kern 83,4 → 55,0 ms, exakt der Fork-Wert.

---

## Die Zahlen

Greedy (`temperature 0`, fester Seed), 400 Token fest, fünf Wiederholungen.
Streuung dadurch unter 0,2 tok/s — mit Temperatur 0,7 und freier Länge hatte
dieselbe Messung einen realen 57-%-Unterschied als Rauschen getarnt.

| 27B, TP2, k=3 | Median | Spanne |
|---|---:|---:|
| Fork-sm75 | 74,33 | 74,30–74,34 |
| Upstream + Fix 1+2 | 68,80 | 68,75–68,89 |
| Upstream + Fix 1+2+3 | **69,71** | 69,69–69,87 (am 08.09. mit 69,64 bestaetigt) |

| Flash-Next 180B, TP2×PP2, k=4 | Median | Spanne |
|---|---:|---:|
| Fork-sm75 | 54,10 | 53,89–54,23 |
| Upstream + Fix 1+2 | 47,10 | 46,96–47,70 |
| Upstream + Fix 1+2+3 | **48,91** | 48,66–48,99 |

**Korrektheit:** mit allen drei Fixes 3 von 3 Hashes byteidentisch mit der
Referenz (q1 `b63fa12008eb4206`, q2 `b5e1d196bf4e3811`, q3 `9c394f0eaae8985b`).
Die Referenz ist doppelt bestätigt: V100 + Upstream **und** RTX + Fork liefern
sie identisch.

**Annahmelänge** in allen 27B-Läufen bitgleich **2,963** (Quote 0,6543). Der
Entwurfspfad ist damit nicht beteiligt — der `lm_head`-Verdacht ist ausgeschlossen.

---

## Nachmittag 08.09.: eingegrenzt, ein Fixversuch widerlegt

### Der Abstand entsteht ausschließlich mit Spekulation

| 27B | Fork | Upstream+3 | Abstand |
|---|---:|---:|---:|
| k=0, kurzer Prompt | 43,5 | 43,5 | **keiner** |
| k=0, 13k Kontext | 24,99 | 25,02 | **keiner** |
| k=3, kurzer Prompt | 74,33 | 69,64 | 6,3 % |
| k=3, 13k Kontext | 29,59 | 28,82 | 2,6 % |

Ohne MTP sind beide Varianten in beiden Kontextlängen **exakt gleich schnell**.
Damit sind der normale Decode-Pfad, der Prefill und die Attention als Ursache
ausgeschlossen. Es bleibt der Spekulationszweig.

Fix 3 bringt +1,25 % (68,78 → 69,64 bei kurzem Prompt, beide Werte doppelt
gemessen und deckungsgleich mit den Nachtwerten 68,80 / 69,71).

Der Rückstand skaliert mit der Modellgröße: 27B 6,2 %, Flash-Next 9,6 %. Das
stützt „Overhead pro GDN-Schicht und Schritt" und macht weitere Flash-Next-Boots
für die *Ursachensuche* wertlos — sie gehören als Abnahme vor den PR, nicht davor.

### Fix 4 (`_sm70_compile_graph_slice_dim` als View) ist WIDERLEGT

`_sm70_compile_graph_slice_dim` (Zeile 1351) materialisiert bei `start != 0` per
`torch.arange` + `index_select`, wo der Fork `split`/`chunk` nutzt — reine Views.
Die Kernel-Zahlen rechnen sich exakt darauf auf: 6537 `_scatter_gather_elementwise`
= 2 × 3220 Decode-Schritte (die zwei Aufrufe in `prepare_gdn_attention_core_inputs`,
Zeilen 2782/2787), und 6741 − 205 = 6536 `elementwise_kernel_with_index` = die
zugehörigen `arange`-Kernel. Eine dokumentierte Begründung existiert nicht; der
Helper kam mit `6ada86e`, einem Massen-Import.

**Trotzdem ist die Materialisierung notwendig.** Auf View umgestellt zerfällt die
Ausgabe bei k=0 vollständig — geprüft über zehn Prompt-Längen von 19 bis 13.004
Token, zehn von zehn kaputt (Tag-Kaskaden `</parameter></function></tool_call>`,
Zahlenketten, falsches Thema). Ohne den Patch: zehn von zehn sauber. Der Gewinn
wäre 1,1 % gewesen (68,78 → 69,55 auf dem Stand Fix 1+2, in beiden
Reihenfolgen gemessen; mit Fix 3 zusammen nie gemessen, weil widerlegt).

Bei k=3 fiel es nicht auf, weil unter Spekulation `auto_sm70_qwen_gdn_full_forward`
(Zeile 2485) eine andere Forward-Route wählt. **Nicht wieder vorschlagen.**

### Qualitätsmatrix 27B: Upstream + Fix 1+2+3 bestanden

Vier Zellen (Fork/Upstream × k=0/k=3), drei Fragen à 30 Sätze, je 13.004 Token
deterministischer Vorkontext (thematisch unabhängiger Hafenlogistik-Bericht,
`scratchpad/vorkontext.txt`, 170 Absätze). Alle Texte händisch gelesen.

Kein Zerfall, keine Schleifen, kein Sprachwechsel, kein Einsickern des
Vorkontexts, volle Satzzahl, fachlich korrekt. Bei k=3 sind q1 und q2
byteidentisch mit dem Fork, bei k=0 q1. Die Abweichungen sind
Formulierungsvarianten, beide sachlich richtig.

Nachgezogen mit dem vollen Prüfling (Fix 1+2+3, `f3_qual_k0`/`f3_qual_k3`):
**alle sechs Ausgaben byteidentisch mit Texten, die schon gelesen und als
korrekt befunden waren** — drei davon sind Fork-Ausgaben. Upstream+3 verlässt
den Variantenraum des Forks also nicht.

**k=3 ist bei 13k Kontext nicht bit-reproduzierbar** — derselbe Fork, zwei Boots:
q3 einmal `7250beea`, einmal `24f8b789`, Annahmequote 0,5299 gegen 0,5265. Bei
kurzem Prompt war k=3 über fünf Läufe bitgleich. Hash-Gleichheit ist bei langem
Kontext also kein taugliches Kriterium mehr.

### Zwei Messfehler, die Ergebnisse verfälscht hatten

**`VLLM_SKINNY_SM75_GDN` existiert in der venv nicht.** Die Modulwahl steht hart
im Code (`qwen3_5.py:534`, `models/qwen4_exp/nvidia/model.py:253`, beide
`get_device_capability() == (7,5)`). Sondenskripte, die über diese Variable
umschalten wollten, liefen **alle am Fork**. Es gibt jetzt einen echten Schalter
`AIFRED_FORCE_UPSTREAM_GDN=1` in beiden Modellklassen (venv-lokal, Backups
`*.aifred_backup`).

**venv und Checkout haben verschiedene Basen.** Die venv ist das 1.5.0-Wheel
plus Overlay (7.647 Zeilen), der Checkout ist `origin/main` plus Fixes (7.642).
Ein Fix im Checkout ist NICHT in der venv. Fix 3 fehlte dort den halben Tag,
wodurch Messungen als „Fix 1+2+3" gelten sollten, die Fix 1+2 waren.

**Pflicht ab jetzt:** vor jeder genannten Zahl beides prüfen —
*welches Modul* (`grep -c 'cannot run on Turing'` im Boot-Log, nur im
Upstream-Modul vorhanden, gegen `grep -c 'qwen_gdn_linear_attn_sm75'`) **und**
*welche Fixes* (Marker `FIX3: One fused launch` bzw. `SLICEDIAG` in der
geladenen venv-Datei). Der Modulnachweis allein genügt nicht.

### Warum die zwei Varianten überhaupt verschieden rechnen

Der Fork zieht seine FLA-Kernel aus `vllm/third_party/flash_linear_attention/ops`
(unverändertes vLLM-Original, denn die 1.781 Zeilen sind eine 0.27.1-Kopie, siehe
`docs/journal/MERGE-PROJECT-HANDOVER.md:295`), das Upstream-Modul aus
`vllm/model_executor/layers/fla/ops` (1Cat-weiterentwickelt). Von 18 Dateien
differieren acht, und zwar genau die Rechenkerne:

| Datei | abweichende Zeilen | Rolle |
|---|---:|---|
| `fused_sigmoid_gating.py` | 557 (279 → 780) | Fix-3-Kernel, **Spekulation** |
| `fused_recurrent.py` | 198 | Decode |
| `kda.py` | 150 | GDN-Kern |
| `chunk_scaled_dot_kkt.py` | 71 | Prefill |
| `chunk_o.py` | 63 | Prefill |
| `chunk_delta_h.py` | 62 | Prefill |
| `layernorm_guard.py` | 40 | Norm |
| `fused_gdn_prefill_post_conv.py` | 36 | Prefill |
| übrige 8 (chunk, cumsum, index, l2norm, op, solve_tril, utils, wy_fast) | 0 | identisch |

Ein vollständiger Baumtausch scheitert an zwei Symbolen, die es im Original nicht
gibt: `fused_sigmoid_gating_delta_rule_update_mixed_qkv` und `..._out`. Die
Signatur von `fused_sigmoid_gating_delta_rule_update` ist ansonsten deckungsgleich
(nur `ddtree_parent_ids` ist 1Cat-neu, DFlash2-Pfad, den wir nicht fahren).

**Da der Abstand nur mit Spekulation auftritt, ist `fused_sigmoid_gating.py` der
Hauptverdächtige** — größte Differenz und genau der Kernel aus Fix 3.

---

## Die offene Frage: woher die restlichen 6,4 % (nur mit Spekulation)

Lokalisiert bis auf Kernel-Ebene, Ursache **nicht** bestimmt. nsys, 27B,
Gerät 0, gleiche Schrittzahl (3.220 gegen 3.225 `_causal_conv1d_update`):

| Kernel | Fork ms / n | Upstream+Fix3 ms / n |
|---|---|---|
| `elementwise_kernel` | 33,2 / 10.402 | 59,2 / 20.395 |
| `vectorized_elementwise` | 9,2 / 3.175 | 49,5 / 26.663 |
| `unrolled_elementwise` | 7,1 / 783 | 34,9 / 10.995 |
| `_scatter_gather_elementwise` | 0,0 / **1** | 30,3 / **6.537** |
| `reduce_kernel` | 2,2 / 205 | 19,7 / 3.473 |
| `elementwise_kernel_with_index` | 1,9 / 205 | 10,9 / 6.741 |
| **Summe Familie** | **71,8** | **219,8** |

Gesamt-Kernelzeit 2.533 gegen 2.656 ms — die Familie erklärt die Differenz
vollständig. Es ist Index- und Verwaltungsarbeit pro Schicht und Schritt, kein
Rechenkernel.

**Verdacht, ungeprüft:** `prepare_gdn_attention_core_inputs` und
`rearrange_mixed_qkv` im Fork machen das Slicing anders. Der nächste Schritt
wäre, diese beiden gegen Upstreams Entsprechungen zu diffen und die
Index-Operationen zu zählen — nicht raten, zählen.

---

## Die offene Entscheidung (Peuqui, 08.09., Frage abgebrochen)

> *„Was ist, wenn wir unseren Code statt des Upstream einfügen?"*

Also: unsere Implementierung upstream bringen, statt ihre zu patchen. Die
Frage ist unbeantwortet. Was dafür und dagegen spricht:

**Dafür:** Unsere Fassung ist nachweislich korrekt (byteidentisch mit der
Volta-Referenz) und 6–11 % schneller. Wir müssten den Restabstand nicht erst
erklären — wir würden die schnellere Variante liefern.

**Dagegen:** Es wäre ein 571 + 1.781 Zeilen umfassender Parallelbaum neben
7.633 Zeilen bestehendem Code. Die erste Rückfrage im Review wäre „warum nicht
den bestehenden reparieren?" — und darauf haben wir keine Antwort, solange die
6–11 % nicht erklärt sind. Zudem widerspricht es unserer eigenen SSOT-Regel.

**Mittelweg, der zu prüfen wäre:** Nur den Teil einreichen, der den Abstand
erzeugt — sobald bekannt ist, welcher es ist. Dann wären es drei kleine Fixes
plus eine begründete Optimierung, alle an ihrer richtigen Stelle in Upstreams
Code. Das ist der Weg, der zu #469 und #514 passt, die beide gemergt wurden.

---

## Was NICHT nochmal untersucht werden muss

Zehn Hypothesen sind gefallen, jede mit Messung:

1. FlashQLA-**Decode**-Route → abgeschaltet, Ausgabe byteidentisch kaputt
2. DFlash2-Metadaten pro Schritt → werden bei MTP nie berechnet
   (`uses_dflash_selector_engine` ist falsch)
3. Unterschiedliche CUDA-Graph-Abdeckung → beide `FULL_AND_PIECEWISE`,
   gleiche Capture-Größen, gleicher Speicher
4. Fehlende Fusion durch `splitting_ops` → GDN ist nur 2,7 % der Kernelzeit
5. GQA-Layout-Umsortierung → Upstream hat `gqa_interleaved_layout` auch
6. Niedrige Annahmequote als Ursache → ist Folge; identisch, sobald korrekt
7. Die sieben GDN/FLA-Rechenschalter → Korrektheit unverändert kaputt
8. `VLLM_SM70_GDN_Z_CONTIGUOUS` → Vorgabewert 0, feuert nie
9. Die anderen 12 Baseline-Schalter → 68,80 gegen 68,83 tok/s, kein Gewinn
10. **`_sm70_compile_graph_slice_dim` als View (Fix 4)** → 1,1 % Gewinn,
    aber die Ausgabe zerfällt bei k=0 über alle zehn geprüften Prompt-Längen.
    Die Materialisierung ist notwendig, ihr fehlt nur der Kommentar.
    Details im Abschnitt „Nachmittag 08.09.".

Außerdem geklärt und abgelegt:

- **GDN-Prefill ist kein Hebel:** 4,91 s von 262 s bei 131k Tokens = 1,9 %.
  Ein TileLang-Port für Turing lohnt nicht. Der Hebel liegt bei QSA/MoE —
  dort brachte #469 Faktor 4,2 auf der RTX.
- **VLK-Kernel ist keine Alternative:** läuft auf Turing, ist aber ab 2.048
  Tokens 15–21 % langsamer als Triton.
- **Der Compile-Cache ist nicht abgeschaltet.** 14 GB auf Platte, frische
  Einträge pro Boot. Die 45 s sind Cache-Fehltreffer durch eigene
  Konfigurationsänderungen — jede neue Schalterkombination ist ein neuer
  Schlüssel.
- **Ladezeit-Aufteilung** (262k Kontext, Flash-Next): 254 s Gewichte von der
  USB-NVMe, 231 s Engine-Init, 45 s Compile, 13 s Graph-Capture. Der Hebel
  wäre die Platte, nicht die Software.

---

## Zustand der Arbeitskopien

**PR-Branch** `sm75-gdn-prefill-route` in `~/Projekte/vllm-research/1Cat-vLLM`,
auf `origin/main` (56f534e, unverändert). 25 Zeilen rein / 16 raus in
`qwen_gdn_linear_attn.py` — enthält Fix 1 **und** Fix 3. Fix 2 (die Zeile in
`config/vllm.py`) ist **noch nicht gebaut**, nur belegt.

Geprüft: `tests/model_executor/layers/test_gdn_prefill_backend_resolve.py`
4 passed, Gegenprobe bestanden (ohne Fix fallen genau die zwei Turing-Tests);
89 bestehende Tests der berührten Module grün; `pre-commit` ohne Fehlschlag
(ruff, format, typos, mypy, SPDX). PR-Text entworfen in
`scratchpad/PR-A-BODY.md` — die Ergebniszahl darin ist noch die des Forks und
muss auf die des Fixes korrigiert werden.

**Overlay** `fork_patches_150/qwen_gdn_linear_attn.py` trägt Fix 1 (nicht
Fix 3 — der ist für uns wirkungslos, weil unser sm75-Modul den Pfad nicht
nimmt). Die venv ist damit identisch. Uncommitted außerdem `STAND.md` und
`scripts/serve-qwen38-flash-next.sh`.

**Produktivsystem** vollständig auf Fork-Stand: kein Testschalter in
`models/qwen4_exp/nvidia/model.py` oder `model_executor/models/qwen3_5.py`,
llama-swap-Config bei 23 Modellen ohne Testschalter, Flash-Next-Einträge auf
`--max-model-len 262144`. Backups: `config.yaml.bak-2026-09-07-vor-max-ctx`.

**Messwerkzeuge** in `scratchpad/` (session-lokal, bei Bedarf nach
`tools/mtp-diagnostics/` übernehmen): `qual_probe.sh` (Korrektheit, drei Fragen
je Boot, SHA-256), `speed27_probe.sh` (deterministisch, fünf Wiederholungen,
Annahmequote), `prof_decode.sh` (nsys-Kernel-Summen), `mk_variant.sh`
(Schalter-Bisektion). Der `nsys` im System ist 2022.4.2 — `--output` und
`--force-overwrite` gehören dort an `nsys start`, nicht an `launch`.

---

## Wenn du hier weitermachst

1. **Erst die 6–11 % klären.** `prepare_gdn_attention_core_inputs` und
   `rearrange_mixed_qkv` gegen Upstream diffen, Index-Operationen zählen.
   Maßstab ist der byteidentische Referenzhash — jede Änderung muss ihn treffen.
2. **Dann Fix 2 bauen** (`config/vllm.py`, pre-Ampere statt Volta) und die
   drei Fixes zu einem Branch zusammenführen.
3. **Dann Peuqui den Diff vorlegen.** AGENTS.md verbietet reine Agenten-PRs;
   er muss jede geänderte Zeile gelesen haben und verteidigen können.
4. **Duplikatsprüfung wiederholen**, sie ist von 07.09. und `main` bewegt sich
   täglich.

Und die Lehre, die diese Sitzung gekostet hat: Neun Vermutungen sind gefallen,
weil ich von einer Beobachtung auf eine Ursache geschlossen habe, statt sie zu
isolieren. Was jedes Mal geholfen hat, war eine Messung, die genau **eine**
Variable ändert — der V100-Gegenversuch (gleicher Code, andere Architektur)
hat mehr gebracht als alles Lesen davor.
