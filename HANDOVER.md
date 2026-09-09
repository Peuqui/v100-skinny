# Übergabe

**Stand 2026-09-08 abends.** Nur der aktuelle Auftrag. Wie der Stack läuft:
`STAND.md`. Warum er so läuft: `docs/journal/`.

---

## Auftrag

Vier Turing-Defekte in 1Cat-vLLM sind gefunden und gemessen. **Nichts ist
eingereicht.** Peuqui hat gesperrt: es geht alles **gemeinsam** raus. Zerfall
UND Tempo sind Blocker — „ich gebe Erreichtes nicht so einfach auf".

Der 08.09. hat den Restabstand **von 7,0 % auf 2,0 % gedrückt** und die
Ursache benannt. Der verbliebene Rest ist lokalisiert und als **nicht
erreichbar** belegt, nicht bloß vermutet.

---

## Die vier Defekte

**1 — Turing startet nicht.**
`_resolve_gdn_prefill_backend` meldet FlashQLA für sm75. Der TileLang-Prefill
fordert 86.016 B dynamisches Shared Memory, Turing gibt 65.536. Worker stirbt
beim Engine-Init. Fix: `is_sm70_or_sm75` trennen, Rückfall auf Triton/FLA.
17 Zeilen. **Gebaut, Commit `5a26136`.**

**2 — Turing rechnet falsch.**
Das Baseline-Tor `sm70_flash_v100_baseline` in `config/vllm.py` (~Zeile 1826)
verlangt `_any_participating_device_is_capability(self, (7, 0))`, also *exakt
Volta*. Ein reines Turing-System bekommt die pre-Ampere-Abstimmung nie und
liefert Müll (Satzwiederholungen, balinesische Codepunkte). Bisektiert auf
**einen** Schalter: `VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH=1` trägt die
gesamte Korrektheit. **Noch nicht gebaut**, siehe „Offene Entscheidung".

Die Funktion stammt aus unserem eigenen PR #514, **gemerged am 07.09.** —
Fix 2 ist damit eine Folgeänderung an akzeptiertem Code.

**3 — Unfusionierter GDN-Kern im Spekulationszweig.**
Upstream ruft `fused_gdn_gating` + `fused_recurrent_gated_delta_rule`, obwohl
es `fused_sigmoid_gating_delta_rule_update` bereits importiert. GDN-Kern
83,4 → 55,0 ms. **Gebaut, Commit `5a26136`.**

**4 (NEU 08.09.) — Der Full-Forward-Wrapper kostet auf Turing 5,4 %.**

`QwenGatedDeltaNetAttention.forward` schickt unter Spekulation die **gesamte
GDN-Schicht** durch den opaken Custom-Op `torch.ops.vllm.qwen_gdn_full_forward`.
Dessen eigener Docstring:

> *„Run the full Qwen GDN attention forward **outside Inductor**. […] the
> projections around the recurrent GDN core keep the strict eager execution
> order instead of being rewritten by Inductor."*

Der Op steht in `splitting_ops`. Folge: Ein- und Ausgangsprojektion samt
gegatetem RMSNorm laufen eager statt fusioniert — **rund fünfzehn zusätzliche
Elementaroperationen pro GDN-Schicht und Schritt**.

Der Wächter (`__init__`, ~Zeile 2484) hängt an **keiner Gerätefähigkeit**,
sondern nur an `VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH` plus „Spekulation
aktiv". **Fix 2 schaltet damit auf Turing eine Volta-Notlösung scharf.**

Fix: den automatischen Arm an Volta binden. **Gebaut, uncommitted**, siehe
„Zustand der Arbeitskopien".

---

## Die Zahlen (alle 08.09., eine Sitzung, gleiche Bedingungen)

27B-NVFP4, TP2, `DEVS=0,2` (Turing-only), greedy Seed 1, 400 Token fest,
fünf Wiederholungen. Streuung unter 0,3 tok/s.

| Variante | tok/s | Ausgabe-SHA | Annahmelänge |
|---|---:|---|---:|
| **k=0, ohne Spekulation (Maßstab)** | 43,62 | `0106659946c064b1` | — |
| Upstream + Fix 1+2+3, k=3 | 69,58 | `0106659946c064b1` | 2,963 |
| dieselbe + Wrapper per Env aus | 73,34 | `0106659946c064b1` | 2,963 |
| **+ Fix 4 als echter Patch** | **73,42** | `0106659946c064b1` | 2,963 |
| sm75-Fork | 74,81 | `0106659946c064b1` | 2,963 |
| Spec-Core-Op statt Wrapper | 75,58 | `88825f5a1db681b8` | 3,109 |

**Rückstand zum Fork: 7,0 % → 2,0 %.**

**Verlustfreiheit** ist der Maßstab, nicht der Leseeindruck: k=0 liefert
`0106659946c064b1`; jede saubere k=3-Variante trifft das auf das Byte.
`ctx_scan` über zehn Prompt-Längen (19 bis 13.004 Token): saubere k=3-Route
**10/10 identisch mit k=0**, Fix 4 **10/10 identisch mit der Referenz**.

**Qualität Langkontext** (drei Fragen à 1.200 Token hinter 13.004 Token
Vorkontext, gelesen): kein Zerfall, kein Einsickern, q3 byteidentisch mit der
am 07.09. geprüften Baseline.

### Flash-Next-Abnahme (180B, TP2×PP2, heterogen) — bestanden

MTPQ-Checkpoint, k=4, `CUDA_VISIBLE_DEVICES=0,2,1,3`, alle Stufen auf Upstream
(`AIFRED_FORCE_UPSTREAM_GDN=1`), 300 Token greedy, drei Wiederholungen.

| | tok/s | Annahmelänge | Ausgabe-SHA |
|---|---:|---:|---|
| Wrapper erzwungen (= ohne Fix 4) | 57,66 | 3,030 | `864572d17f5fa8c4` |
| **Fix 4** | **58,83 / 59,06** | 3,030 | `864572d17f5fa8c4` |
| Produktivkonfig (Fork auf Turing) | 60,29 | 3,030 | `864572d17f5fa8c4` |

**+2,0 %** — weniger als die 5,4 % beim 27B, weil nur die halbe Schichtzahl auf
Turing liegt. **Alle Varianten liefern denselben Text**, Fork eingeschlossen.

**Der Pro-Gerät-Nachweis, im selben Lauf, pro Rang in eine Datei geschrieben
(`AIFRED_STATE_FILE`, weil INFO von Nebenrängen gefiltert wird):**

```
dev=0  cap=(7,5)  volta=False  auto=True  maybe=False   ← Turing: Wrapper aus
dev=1  cap=(7,5)  volta=False  auto=True  maybe=False   ← Turing: Wrapper aus
dev=2  cap=(7,0)  volta=True   auto=True  maybe=True    ← Volta:  Wrapper an
dev=3  cap=(7,0)  volta=True   auto=True  maybe=True    ← Volta:  Wrapper an
```

`auto=True` auf allen vier Rängen belegt, dass der Wrapper ohne Fix 4 überall
scharf gewesen wäre. **Fix 4 ändert am Volta-Verhalten nichts** — gemessen,
nicht argumentiert. Mit `device_id=0` hätte überall `(7,5)` gestanden und der
Patch hätte die V100-Stufen still mitentwaffnet.

---

## Was am 08.09. widerlegt wurde

**Spec-Core-Route** (`VLLM_SM70_QWEN_GDN_SPEC_CORE_OP=1`) — der schnellste
gemessene Weg (75,58), **disqualifiziert**: 0 von 10 Prompt-Längen stimmen mit
k=0 überein. Er ist nicht verlustfrei; sein Tempovorteil kommt aus der
gestiegenen Annahmequote (3,109 statt 2,963), er prüft also weniger statt
schneller zu rechnen. Textbefund zusätzlich: q1 enthält eine falsche
de-Broglie-Relation („p = h/m·v" als Wellenlänge), q3 kehrt die
Druckargumentation um (höherer Wanddruck, trotzdem Anziehung) und widerspricht
der eigenen Definition (konvex schwäche den Effekt ab); Nummerierungsfehler in
3 von 3 Antworten gegen 0 von 3 bei der Referenz. Texte:
`~/.cache/mtp-diagnostics/qual_spc_qual_k3/` gegen `qual_ff_qual_up_k3_off/`.

**`_sm70_compile_graph_slice_dim` billiger machen — zweimal gescheitert.**
Die echte Stelle ist `Qwen3_5GatedDeltaNet.forward_cuda`
(`vllm/model_executor/models/qwen3_5.py`, ~Zeile 470/472), nachgewiesen über
eine Zustandsmarke in `__init__`: `split_input=True, dflash2_split=False,
qpn8_ba=False`.

| Formulierung | Kernel | Ergebnis |
|---|---:|---|
| `arange` + `index_select` (heute) | 2 | korrekt |
| reiner View, nur für `z` | 0 | **Ausgabe zerstört, 10/10**, 54,82 tok/s |
| `slice().contiguous()` (Helfer) | 1 | **Ausgabe zerstört, 10/10**, 54,75 tok/s |

Zwei semantisch gleichwertige Umformulierungen zerstören die Ausgabe identisch
(Che-Guevara-Text, Zeitstempelketten, `</parameter></function></tool_call>`).
Die `index_select`-Form ist **tragend**. Dahinter steckt ein echter Defekt auf
diesem Pfad — ein nicht frisch materialisiertes `z` liefert Müll. Das zu
finden ist ein eigenes Projekt und ein eigener Issue wert.

---

## Der Restabstand: 2,0 %, benannt und geschlossen

Decode-Profil nach Fix 4, Gerät 0, nsys:

```
FORK          :  94.135 Kernel, 2.506,9 ms
UPSTREAM+Fix4 : 106.794 Kernel, 2.472,9 ms
Differenz     : +12.659 Kernel,   −34,0 ms
```

Die Kernelzeit ist jetzt **niedriger** als beim Fork. Was bleibt, sind
Startkosten: **+3 Kernel pro GDN-Schicht und Schritt** (20 gegen 23), alle drei
aus der Eingangsprojektion:

| | Fork | Upstream + Fix 4 |
|---|---|---|
| `z` | View, materialisiert erst im `reshape` der Norm-Fusion | `arange` + `index_select` |
| `a` | View + `.contiguous()` | `arange` + `index_select` + `.contiguous()` |
| `b` | `.contiguous()` | `.contiguous()` |
| **Summe** | **2** | **5** |

Vor Fix 4 waren es rund fünfzehn Zusatzkernel pro Schicht; die zwölf aus dem
un-fusionierten RMSNorm sind weg.

> **WIDERLEGT am 09.09.2026 — diese Kernel-Rechnung ist falsch.** Wenige Zeilen
> unter dem Split stehen `z = z.contiguous().reshape(...)`, `b = b.contiguous()`
> und `a = a.contiguous()`. Nach `index_select` ist das Ergebnis bereits
> zusammenhängend, das spätere `.contiguous()` also ein **No-op**; mit einem View
> kopiert es dort erst recht. Die Kopie verschwindet nie, sie **wandert nur** —
> und wird später sogar minimal teurer. Die Tabelle zählt die Split-Stelle und
> übersieht das nachfolgende `.contiguous()`.
>
> Gemessen (27B, TP2 auf 2× RTX 8000, k=3, greedy Seed 1, 5 Läufe je Variante,
> Referenz-SHA `0106659946c064b1`):
>
> | Variante | Median | SHA |
> |---|---:|---|
> | Fork-sm75-Pfad | 74,77 | Referenz |
> | gemeinsamer Pfad, unverändert | 73,49 | Referenz |
> | gemeinsamer Pfad, `a` als View | 73,22 | Referenz |
> | Indexvektor gecacht (kein `arange` je Schritt) | 74,20 | Referenz |
>
> Jede Sparmaßnahme war **langsamer**. `a` als View ist korrekt — damit ist die
> unten gestellte offene Frage beantwortet —, bringt aber kein Tempo.
>
> **Der Restabstand von 1,74 % sitzt NICHT in der Eingangsprojektion und ist
> wieder unlokalisiert.** Wer ihn sucht, prüft die übrigen Unterschiede des
> sm75-Pfads: eigene Kern-Op, Faltungsbehandlung, Norm-Fusion.
>
> **Korrektheitsschuld im eigenen Fork:** `qwen_gdn_attention_core_sm75`
> registriert `mutates_args=["a_or_z_out", "core_attn_out"]`, ruft aber
> `causal_conv1d_update` auf seinem qkv-Eingang — und die schreibt nachweislich
> in ihr Eingangsargument (`out = x`, so kommentiert im Quelltext). Wir
> unterdeklarieren also eine Mutation. Byteweise geht es gut, weil die Faltung
> nach `[0:qkv_size]` schreibt und `z` bei `[qkv_size:...]` liegt. Der
> gemeinsame Pfad deklariert korrekt — und **deshalb** zerstört dort ein
> `z`-View die Ausgabe: die Kopie fällt in das spätere `.contiguous()`, und das
> darf Inductor hinter die mutierende Kern-Op schieben.
>
> Details und alle Zahlen: Gedächtnisnotiz `project_z_slice_materialization_closed`.

**Erledigt am 09.09.2026:** `a` allein als View geht durch (Ausgabe korrekt,
Referenz-SHA), bringt aber kein Tempo, sondern kostet 0,4 %. Siehe den Kasten
oben — es gibt in der Eingangsprojektion nichts zu sparen.

---

## Zustand der Arbeitskopien

**PR-Branch** `sm75-gdn-prefill-route` in `~/Projekte/vllm-research/1Cat-vLLM`,
Basis `origin/main` 56f534e, Commit `5a26136` (Fix 1 + Fix 3), auf den Fork
gepusht. **Kein PR eröffnet.**

**Uncommitted im Checkout (Fix 4):**
- `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`: Helfer
  `_sm70_current_device_is_volta()` plus Bindung des automatischen Arms
- `tests/model_executor/layers/test_gdn_full_forward_device_gate.py`: 5 Tests

Geprüft: 9 Tests grün (5 neu + 4 aus Fix 1), `pre-commit` vollständig grün
(inklusive `check-torch-cuda-call`). **Gegenprobe gemacht:** mit der
Datei-Konvention `is_device_capability(70)` ohne `device_id` fällt genau der
Mischknoten-Test.

Das Prädikat fragt bewusst `torch.accelerator.current_device_index()` statt
Gerät 0 — auf einem Knoten mit Volta **und** Turing liest Gerät 0 aus jedem
Rang dieselbe Karte. Peuqui hat das am 08.09. ausdrücklich so festgelegt.

**venv** (`/home/mp/vllm/venv` → `.venv-sm70-150`): Fix 1 + Fix 3 + **Fix 4**
deployt, Fix 4-alt (SLICEDIAG) nicht. Testschalter `AIFRED_FORCE_UPSTREAM_GDN`
in beiden Modellklassen. Backups: `qwen_gdn_linear_attn.py.OHNE_FIX5`
(= dokumentierter Stand ohne Fix 4), `.OHNEFIX4`, `.MITFIX4`,
`qwen3_5.py.aifred_backup`.

**Wichtig:** Peuqui hat am 08.09. festgelegt, dass die vLLM-Installation bis
zur Lösung **nicht produktiv** ist und AIfred so lange nicht genutzt wird. Die
venv darf also im Messzustand bleiben.

---

## Offene Entscheidung: Fix 2 bauen — wie weit?

Das Tor `sm70_flash_v100_baseline` schaltet **den ganzen Baseline-Block** frei,
nicht nur `VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH`: auch
`VLLM_SM70_GDN_DECODE_FLASHQLA`, die sechs GDN/FLA-Zeitpläne und die
Multimodal-Vorgaben. Zwei Möglichkeiten:

- **eng:** nur den Compile-Graph-Schalter auf pre-Ampere ziehen. Minimal, aber
  ein Sonderweg im Code.
- **ganz:** das Tor auf pre-Ampere erweitern. Sauberer, aber Turing bekommt
  dann alle 13 Schalter. Gemessen: die anderen zwölf brachten 68,80 gegen
  68,83 tok/s, also nichts — Korrektheit damals nicht separat geprüft.

Peuqui entscheidet. Ohne Fix 2 ist der PR-Satz unvollständig, denn ein fremder
Turing-Nutzer setzt `VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH=1` nicht von Hand.

---

## Was NICHT nochmal untersucht werden muss

Zwölf Hypothesen sind gefallen, jede mit Messung:

1. FlashQLA-**Decode**-Route → abgeschaltet, Ausgabe byteidentisch kaputt
2. DFlash2-Metadaten pro Schritt → werden bei MTP nie berechnet
3. Unterschiedliche CUDA-Graph-Abdeckung → beide `FULL_AND_PIECEWISE`
4. Fehlende Fusion durch `splitting_ops` → GDN ist nur 2,7 % der Kernelzeit
5. GQA-Layout-Umsortierung → Upstream hat `gqa_interleaved_layout` auch
6. Niedrige Annahmequote als Ursache → ist Folge
7. Die sieben GDN/FLA-Rechenschalter → Korrektheit unverändert kaputt
8. `VLLM_SM70_GDN_Z_CONTIGUOUS` → Vorgabewert 0, feuert nie
9. Die anderen 12 Baseline-Schalter → kein Gewinn
10. `_sm70_compile_graph_slice_dim` als View → zerstört die Ausgabe (08.09.
    an der richtigen Stelle für `z` allein reproduziert)
11. **`fused_sigmoid_gating.py` als Ursache → ENTLASTET.** Der GDN-Kern braucht
    in beiden Varianten exakt **55,0 ms**. Die 557 Differenzzeilen zwischen den
    FLA-Bäumen spielen keine Rolle.
12. **Spec-Core-Route → nicht verlustfrei**, siehe oben.

Außerdem weiter gültig: GDN-Prefill ist kein Hebel (1,9 %); VLK-Kernel ist
langsamer als Triton; der Compile-Cache ist nicht abgeschaltet; der
Ladezeit-Hebel wäre die Platte.

**Nicht mehr behaupten:** dass die Zusatzkernel in
`prepare_gdn_attention_core_inputs` stünden (ROCm-only) oder in
`QwenGatedDeltaNetAttention.forward_cuda` (für dieses Modell tot).

---

## Wenn du hier weitermachst

1. **Fix 2 bauen**, nach Peuquis Scope-Entscheidung, und alle vier auf einen
   Branch führen.
2. ~~Flash-Next als Abnahme fahren~~ — **ERLEDIGT 08.09. abends**, bestanden.
   Siehe Abschnitt „Flash-Next-Abnahme" oben. Sonde im Repo:
   `tools/mtp-diagnostics/flashnext_ab.sh`.
3. **PR-Text schreiben.** Entwurf für Fix 1+3 liegt session-lokal und ist
   verloren; neu aufsetzen. Reihenfolge der Argumente: Fix 2 macht Turing
   überhaupt benutzbar, Fix 1 macht es bootbar, Fix 3 und 4 holen das Tempo.
4. **Eigener Issue** an 1Cat: Der SM70-Qwen3.5-GDN-Pfad hängt still davon ab,
   dass `z` frisch materialisiert wird. Ein einfacher Slice-View an derselben
   Stelle liefert Müll. Das ist ein Defekt, kein Stilproblem, und sie sollten
   ihn kennen — unabhängig von unserem PR.
5. **Duplikatsprüfung ist frisch (08.09. abends):** `origin/main` steht auf
   `e7fa44d`, 46 neue Commits seit 56f534e, alle H3-Video — keiner berührt
   `qwen_gdn_linear_attn.py`, `qwen3_5.py` oder `envs.py`. Im ganzen Repo gibt
   es **keinen** Vorgang zu Turing, sm75 oder RTX 8000.
6. **Alle neun eigenen PRs sind gemerged** (#469 #485 #511 #512 #514 #516 #518
   #528 #536), keiner offen. #455 geschlossen.

---

## Lehren, die diese Sitzung gekostet hat

Die drei vom 07./08.09. gelten weiter: eine Variable je Messung; prüfen, dass
die Messung misst, was sie soll; Modul **und** Fixstand vor jeder Zahl belegen.
Am 08.09. abends kamen zwei dazu, beide teuer:

**Ein Schalter muss beweisen, dass er feuert.** Vier Bisect-Läufe („z als View
ist unbedenklich, 10/10 byteidentisch") waren wertlos, weil der Schalter in
`QwenGatedDeltaNetAttention.forward_cuda` saß — einer Methode, die
`Qwen3_5GatedDeltaNet` überschreibt. Die Läufe maßen den unveränderten Code.
Erst eine Zustandsmarke in `__init__` hat die lebende Stelle bewiesen.

**Aus einer getracten Funktion darf man nicht loggen.**
`logger.warning_once` im `forward_cuda` bricht den Boot ab:
`torch._dynamo.exc.Unsupported: logging.Logger method not supported`. Genau
dafür gibt es `_log_runtime_route_once` mit seiner `is_compiling()`-Sperre —
die aber schweigt, wenn man sie zum Tracing-Nachweis braucht. Der einzig
brauchbare Ort für eine Diagnosemarke ist `__init__`.
