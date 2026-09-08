# Messwerkzeuge Turing/MTP

Sonden für den Vergleich **sm75-Fork gegen Upstream-GDN + Fixes** auf den
RTX 8000. Warum es diesen Vergleich gibt: `../../HANDOVER.md`.

---

## ⚠️ Drei Fallen, die am 08.09.2026 reale Zahlen verfälscht haben

Jede hat einen halben Tag Messzeit gekostet. Vor **jeder** genannten Zahl prüfen:

### 1. `VLLM_SKINNY_SM75_GDN` existiert nicht

Der Schalter wird in der venv **nirgends gelesen**. Die Modulwahl steht hart im
Code, in beiden Modellklassen, jeweils an `get_device_capability() == (7, 5)`:

* `vllm/model_executor/models/qwen3_5.py` (~Zeile 534) — 27B
* `vllm/models/qwen4_exp/nvidia/model.py` (~Zeile 253) — Flash-Next

Sonden, die über diese Variable umschalten wollten, liefen **alle am Fork** und
verglichen ihn mit sich selbst. Umschalten geht nur über einen echten Schalter;
in der venv ist dafür `AIFRED_FORCE_UPSTREAM_GDN=1` eingebaut (venv-lokal,
Backups `*.aifred_backup`). Der Patch fügt an beiden Stellen eine Zeile ein:

```python
if (
    os.environ.get("AIFRED_FORCE_UPSTREAM_GDN") != "1"
    and torch.cuda.is_initialized()
    and torch.cuda.get_device_capability(torch.cuda.current_device()) == (7, 5)
):
```

### 2. Modulnachweis aus dem Boot-Log — nicht aus der Absicht

```bash
grep -c 'cannot run on Turing'      $W/boot.log   # >0 → Upstream-Modul (Fix-1-Warnung)
grep -c 'qwen_gdn_linear_attn_sm75' $W/boot.log   # >0 → Fork-Modul
```

Genau eines von beiden muss anschlagen.

### 3. venv und Checkout haben verschiedene Basen

Die venv ist das **1.5.0-Wheel plus Overlay** (~7.647 Zeilen in
`qwen_gdn_linear_attn.py`), der Checkout `~/Projekte/vllm-research/1Cat-vLLM` ist
**`origin/main` plus Fixes** (~7.642). Ein Fix im Checkout ist **nicht** in der
venv. Fix 3 fehlte dort einen halben Tag, wodurch Messungen als „Fix 1+2+3"
galten, die Fix 1+2 waren. Also zusätzlich am geladenen File prüfen:

```bash
G=/home/mp/vllm/venv/lib/python3.12/site-packages/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
grep -c 'cannot run on Turing'   $G   # Fix 1
grep -c 'FIX3: One fused launch' $G   # Fix 3
grep -c SLICEDIAG                $G   # Fix 4 (WIDERLEGT, darf nicht 1 sein)
```

Fix 2 ist kein Codepatch in dieser Datei, sondern das Baseline-Tor in
`config/vllm.py`; in den Sonden wird er durch
`VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH=1` nachgebildet.

---

## Die Sonden

Alle rufen sich `<name> <fork|upstream> <k>` und legen ihre Ergebnisse unter
`~/.cache/mtp-diagnostics/qual_<name>/` ab (`boot.log`, `result.json`,
`text_*.txt`). Sie brechen ab, wenn ein `api_server` läuft oder die Zielkarten
belegt sind. Gerätesatz über `DEVS` (Vorgabe `0,2` = die beiden RTX 8000).

| Skript | Zweck |
|---|---|
| `speed_27b.sh` | Tempo, kurzer Prompt, 400 Token fest, 5 Wiederholungen |
| `qual_longctx.sh` | Qualität: 3 Fragen à 30 Sätze hinter 13.004 Token Vorkontext |
| `ctx_scan.sh` | dieselbe Frage über 10 Prompt-Längen von 19 bis 13.004 Token |
| `venv_fix_toggle.sh` | `on\|off` — schaltet Fix 4 in der venv um (nur für Gegenproben) |
| `mk_vorkontext.py` | erzeugt `vorkontext.txt` deterministisch neu |
| `probe.sh`, `matrix.sh`, `bench_gdn.sh`, `prof_prefill.sh` | älter, siehe Journal |

### Messmethode, die sich bewährt hat

Greedy (`temperature 0`, fester Seed), **feste** `max_tokens`. Damit liegt die
Streuung unter 0,2 tok/s. Mit Temperatur 0,7 und freier Länge tarnte dieselbe
Messung einen realen 57-%-Unterschied als Rauschen.

**Grenze der Methode:** Bei 13k Kontext ist k>0 **nicht bit-reproduzierbar** —
derselbe Fork über zwei Boots lieferte q3 einmal `7250beea`, einmal `24f8b789`
(Annahmequote 0,5299 gegen 0,5265). Bei kurzem Prompt war k=3 über fünf Läufe
bitgleich. Hash-Gleichheit ist bei langem Kontext also **kein** Kriterium; dort
zählt das Lesen der Texte.

### Der Vorkontext

170 Absätze eines erfundenen Hafenlogistik-Sachberichts, ~12.900 Token,
deterministisch aus Bausteinen kombiniert (nicht-repetitiv, damit der Prefill
echte Arbeit hat) und **thematisch unabhängig** von den Prüffragen. Der Bericht
ist als „Hintergrundmaterial … nicht Teil der Aufgabenstellung" ausgewiesen —
sickern seine Begriffe in die Antwort, ist das ein Befund.

Prüffragen: Quantenphysik, Regenbogeneffekt, **Coandă-Effekt**. Letzterer ist die
Fangfrage — er zeigt Halluzinationen zuverlässig.

### Worauf beim Lesen zu achten ist

Zerfall zeigt sich als Tag-Kaskade (`</parameter></function></tool_call>` in
Endlosschleife), als Zahlen- oder Nullkette, als Sprachwechsel, als falsches
Thema oder als Abbruch nach wenigen Token (`finish_reason: stop` bei kleiner
`completion_tokens`). Regex taugt zum Auffinden, nicht zum Urteil — die Texte
werden gelesen.
