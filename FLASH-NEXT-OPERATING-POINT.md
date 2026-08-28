# Qwen3.8-Flash-Next: statischer Betriebspunkt (SSOT)

Stand 2026-08-28, vermessen in der MTP-Kampagne (Details:
QWEN4EXP-PORT-HANDOVER.md). **Diese Datei ist die Referenz für jede
Integration (AIfred, llama-swap) und jeden künftigen Vergleich.**
AIfreds Auto-Kalibration kennt PP/PLE-Kaskade/heterogene Splits nicht —
für dieses Modell den Betriebspunkt STATISCH übernehmen, nicht kalibrieren.

## Modell

`/home/mp/models/Qwen3.8-Flash-Next-180B-A4B-NVFP4-MTPQ`
(Symlink-Transplant: RadixArk-Basis + NVFP4-MTP-Block aus provsalt;
Werkzeug: github.com/Peuqui/mtp-quant-transplant. NICHT den rohen
RadixArk-Snapshot fahren — dessen BF16-Draftkopf macht MTP zum Verlust.)

## Serverstart

```bash
cd /home/mp/Projekte/v100-skinny
VLLM_SM70_E5_CACHE=0 \
CUDA_VISIBLE_DEVICES=0,2,1,4 \
TP=2 PP=2 K=4 GMU=0.95 MML=16384 PORT=<port> \
PP_PARTITION=24,24 PLE_HOST_GIB=6 \
EXTRA_ARGS="--compilation-config {\"cudagraph_capture_sizes\":[1,2,4,5,8]}" \
bash scripts/serve-qwen38-flash-next.sh /home/mp/models/Qwen3.8-Flash-Next-180B-A4B-NVFP4-MTPQ
```

Nicht verhandelbar und warum:
- `VLLM_SM70_E5_CACHE=0` **vor Prozessstart** (Modul-Konstante!) — sonst
  Crash `_e5_apply_ints` am QSA-Ring, maskiert als Engine-Timeout. In
  llama-swap in den `env:`-Block, niemals in EXTRA_ARGS.
- `K=4` ist das einzige sinnvolle k: k=3 −33 %, k=5–8 vom QSA-Ring
  gesperrt (Blockgröße 48), k=9 Akzeptanz 0 %.
- Capture `[1,2,4,5,8]`: Größe 5 = Verifier-Batch (+5 %); alles >8 ist
  auf diesem Stack kaputt — das 27B-Schema `[k+1,2(k+1)]` NICHT
  übernehmen (halbiert den Durchsatz).
- Kartenreihenfolge `0,2,1,4`: RTX-Stufe vorn (Konvention der
  Capability-Gates), GPU 3 bleibt frei für Vigilantia/TTS.

Boot-Dauer ~7 min (llama-swap: healthCheckTimeout beachten, langer TTL;
schnelle Swaps sind mit dieser Modellklasse ohnehin nicht sinnvoll).

## Referenzwerte (Abnahme-Kriterien nach jeder Änderung)

| Messung | Soll |
|---|---|
| bench.py, schwerer Prompt, 200 tok, n=3 | **51,9 tok/s** (±2) |
| vorhersagbarer Prompt | 68,2 tok/s |
| k=0-Kontrolle (gleicher Betriebspunkt) | 32,2 tok/s |
| Akzeptanz / Länge (normaler Text) | ~70 % / ~3,8–3,9 |
| health_probe.py Prefill | ≥ −0,5 |
| Kohärenz | 8/8 |

Alternativer Betriebspunkt für vollen Kontext (MML 262144, k=0, ohne
MTP): vertauschte Anordnung `1,4,0,2`, Split 6/42 → 34,0 tok/s.
MTP und Vertausch schließen sich derzeit aus (Handover, „Abend II").

## Offenes Arbeitspaket AIfred

Kalibrationsroutine um einen Bypass erweitern: Modelle mit statischem
Betriebspunkt (diese Datei) werden nicht kalibriert, sondern übernehmen
die Konfiguration 1:1; Kalibration nur bei Hardware-/Modellwechsel neu
anfassen. vLLM-Backend-Adapter kennt PP und dieses Startmuster noch nicht.
