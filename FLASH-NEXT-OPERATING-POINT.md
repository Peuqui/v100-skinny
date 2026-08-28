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

## llama-swap-Einbettung (UMGESETZT 2026-08-28)

Eintrag `Qwen3.8-Flash-Next-180B-A4B-NVFP4-MTPQ-vllm` in
`~/.config/llama-swap/config.yaml`, Mitglied der exklusiven `main`-Gruppe
(Kollisionsschutz gegen die llama.cpp-Modelle). Startet den Server direkt
im Vordergrund (`python -m vllm.entrypoints.openai.api_server`) — das
Serve-Script daemonisiert und ist als llama-swap-cmd unbrauchbar.
`--served-model-name` MUSS dem Eintragsnamen entsprechen (vLLM prüft ihn,
llama.cpp nicht). Abnahme: Kohärenz 3/3, ~51 tok/s Decode (Referenz 51,9).

Fallstricke ZUSÄTZLICH zu den dreien oben, alle 2026-08-28 real aufgetreten:

4. **ninja fehlt im Service-PATH** — die llama-swap-Unit setzt einen
   Minimal-PATH; der JIT-Build der Skinny-Kernel stirbt mit
   `[Errno 2] ... 'ninja'`. Fix: `PATH=<venv>/bin:...` im env-Block
   (ninja liegt in der venv selbst).
5. **systemd-Härtung blockt Gloo und JIT-Caches** — `RestrictAddressFamilies`
   ohne AF_NETLINK lässt torch.distributed/Gloo mit „Address family not
   supported" crashen (getifaddrs braucht Netlink); `ProtectHome=read-only`
   blockt `~/.cache`/`~/.triton`/`~/.tilelang`; `ProtectSystem=strict` macht
   /tmp read-only (Inductor). Fix: Drop-in
   `/etc/systemd/system/llama-swap.service.d/llama-swap-vllm-support.conf`
   (AF_NETLINK, ReadWritePaths, PrivateTmp) + `TORCHINDUCTOR_CACHE_DIR`
   im env-Block (Cache persistent trotz PrivateTmp).
6. **vLLM-Worker überleben llama-swaps terminateProcessTree** — sie
   verlassen die Prozessgruppe (Ctrl+C-Schutz); stirbt der API-Server auf
   SIGTERM zuerst, verwaisen sie mit vollem VRAM und der nachfolgende
   llama-server scheitert am KV-Cache-Alloc. Fix:
   `cmdStop: .../scripts/vllm-swap-stop ${PID}` (AIfred-Repo) — sammelt
   die Nachfahren VOR dem Signal ein, TERM auf alle, nach 25 s KILL.
7. **AIfreds `llama-swap-build-config` löschte den Eintrag** — dessen
   `find_model_path` deutete das `-m` aus `python -m` als Modellpfad-Flag
   („GGUF fehlt" → Eintrag entfernt). Fix im AIfred-Repo: Guard
   `is_llama_server_cmd()`, fremde Backends werden weder geprunt noch
   normalisiert (Status „extern").

## Offenes Arbeitspaket AIfred

Kalibrationsroutine: vLLM-Betriebspunkte selbst finden (Peuquis Ansage —
kein Pin-Bypass als Endlösung; diese Datei ist Ground-Truth zur
Validierung: findet die Routine ~52 tok/s, ist sie gut). Suchraum: TP×PP
samt Kartenreihenfolge, k mit Sperrzonen (QSA-Blockgrößen-Arithmetik),
cudagraph_capture_sizes, Env-Schalter, PLE-Host-Offload — letzterer als
Stellschraube für VOLLEN Kontext im VRAM (Anforderung Peuqui 2026-08-28),
nicht als Festwert. vLLM-Backend-Adapter kennt PP und dieses Startmuster
weiterhin nicht; Model-Discovery und Modell-/Laufzeitpfade müssen
installationsagnostisch (konfigurierbar) werden.
