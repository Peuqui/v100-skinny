# Qwen3.8-Flash-Next: statischer Betriebspunkt (SSOT)

> **Teilweise ueberholt — gepruefte Startzeile siehe `STAND.md` (07.09.2026).**
> Die Zahlen unten gelten weiter; der Serverstart darunter bootet auf dem
> 1.5.0-Stand NICHT mehr (fehlendes `QUANT_BACKEND`), und der genannte
> Checkpoint existiert nicht mehr auf der Platte. Beides unten markiert.

Stand 2026-08-28, vermessen in der MTP-Kampagne (Details:
docs/journal/QWEN4EXP-PORT-HANDOVER.md). **Diese Datei ist die Referenz für jede
Integration (AIfred, llama-swap) und jeden künftigen Vergleich.**
AIfreds Auto-Kalibration kennt PP/PLE-Kaskade/heterogene Splits nicht —
für dieses Modell den Betriebspunkt STATISCH übernehmen, nicht kalibrieren.

## Modell

`/home/mp/models/Qwen3.8-Flash-Next-180B-A4B-NVFP4-MTPQ`
**(FEHLT seit spaetestens 07.09.2026 auf der Platte — neu erzeugen; der
provsalt-MTP-Block ist ebenfalls nicht mehr da, das Werkzeug liegt in
`~/Projekte/mtp-quant-transplant`. Ohne MTPQ keine belastbaren MTP-Zahlen.)**
(Symlink-Transplant: RadixArk-Basis + NVFP4-MTP-Block aus provsalt;
Werkzeug: github.com/Peuqui/mtp-quant-transplant. NICHT den rohen
RadixArk-Snapshot fahren — dessen BF16-Draftkopf macht MTP zum Verlust.)

## Serverstart

```bash
cd /home/mp/Projekte/v100-skinny
VLLM_SM70_E5_CACHE=0 \
CUDA_VISIBLE_DEVICES=0,2,1,4 \
TURBOMIND=1 QUANT_BACKEND=turbomind \
ENV_PREFIX=<repo>/.venv-sm70-150 \
TP=2 PP=2 K=4 GMU=0.95 MML=262144 PORT=<port> \
PP_PARTITION=24,24 PLE_HOST_GIB=6 \
EXTRA_ARGS="--compilation-config {\"cudagraph_capture_sizes\":[1,2,4,5,8]}" \
bash scripts/serve-qwen38-flash-next.sh /home/mp/models/Qwen3.8-Flash-Next-180B-A4B-NVFP4-MTPQ
```

Nicht verhandelbar und warum:
- `QUANT_BACKEND=turbomind` (seit 1.5.0, ergaenzt 07.09.2026): Der
  Skript-Default `marlin` sticht `TURBOMIND=1` bedingungslos aus
  (`envs.use_sm70_turbomind` gibt bei "marlin" sofort False zurueck). Auf den
  sm70-Stufen bricht NVFP4-MoE dann mit `NotImplementedError: ModelOpt NVFP4
  MoE on SM70 requires the TurboMind backend` ab.
- `ENV_PREFIX` setzen: der Skript-Default ist `.venv-sm70-130`, nicht 150.
- `VLLM_SM70_E5_CACHE=0` **vor Prozessstart** (Modul-Konstante!) — sonst
  Crash `_e5_apply_ints` am QSA-Ring, maskiert als Engine-Timeout. In
  llama-swap in den `env:`-Block, niemals in EXTRA_ARGS.
- `K=4` ist das einzige sinnvolle k: k=3 −33 %, k=5–8 vom QSA-Ring
  gesperrt (Blockgröße 48), k=9 Akzeptanz 0 %.
- Capture `[1,2,4,5,8]`: Größe 5 = Verifier-Batch (+5 %); alles >8 ist
  auf diesem Stack kaputt — das 27B-Schema `[k+1,2(k+1)]` NICHT
  übernehmen (halbiert den Durchsatz).
- Kartenreihenfolge: RTX-Stufe vorn (Konvention der Capability-Gates).
  **ÜBERHOLT seit 07.09.2026:** `0,2,1,4` galt, solange GPU 3 für Vigilantia/TTS
  reserviert war. Vigilantia/TTS liegt inzwischen auf der USB4-Karte GPU 4 —
  für die Modelle daher **`0,2,1,3`** verwenden (GPU 3 ist PCIe-direkt, GPU 4
  hängt am USB4-Tunnel mit drei Bridge-Hops). Siehe `STAND.md`.

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

## Topologie-Vergleich (23.09.2026, aktueller Stand)

Alle Werte mit demselben Skript (`~/.cache/bench-scripts/prefill_probe.py`),
denselben Keimen und demselben Betriebspunkt; Prompt 1800 Sätze = 29.159–29.247
Tokens (Tokenzahl je Lauf mitgeschrieben, nicht geschätzt), Präfix-Cache kalt.

| Topologie | Decode ohne Spekulation | Decode mit MTP k=4 | kalter Prefill 29k |
|---|---:|---:|---:|
| **TP2×PP2** (Produktion) | 28,9–31,2 tok/s | 32,5–40,4 tok/s | **18,6–19,1 s** |
| TP4 / PP1 (22./23.09.) | 30,1 tok/s | — (siehe unten) | 36,1 s |

**Im Decode ist TP4 gleichauf**, es liegt mitten im Streuband von TP2×PP2. Der
gesamte Nachteil steckt im Prefill: 36,1 gegen 19,1 s, weil TP4 je Schicht ein
AllReduce über vier Karten fährt, bei uns über PCIe Gen3 x4 ohne P2P durch den
Host. 1Cats Referenz (80,7 tok/s reiner Decode auf 4× V100) gilt für vier
gleiche Karten auf SXM2-Boards mit NVLink.

**KORREKTUR zu STAND 30:** dort steht „Decode 30,1 tok/s ohne MTP gegen 43–45
mit MTP ⇒ TP4 ist bei uns fast doppelt so langsam". Das vergleicht den Drafter,
nicht die Topologie — Spekulation betrifft nur den Decode. Die Aussage gilt für
den Prefill und ist für den Decode falsch. Der hier ergänzte k=0-Wert für
TP2×PP2 ist der fehlende Partner; die älteren k=0-Zahlen (31,4 aus
QWEN4EXP-PORT-HANDOVER, 32,2 in der Referenztabelle unten) stammen vom
28.08./davor, auf dem MTPQ-Checkpoint und vor `moe_qpn` — als Vergleich
untauglich.

**MTP läuft unter TP4 nicht** (STAND 30 (c)): der Drafter liegt dann auch auf
den RTX, seine FP8-Experten haben nur einen exakt-SM70-Pfad
(`qwen4_exp/nvidia/mtp.py`, `is_exact_sm70_cuda_platform()`), und der generische
Triton-`fused_moe` stirbt mit `ValueError: type fp8e4nv not supported in this
architecture`. Unser PP2-Aufbau funktioniert, weil der Drafter auf der letzten
Stufe und damit auf einer V100 landet. Selbst gelöst bliebe TP4 die schlechtere
Wahl: gleicher Decode, doppelter Prefill.

**PP4 / TP1** ist an der Speichergeometrie gescheitert, nicht am Tempo: die
PLE-Tabelle ist TP-geteilt, bei TP1 müsste ein Rang die vollen 44,7 GiB halten.
Mit der Kaskade (Store-Karte GPU 4, 26,36 GiB, Platte aus) bootet es, siehe
STAND.

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
