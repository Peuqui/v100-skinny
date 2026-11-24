# Variable Prüflänge GLM (Stufe 1): Anleitung für Kalibrier- und Adaptiv-Boot

Stand 23.11.2026. Code: Baum `1Cat-vLLM-adaptive-verify`, Zweig `glm-adaptive-verify` (aus `merge-1cat-2d218ab03`, nicht gepusht).
Spezifikation: `DESIGN.md` (Empfehlung A: der Entwurf behält den vollen Block K_max, der Scheduler wählt je Runde die Prüflänge n).
Diese Datei beschreibt nur die Änderungen am llama-swap-Eintrag; Bau und Boot macht die Hauptsitzung.

## 0. Was sich gegenüber dem heutigen Eintrag ändert

Grundlage ist der Eintrag `GLM-5.3-Flash-320B-A18B-NVFP4-DFlash2-vllm` (v14, k=4, Abfragelänge 5). Es gibt keinen neuen nativen Bau: die
Änderung ist reines Python, die `.so`-Dateien sind aus `1Cat-vLLM-merge-v14` kopiert (unversioniert, nie committen).

| Posten | heute (k=4) | Kalibrier- und Adaptiv-Boot (K_max = 6) |
|---|---|---|
| `PYTHONPATH` (zweiter Eintrag) | `/home/mp/Projekte/vllm-research/1Cat-vLLM-v14` | `/home/mp/Projekte/vllm-research/1Cat-vLLM-adaptive-verify` |
| `num_speculative_tokens` in `--speculative-config` | `4` | `6` (Entwurfsblock und K_max; Abfragelänge 7) |
| `--compilation-config` | `{"cudagraph_capture_sizes":[5],"max_cudagraph_capture_size":5}` | `{"cudagraph_capture_sizes":[7],"max_cudagraph_capture_size":7}` (K_max + 1; die Kürzungen 7 bis 2 kommen aus `tail_query_lens`, nicht aus dieser Liste) |
| `--kernel-config` | `{"triton_sm7x_tensor_cores":true}` | `{"triton_sm7x_tensor_cores":true,"adaptive_verify":{…}}` (siehe 1 und 2) |
| Cache-Verzeichnisse `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, `VLLM_CACHE_ROOT` | `…-glm-prod-v14k4-2026-11-23` | neue, leere Verzeichnisse, z. B. `…-glm-av-k6-2026-11-23` (K und Abfragelänge ändern den Compile-Hash; nicht in die Produktions-Caches schreiben) |

Alles andere (Modell, Pipeline-Aufteilung, `--max-num-seqs 1`, `--max-num-batched-tokens 1024`, Umgebungsvariablen) bleibt. `--max-num-seqs 1` ist
Voraussetzung: bei mehr als einer decodierenden Anfrage plant der Scheduler `max_tokens` (K_max) und es gibt keine Kürzungsgraphen.

Vor dem Ändern den Eintrag sichern (`config.yaml.vor-adaptive-verify-2026-11-23`); die Datei nur ändern, wenn gerade kein Modell bootet
(`--watch-config` bricht sonst den Boot ab).

Der Kalibrier- und der Adaptiv-Boot setzen `min_tokens` gleich (1), damit sie dieselben Graphen und denselben Speicher belegen (Kürzungslängen
7, 6, 5, 4, 3, 2 je Pipeline-Stufe zusätzlich zum vollen Graphen 7; n = 1 entspricht Abfragelänge 2). Der KV-Pool beider Boots ist damit
vergleichbar; er liegt nach Messung zwischen k=4 (655 360 Token) und k=7 (550 502 Token).

## 1. Kalibrier-Boots (erzwungene Prüflänge)

`forced_tokens` gilt ab Boot und ist nicht zur Laufzeit änderbar; jeder Wert ist ein eigener Boot. Der Entwurf läuft in jedem Boot mit dem vollen
Block 6, nur die Prüfung der Zielmodelle nimmt n Token.

Zusatz in `--kernel-config` (ersetzt den Wert von `adaptive_verify`, sonst nichts ändern):

```
--kernel-config '{"triton_sm7x_tensor_cores":true,"adaptive_verify":{"enabled":true,"forced_tokens":6}}'
```

`forced_tokens` und die Obergrenze `max_tokens` (Standard K_max = 6) müssen zusammenpassen; ein Wert außerhalb von `[min_tokens, max_tokens]`
bricht den Start mit einem `ValueError` ab (`adaptive_verify.forced_tokens … is outside …`).

Empfohlene Reihenfolge (Peuqui entscheidet über den Umfang; jeder Boot ist teuer):

1. **`forced_tokens` 6**: liefert die unzensierten Annahmefolgen. Aus `/metrics` je Textsorte
   `vllm:spec_decode_num_accepted_tokens_per_pos{position="0".."5"}` geteilt durch `vllm:spec_decode_num_drafts` ergibt A_1 … A_6
   (kumulativ), siehe DESIGN 5.5 Schritt 2. Zugleich Kostenpunkt t(6).
2. **`forced_tokens` 3** und **`forced_tokens` 1**: Kostenpunkte t(3), t(1). Aus den drei Punkten t0 und c der Geraden t(n) = t0 + c·n
   und damit ρ = c / t0 (DESIGN 5.5 Schritt 1). Liegen die drei Punkte nicht auf einer Geraden, zusätzlich 2, 4 und 5.
3. Kostenmessung kurz und bei langem Kontext (150k und Maximum), weil c mit dem Kontext wachsen kann (DESIGN 5.5); falls ja, ρ je
   Kontextklasse bestimmen.

Vor dem Vergleich der Texte (Test T5 der Spezifikation, Zustandsrichtigkeit auf der GPU): greedy-Texte der Boots mit `forced_tokens` n
gegen die heutige Produktion mit k = n (zuerst n = 4 gegen die Produktion). Wortgleichheit über Prozesse hinweg ist nur bei gleichem
Compile-Artefakt zu erwarten; Abweichungen händisch lesen.

Auf diese Zeilen im Boot-Log achten (je Pipeline-Stufe):

- `Adaptive verify length (forced) enabled: n in [1, 6], first round N.` (vom Scheduler; N = `forced_tokens`)
- `Capturing target tail query lengths (7, 6, 5, 4, 3, 2).` (Zielmodell-Manager; das ist die Einmal-Liste aus `tail_query_lens` mit dem vollen Graphen 7)
- Die Zeile des Entwurfs-Managers `Capturing target tail query lengths` darf **nicht** erscheinen (der Entwurf behält seine Blockbreite).
- Graph-Erfassung ohne Fehler auf allen fünf Stufen, Dauer und GiB je Stufe notieren (`Graph capturing finished`); GPU-Spitzen mit
  `~/.cache/bench-scripts/gpu_log.sh`.
- Offen laut DESIGN 6.2: ob für GLM `qsa_dense_short_context` oder `compressed_buckets` aktiv sind (`QSA short-context graph variants captured …`
  im Log); dann fehlen diese Varianten für die Kürzungslängen.

## 2. Adaptiv-Boot

Nach der Kalibrierung `verify_cost_ratio` (ρ) aus dem Ergebnis von Schritt 2 eintragen; der Standardwert 0,108 stammt aus den gelieferten
Zahlen 4,2 ms / 39 ms und ist nur ein Platzhalter.

```
--kernel-config '{"triton_sm7x_tensor_cores":true,"adaptive_verify":{"enabled":true,"min_tokens":1,"initial_tokens":4,"verify_cost_ratio":0.108,"switch_margin":0.02,"ema_alpha":0.28,"prior_conditional_acceptance":0.7,"prior_weight":2.0}}'
```

Felder (alle in `kernel_config.adaptive_verify`, keine Umgebungsvariable; `max_tokens` weglassen heißt K_max):

| Feld | Bedeutung | Standard |
|---|---|---|
| `enabled` | Scheduler wählt n je Runde (nur Methode `dflash`; bei MTP oder anderen Methoden `ValueError`) | `false` |
| `min_tokens` / `max_tokens` | Grenzen für n (`max_tokens` ≤ `num_speculative_tokens`) | `1` / K_max |
| `initial_tokens` | n der ersten Decode-Runde je Anfrage, vor jeder Rückmeldung | `max_tokens` |
| `forced_tokens` | feste Prüflänge (Kalibrierung), schaltet den Regler ab | nicht gesetzt |
| `ema_alpha` | Gewicht der neuesten Runde in der Schätzung der Annahme je Position | `0.28` |
| `prior_conditional_acceptance`, `prior_weight` | Annahme, die für noch nicht beobachtete Positionen gilt, und ihr Gewicht | `0.7`, `2.0` |
| `verify_cost_ratio` | ρ = c / t0 der Zielmodell-Prüfrunde | `0.108` |
| `switch_margin` | relativer Durchsatzgewinn, den ein neues n braucht | `0.02` |

Der Regler wählt n = argmax E(n) / (1 + ρ·n) aus der je Position geschätzten bedingten Annahme (Positionen jenseits des zuletzt geprüften n
laufen mit dessen bedingter Annahme weiter); nach einer voll angenommenen Runde prüft er sofort eine Position länger (wie 1Cats Regler).
Beim Wechsel von n gibt es kein Log je Runde; der Scheduler schreibt nur die Zeile beim Start
(`Adaptive verify length (throughput) enabled: n in [1, 6], first round 4.`). Das Histogramm von n und der Verzug der Rückmeldung (1 bis 2
Runden bei PP5) sind aus dem Log nicht ablesbar; falls die A/B sie braucht, kommt eine Protokollzeile in einem späteren Schritt (DESIGN 7.1 Punkt 7).

Abnahme und Mess-Reihen: DESIGN 7.3 (Prüfstufe 2, A/B T P T P, P = Produktion k=4; Langkontext in allen Armen).

## 3. Rücksprung

Der Produktionsbaum `1Cat-vLLM-v14` und seine Caches bleiben unberührt. Rücksprung = Eintrag aus der Sicherung
`config.yaml.vor-adaptive-verify-2026-11-23` zurückspielen. Die neuen Cache-Verzeichnisse dürfen nach der Messung gelöscht werden
(`readlink -f` vorher, Peuqui-Regel).

## 4. Hinweise zum Boot dieses Baums

- Vor dem ersten Boot die Vorab-Quellprüfung aus CLAUDE.md (`~/.cache/bench-scripts/scan_arch_gates.sh`) für die Kerne bei 2 bis 7 Zeilen auf V100
  und RTX (KDA-Conv, fusionierte GDN-Metadaten, Tail-Ring); Test T5 der Spezifikation ist Pflicht vor jedem Produktionsboot.
- Kein neuer Bau: `_C` und alle Bibliotheken sind die von `1Cat-vLLM-merge-v14`. Wird in `merge-v14` neu gebaut, die `.so` neu kopieren.
- `PYTHONPATH` darf nicht beide Bäume enthalten.
