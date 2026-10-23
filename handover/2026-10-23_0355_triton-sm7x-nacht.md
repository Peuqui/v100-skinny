# Übergabe 23.10.2026, ~03:55 — Triton sm7x: Nachtarbeit (Kontext vor Kompression)

Ausführlich mit allen Zahlen: `v100-skinny/HANDOVER.md` (Morgen-Bericht ganz oben, darunter Nachtrag 22.10. 20:50 mit
Einträgen bis 03:45). Gedächtnis: `feedback_unify_hand_kernels_in_triton.md` (alle Regeln Peuquis),
`project_triton_sm7x_tensor_cores.md`, `project_moe_skinny_roofline_2026-10-23.md`, `project_prefill_profiles_2026-10-21.md`
(oben: Profil 23.10.), `feedback_handover_on_agent_orc_request.md`, `feedback_watcher_for_every_job.md` (Nachtrag 03:20).

## Ziel
Handkerne in eigenem Triton 3.6 mit Tensorkernen auf V100 (sm_70) und RTX 8000 (sm_75) vereinheitlichen; je Form und Karte
gewinnt der Schnellere (Handkern, cuBLAS oder Triton), nie langsamer; Programmierteil zuerst, Modell-A/B/A/B gesammelt am
Schluss (DSv4 + 27B, GLM-5.3 wenn reif). AIfred-Dienst aus bis Projektende. Befehle nur über freigegebene Muster
(`env …`, `timeout … bash skript`, `systemd-run` per `env XDG_RUNTIME_DIR=… DBUS_…`; Läufe nie hinter `&&`-Prüfketten).

## Stand Triton-Baum `/home/mp/Projekte/vllm-research/triton-sm7x`, Zweig `sm7x-tensor-cores` (lokal, sauber)
Neu heute Nacht (alle mit Volta-Prüfungen 0, Dot-Tests V100/RTX je 283 bestanden, Budget-PTX-Vergleich):
- `b5633732c9` Dekodier-Kette zwischen Load und Dot läuft im registergestuften Doppelpuffer (vor sm_80).
- `b919d9104e` reshape/trans in der Kette; `hoistConvertDotOperand` hoistet vor sm_80 reine Umordnungsketten nicht mehr
  bis zum Load (Ketten MIT Rechenarbeit weiter, sonst GLM 160 KB Smem).
- `cf50c58c44` Dekodier-BEREICH (Mehrfachergebnisse, join/split) + Nebenoperanden aus der Schleife (Gruppenskalen).
Verworfen (A/B): lineares konfliktfreies A-Layout für LDS.128 (Patch `~/.cache/bench-scripts/triton-sm7x/patches/rejected/`,
1–3 % langsamer trotz 0 Bank-Konflikten), Split-K-im-Block-MoE in Triton (1,5–2,1×), RB-2-MoE.

## Kernfamilien (Entwürfe in `~/.cache/bench-scripts/triton-sm7x/`)
- **W8A16 Block-FP8:** `w8a16_qpn8.py` liest das QPN8-Layout direkt (eine Gewichtskopie). Familie `qpn8c`: V100-Decode
  wkv 0,67, wq_a 0,80, wq_b 0,85, wo_b 0,88, w13 0,97, w2 1,14; RTX-Prefill M 512 alle 0,84–0,95; Rest gemischt. Route
  zurückgestellt (bräuchte Kacheltabelle je M/N; ~1 % Anteil).
- **MoE (MXFP4, Volta-Ordnung M8N8K4):** `moe_qpn_triton.py`, bitgleicher Fehler wie Handkern (`check_moe_qpn_triton.py`).
  V100 Triton/Hand: 512 Token 1,07–1,08, 640 1,08–1,10, 768 0,92–0,96, 1024 0,80/0,85, 2048 0,68/0,69. Kniffe: k-Reihen-
  folge „alle niedrigen, dann alle hohen Nibbles“, Byte-Paare (0,2)/(1,3) ohne PRMT, Skala in fp16 im Dekodieren, stages 1.
  Turing-Ordnung (M16N8K8) nicht gebaut (RTX-Handkern bis 512 nahe Speicherlimit).

## Wichtigster Befund (Profil 03:44, `~/.cache/bench-scripts/nsys/ds-stages-1023/`, Produktion 79e07bfef)
150k-Prefill 65,4 s. **RTX-Stufen sind der Engpass**, V100-Stufen warten 16–17 s im NCCL. RTX-MoE = 45 % der Stufe und läuft
noch mit dem ALTEN Kern (W13 6,56 / W2 3,25 ms je 512er-Chunk); der m16n8k8-MoE-Kern 039c1b6af ist im Fork, nicht in
Produktion ⇒ mit ihm RTX-MoE −34 %, DSv4-Prefill grob −15 %.

## Offene Entscheidungen für Peuqui
1. Turing-Kerne (039c1b6af u. a.) bald in Schlussmessung/Produktion — größter Hebel.
2. DSv4-Chunks 512 → 1024 (V100-Speicher): MoE je Token RTX −40 %, V100 −26 % mit Triton-MoE.
3. W8A16-Route trotz kleinem Anteil bauen oder liegen lassen.

## Nächste Schritte (wenn keine Entscheidung abgewartet werden muss)
- RTX-Restposten nach dem Turing-MoE: fp16-cuBLAS-GEMMs (~11 s/Stufe; zuordnen, welche Ops), QPN8-FP8 dicht 5,8 s
  (Triton W8A16 gewinnt bei M 512), Indexer (`_relu_weight_headsum` 2,9 s; Triton-Indexer-Route 7458c6257 auf RTX bei
  18,7k Keys 1,18× — mit aktuellem Compiler neu messen).
- Routen in den Fork (`1Cat-vLLM-fork-main`, Zweig `triton-sm7x-routes`, Muster 7458c6257), dann gesammelte A/B/A/B.
- Wiederholungen nur als A/B im selben Bench (`ab_builds.sh`); GEMM-Zahlen vor/nach 23.10. 00:00 nicht vergleichen.
