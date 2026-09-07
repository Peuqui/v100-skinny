# Übergabe

**Stand 2026-09-07 17:45.** Nur der aktuelle Auftrag. Wie der Stack läuft:
`STAND.md`. Warum er so läuft: `docs/journal/`.

---

## Erledigt in dieser Sitzung

**Der NaN-Defekt unter CUDA-Graphen ist gefunden und behoben** (18bfbb1,
gepusht auf `fork/work`).

Ursache: Der sm75-GDN-Builder ist eine eigenständige Klasse, keine Unterklasse
des Standard-`GDNAttentionMetadataBuilder`. `mamba_hybrid.py::get_extra_attn_kwargs`
verteilt die Spekulations-Metadaten per `isinstance` gegen den Standardtyp — der
sm75-Builder fiel durch. Zur Laufzeit wurde der Ziel-Forward daher als Prefill
klassifiziert, während `build_for_cudagraph_capture` (der diesen Weg umgeht) den
Graphen als Spekulations-Decode aufzeichnete. Die persistenten Spec-Puffer wurden
nie gefüllt, der Graph las sie dennoch → NaN ab dem ersten FULL-Replay.

Der V1-Runner hatte die Behandlung bereits; sie ist beim Portieren auf den
V2-Runner nicht mitgewandert. Deshalb war der Defekt V2-spezifisch — und deshalb
traf er Flash-Next, das V2 erzwingt.

Verifiziert: 27B über beide Architekturen, beide Runner, k=1 und k=3; Flash-Next
TP2×PP2 mit k=4 liefert 600 Token sauberes Deutsch. Details in `STAND.md`.

Nebenbefund mit Breitenwirkung: Der dokumentierte Flash-Next-Betriebspunkt
bootet nicht mehr (`QUANT_BACKEND`), und der MTPQ-Checkpoint ist verschwunden.

---

## Als nächstes

1. **GDN-Anteil am Prefill messen** (`tools/mtp-diagnostics/prof_prefill.sh`).
   Entscheidet, ob ein FlashQLA-Port auf Turing überhaupt einen Hebel hat — der
   End-zu-End-Vorsprung von FlashQLA beträgt gemessen nur 2,5 % (Prefill).
2. **Klären, ob der fork-eigene sm75-GDN-Backend nötig ist** (offener Punkt 5 in
   `STAND.md`). Falls nicht, entfällt eine 560-Zeilen-Kopie und die
   `isinstance`-Falle kann nicht wiederkehren.
3. **Erst danach** über einen Upstream-Beitrag entscheiden. Reihenfolge mit
   Peuqui abgesprochen: intern klären → Issue mit Messlage und Richtungsfrage →
   Zuschnitt nach deren Antwort. **Nichts senden ohne Freigabe**; AGENTS.md-
   Pflichten beachten (Duplikatsprüfung, Testkommandos samt Ergebnis im PR-Text,
   KI-Einsatz deklarieren).

## Aufräumauftrag (läuft)

Dokumentstruktur ist umgestellt: `STAND.md` (Zustand), `HANDOVER.md` (Auftrag),
`docs/journal/` (Logbücher, eingefroren), `upstream-contrib/` (Außenwirkung).
Offen sind die Korrekturen in `FLASH-NEXT-OPERATING-POINT.md` und
`upstream-contrib/README.md` — siehe `STAND.md`, offene Punkte 2 und 3.

## Nicht mehr untersuchen

- **Split-Graphen** (`VLLM_SM70_MTP_SPLIT_DRAFT_CUDAGRAPHS`) als Ursache des
  NaN: widerlegt. Der Schalter griff nachweislich, das NaN blieb. Damit ist auch
  die geteilte In-place-Mutation von `cudagraph_capture_sizes` ausgeschlossen.
- **Der 27B als „schnelles Fahrzeug"**: Er bootet in 6,5 min, nicht in 60–90 s,
  und bildet den Zielfall nicht ab (kein PP, anderer MTP-Pfad). Für Korrektheit
  taugt er, für Verifikation nicht.
