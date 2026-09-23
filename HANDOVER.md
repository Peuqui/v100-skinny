# Übergabe — Stand 23.09.2026 abends

**Auftrag für die neue Instanz: die Geräteliste der PLE-Kaskade bauen
(STAND.md Punkt 39).** Alles andere von heute ist abgeschlossen und gepusht.

## Was zu tun ist

Die Überlaufkaskade kennt genau eine Store-Karte. Deshalb schiebt Flash-Next
unter PP4 16,8 GB der PLE-Tabelle auf die SSD, obwohl auf den anderen
Pipeline-Karten zusammen rund 45 GB VRAM brachliegen. Das kostet gemessen
12 % Decode.

Reihenfolge der Stufen bleibt wie heute — **eigenes VRAM → Host (gedeckelt) →
Karten → Platte**; es fehlt nur die Mehrzahl bei der dritten Stufe. Peuqui hat
das so festgelegt, nachdem die Messung zeigte, dass der Host pro Zugriff
schneller ist als eine Nachbarkarte (3200 gegen 1766 MiB/s), der Host-Anteil
aber wegen Swap-Gefahr bei 12 GiB gedeckelt bleiben muss.

Schnitt und Fallstricke stehen in STAND.md Punkt 39, einschließlich der
kniffligen Stelle (`_remote_lookup` macht heute einen Gather über eine
Tabelle) und der Begründung, warum `plan_ple_placement` unberührt bleibt.

## Womit anfangen

`STAND.md` lesen, Punkte 37–39 sind von heute. Nicht mit den Logbüchern
anfangen (siehe Dokumentstruktur weiter unten in STAND).

## Was heute fertig wurde (nicht neu aufrollen)

- **FA2-V100-Bibliothek neu gebaut**, 42 Attention-Testfehler → 0, beide
  Produktionsmodelle bitgleich abgenommen (Punkt 35).
- **PLE-Store-Transfer 10× beschleunigt** (`MADV_SEQUENTIAL` statt
  `MADV_RANDOM` auf dem Massenpfad), Fork `6ca3e981`, PR #646 aktualisiert
  mit `93dac284` (Punkt 37).
- **PP4 schlägt die Produktion** — Prefill 13,5 gegen 18,6 s, Decode
  gleichauf bis besser, Nadeln 4/4 und 4/4 (Punkt 38). Noch KEIN
  Produktionswechsel: Wiederholung an einem anderen Tag steht aus.
- **qsa.py an 1Cat PR #664 angeglichen** (`4abea4ab`), **Testfassung aus
  #618 nachgezogen** (`5c98dac5`).
- Messfehler im Werkzeug gefunden und behoben: `dsv4_bench.py --lang` maß
  35.812 statt 18.000 Tokens (Punkt 34). Jede Ausgabezeile druckt jetzt die
  echte Promptlänge mit.

## Offene Test-Einträge in llama-swap

`…-pp4-test-vllm`, `…-pp4-disk-test-vllm`, `…-k0-test-vllm`,
`…-prof2-test-vllm`. Aufräumen, sobald die Geräteliste gemessen ist — der
pp4-disk-Eintrag ist die Vorlage für den Produktionskandidaten.

## Dauerhafte Regeln, die heute teuer waren

- **Erste Anfrage nach einem Modellstart ist wertlos** (Triton-JIT, kalte
  Graphen): 18 gegen 36 tok/s. Nie in eine Tabelle übernehmen.
- **Tokenzahl je Messung mitschreiben, nie aus der Beschriftung schließen.**
- **Test-Boots belegen GPU 4** und damit AIfreds Seitenkanal (Vision/TTS).
  Vorher fragen, sonst fällt dem Nutzer mitten im Betrieb das Bild aus.
