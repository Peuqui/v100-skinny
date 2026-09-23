# Übergabe — Stand 23.09.2026 spät

**Kein offener Bauauftrag.** Flash-Next läuft seit heute spät als PP4 mit der
PLE-Store-Stufe auf den Pipeline-Karten (STAND.md Betriebspunkt und Punkt 40).
Fork und `work-main` sind gepusht.

## Womit anfangen

`STAND.md` lesen, Punkte 38–40 sind von gestern und heute. Nicht mit den
Logbüchern anfangen.

## Was als Nächstes ansteht (Reihenfolge nach Nutzen)

1. **Produktion an einem anderen Tag nachmessen** (Punkt 38): dieselben
   12 Keime mit `prefill_probe_swap.py` gegen den Produktionseintrag, dazu
   ein AIfred-Alltagsgespräch. Weicht der Decode von 36,9 tok/s deutlich ab,
   erst Host-Speicher und Swap prüfen.
2. **PR #646 an den neuen Vertrag angleichen** (Kartenliste, Freihalte-Werte,
   spätes Laden, Kopier-Korrektur). Das ist Außenwirkung: Text zuerst
   Peuqui zeigen, AGENTS.md-Regeln beachten.
3. **llama-swap aufräumen:** `…-k0-test`, `…-prof2-test`, `…-pp4-test`,
   `…-pp4-disk-test`, `…-pp4-cards-test`, `…-pp4-cards-host3-test`. Von Hand,
   Sicherung vorher, Gruppenliste mitpflegen.

## Dauerhafte Regeln, die heute teuer waren

- **Messreihen nur mit gleichem Host-Anteil vergleichen.** GPU 4 gegen
  Pipeline-Karten sah erst nach 15 % Unterschied aus; die Hälfte davon war
  `HOST_GIB` 3 gegen 12.
- **Startzeiten nur beim ersten Boot eines Eintrags vergleichen.** Der zweite
  Boot trifft den warmen Compile-Cache (8 statt 14 min).
- **Karten bis zum Puffer füllen ist der eigentliche Test.** Solange eine
  Karte halb leer bleibt, fällt jeder Speicherfehler in der Rechnung nicht auf.
- **Parallele Sitzungen öffnen Chrome** (2,6 GB). Messskripte schreiben je
  Lauf `MemAvailable` und Swap mit — gestörte Läufe erkennen, nicht mitteln.
- **Test-Boots belegen GPU 0–3**, Varianten mit Store auf GPU 4 zusätzlich
  AIfreds Seitenkanal — vorher fragen.
