"""Deterministischer Vorkontext: Sachbericht Hafenlogistik, nicht-repetitiv,
thematisch unabhaengig von den drei Prueffragen. Ohne Zufall, ohne Schleifen-
wiederholung — jeder Absatz kombiniert andere Bausteine."""
import pathlib
TERMINALS = ["Altenwerder", "Burchardkai", "Tollerort", "Eurogate", "Steinwerder",
             "Waltershof", "Predoehlkai", "Kroetenberg", "Sandtorhafen", "Grasbrook"]
KRANE = ["Portalkran", "Doppelausleger", "Tandemspreader", "Schwerlastportal",
         "Halbportalkran", "Verladebruecke", "Reachstacker", "Portalhubwagen"]
WARE = ["Kuehlcontainer", "Gefahrgut der Klasse 3", "Stueckgut", "Schwergut",
        "Massengut", "Projektladung", "Leercontainer", "Tankcontainer"]
BEFUND = [
    "Die Umschlagleistung lag {a} Prozent ueber dem Vorjahreswert, wobei die "
    "Wartezeit an der Landseite um {b} Minuten stieg.",
    "Der Liegeplatz wurde {a} Stunden vor der planmaessigen Ankunft freigegeben; "
    "die Vorstauplanung musste um {b} Positionen angepasst werden.",
    "Bei der Kontrolle fielen {a} Einheiten wegen fehlender Begleitpapiere aus, "
    "was den Abtransport um {b} Stunden verzoegerte.",
    "Die Gleisanbindung nahm {a} Waggons auf, {b} weitere warteten im Vorbahnhof "
    "auf die Rangierfreigabe.",
    "Der Tiefgang erlaubte nur {a} Prozent der geplanten Beladung; {b} Einheiten "
    "blieben fuer die naechste Abfahrt am Kai.",
    "Die Zollabfertigung schloss {a} Sendungen ab, bei {b} Sendungen war eine "
    "Beschau angeordnet.",
    "Die Kuehlkette wurde durchgehend bei {a} Grad Celsius gehalten; {b} Messpunkte "
    "wichen kurzzeitig ab.",
    "Nach dem Sturmtief ruhte der Betrieb {a} Stunden, die Nacharbeit band {b} "
    "zusaetzliche Schichten.",
]
out = []
i = 0
while True:
    t = TERMINALS[i % len(TERMINALS)]
    k = KRANE[(i * 3) % len(KRANE)]
    w = WARE[(i * 5) % len(WARE)]
    b = BEFUND[(i * 7) % len(BEFUND)]
    a_val = 3 + (i * 11) % 97
    b_val = 2 + (i * 13) % 61
    out.append(
        f"Abschnitt {i + 1}. Am Terminal {t} wurde {w} mit einem {k} umgeschlagen. "
        + b.format(a=a_val, b=b_val)
        + f" Die Schichtleitung vermerkte Vorgang {1000 + i * 7} im Tagesbuch."
    )
    i += 1
    if i >= 170:
        break
text = ("Hintergrundmaterial fuer die Akte, nicht Teil der Aufgabenstellung:\n\n"
        + "\n".join(out))
open(pathlib.Path(__file__).with_name("vorkontext.txt"), "w").write(text)
print(f"{i} Absaetze, {len(text)} Zeichen")
