"""Kann eine brachliegende GPU als PLE-Auslagerungsziel dienen?

Drei Fragen, in dieser Reihenfolge — Korrektheit vor Tempo:
1. Welche Kartenpaare melden Peer-Zugriff? (P2P ist wegen des defekten
   Ryzen-Root-Ports fuer NCCL abgeschaltet; ob schlichter Speicherzugriff
   geht, ist ungetestet.)
2. Liefert ein Peer-Lesevorgang KORREKTE Bytes? Stillschweigend falsche
   Daten waeren der schlimmste Ausgang.
3. Wie schnell ist das Zugriffsmuster der PLE — 16 Zeilen a 160 Byte pro
   Token — gegen die beiden Alternativen: lokales VRAM und gepinnter Host?

Aufruf: peer_ple_probe.py [zielkarte]   (Standard: 3, die freie V100)
"""
import sys, time
import torch

ZIEL = int(sys.argv[1]) if len(sys.argv) > 1 else 3
ROWS, DIM, GATHER = 4_000_000, 160, 16   # ~640 MB Tabelle, 16 Zeilen je Token
WIEDERHOLUNGEN = 2000

n = torch.cuda.device_count()
print(f"{n} Karten sichtbar\n")

print("1) Peer-Matrix (Zeile = von, Spalte = nach):")
print("     " + " ".join(f"{j:>4}" for j in range(n)))
for i in range(n):
    zeile = [("  ja" if torch.cuda.can_device_access_peer(i, j) else "nein") if i != j else "   —"
             for j in range(n)]
    print(f"  {i}: " + " ".join(f"{z:>4}" for z in zeile))

def gather_zeit(quelle: torch.Tensor, idx_dev: str) -> float:
    idx = torch.randint(0, ROWS, (GATHER,), device=idx_dev)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(WIEDERHOLUNGEN):
        _ = quelle[idx]
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / WIEDERHOLUNGEN * 1e6   # µs je Token

print(f"\n2) Korrektheit: Karte {ZIEL} beschreiben, von Karte 0 lesen")
torch.cuda.set_device(0)
muster = torch.arange(ROWS * DIM, dtype=torch.float16, device=f"cuda:{ZIEL}").reshape(ROWS, DIM)
probe_idx = torch.tensor([0, 1, ROWS // 2, ROWS - 1], device=f"cuda:{ZIEL}")
gelesen = muster[probe_idx].to("cuda:0")
erwartet = (probe_idx.unsqueeze(1) * DIM + torch.arange(DIM, device=f"cuda:{ZIEL}")).to(torch.float16).to("cuda:0")
ok = torch.equal(gelesen, erwartet)
print(f"   {'KORREKT' if ok else 'FALSCHE DATEN — Abbruch'}")
if not ok:
    sys.exit(1)

print("\n3) Zugriffsmuster der PLE (16 Zeilen a 160 B, je Token)")
lokal = torch.empty(ROWS, DIM, dtype=torch.float16, device="cuda:0")
print(f"   lokales VRAM (Karte 0) : {gather_zeit(lokal, 'cuda:0'):8.1f} µs")
del lokal; torch.cuda.empty_cache()
print(f"   Nachbarkarte {ZIEL}        : {gather_zeit(muster, f'cuda:{ZIEL}'):8.1f} µs")
host = torch.empty(ROWS, DIM, dtype=torch.float16, device="cpu").pin_memory()
idx_h = torch.randint(0, ROWS, (GATHER,))
t0 = time.perf_counter()
for _ in range(WIEDERHOLUNGEN):
    _ = host[idx_h].to("cuda:0", non_blocking=True)
torch.cuda.synchronize()
print(f"   gepinnter Host         : {(time.perf_counter()-t0)/WIEDERHOLUNGEN*1e6:8.1f} µs")
