"""Fairer Vergleich: Ergebnis muss IMMER auf der Rechenkarte landen.

Rechenkarte 1 (V100, im Gitter eine PP-Stufe), Lager Karte 3 (freie V100,
peer-erreichbar von 1). Gemessen wird das PLE-Muster: 16 Zeilen a 160 Byte.
"""
import time, torch

RECHEN, LAGER = 1, 3
ROWS, DIM, G, N = 4_000_000, 160, 16, 2000

def zeit(fn):
    for _ in range(50): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / N * 1e6

torch.cuda.set_device(RECHEN)
print(f"Peer {RECHEN}->{LAGER}: {torch.cuda.can_device_access_peer(RECHEN, LAGER)}")

idx = torch.randint(0, ROWS, (G,), device=f"cuda:{RECHEN}")

lokal = torch.empty(ROWS, DIM, dtype=torch.float16, device=f"cuda:{RECHEN}")
t_lokal = zeit(lambda: lokal[idx])
del lokal; torch.cuda.empty_cache()

peer = torch.empty(ROWS, DIM, dtype=torch.float16, device=f"cuda:{LAGER}")
idx_l = idx.to(f"cuda:{LAGER}")
t_peer = zeit(lambda: peer[idx_l].to(f"cuda:{RECHEN}"))

host = torch.empty(ROWS, DIM, dtype=torch.float16, device="cpu").pin_memory()
idx_h = idx.cpu()
t_host = zeit(lambda: host[idx_h].to(f"cuda:{RECHEN}", non_blocking=True))

print(f"\nje Token (16 Zeilen a 160 B), Ergebnis auf Karte {RECHEN}:")
print(f"  lokales VRAM      : {t_lokal:7.1f} µs   (Bezug)")
print(f"  Nachbarkarte {LAGER}    : {t_peer:7.1f} µs   ×{t_peer/t_lokal:.1f}")
print(f"  gepinnter Host    : {t_host:7.1f} µs   ×{t_host/t_lokal:.1f}")
print(f"\nNachbarkarte gegen Host: ×{t_host/t_peer:.2f} schneller" if t_peer < t_host
      else f"\nHost gegen Nachbarkarte: ×{t_peer/t_host:.2f} schneller")
