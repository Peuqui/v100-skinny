"""Nur der Transport: 2.560 Byte je Token (16 Zeilen a 160 B) auf die
Rechenkarte holen — einmal von der Nachbarkarte, einmal aus gepinntem Host.
Kein Gather, kein Geraetewechsel im Messpfad."""
import time, torch

RECHEN, LAGER, BYTES, N = 1, 3, 16 * 160, 5000
torch.cuda.set_device(RECHEN)
elems = BYTES // 2  # float16

ziel = torch.empty(elems, dtype=torch.float16, device=f"cuda:{RECHEN}")
peer = torch.empty(elems, dtype=torch.float16, device=f"cuda:{LAGER}")
host = torch.empty(elems, dtype=torch.float16).pin_memory()
lokal = torch.empty(elems, dtype=torch.float16, device=f"cuda:{RECHEN}")

def zeit(src):
    for _ in range(200): ziel.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N): ziel.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / N * 1e6

t_l, t_p, t_h = zeit(lokal), zeit(peer), zeit(host)
print(f"Peer {RECHEN}->{LAGER}: {torch.cuda.can_device_access_peer(RECHEN, LAGER)}")
print(f"\n{BYTES} Byte je Token auf Karte {RECHEN} holen:")
print(f"  lokales VRAM    : {t_l:6.2f} µs")
print(f"  Nachbarkarte {LAGER}  : {t_p:6.2f} µs   ×{t_p/t_l:.1f} gegen lokal")
print(f"  gepinnter Host  : {t_h:6.2f} µs   ×{t_h/t_l:.1f} gegen lokal")
v = "Nachbarkarte" if t_p < t_h else "Host"
print(f"\n→ {v} gewinnt, Faktor {max(t_h,t_p)/min(t_h,t_p):.2f}")
