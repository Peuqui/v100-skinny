import os, time, glob
def snap():
    out = {}
    for d in glob.glob('/proc/[0-9]*'):
        try:
            io = dict(l.split(': ') for l in open(d + '/io').read().splitlines())
            comm = open(d + '/comm').read().strip()
            out[int(d[6:])] = (comm, int(io['read_bytes']), int(io['write_bytes']))
        except Exception:
            pass
    return out
def disks():
    r = {}
    for l in open('/proc/diskstats'):
        f = l.split()
        if f[2].startswith(('nvme', 'sd')) and not f[2][-1].isdigit() or f[2].startswith('nvme') and 'p' not in f[2][4:]:
            r[f[2]] = (int(f[5]) * 512, int(f[9]) * 512, int(f[11]))
    return r
a, da = snap(), disks(); time.sleep(6); b, db = snap(), disks()
print('Geräte (MB/s lesen, schreiben, Anfragen in Bearbeitung):')
for k in db:
    if k in da:
        print(f'  {k}: {(db[k][0]-da[k][0])/6e6:7.1f} {(db[k][1]-da[k][1])/6e6:7.1f}  inflight={db[k][2]}')
rows = []
for pid, (comm, rb, wb) in b.items():
    if pid in a:
        rows.append(((rb - a[pid][1]) / 6e6, (wb - a[pid][2]) / 6e6, pid, comm))
print('Prozesse mit der größten I/O (lesen MB/s, schreiben MB/s):')
for r, w, pid, comm in sorted(rows, key=lambda x: -(x[0] + x[1]))[:8]:
    print(f'  {pid:>8} {comm:<18} {r:7.1f} {w:7.1f}')
