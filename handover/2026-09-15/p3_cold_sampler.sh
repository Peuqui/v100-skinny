#!/usr/bin/env bash
# Paket 3, kalte SSD-Messung: Plattenzugriffe des PLE-Offload-Workers mitschreiben.
# Rein lesend (/proc, journalctl, llama-swap /running). Ende nach 40 Minuten.
OUT=/home/mp/Projekte/vllm-research/v100-skinny/handover/2026-09-15/p3_cold_worker_io.csv
echo "zeit,worker_pid,gelesen_mib,major_faults,cached_mib,modellzustand" > "$OUT"
SINCE=$(date '+%Y-%m-%d %H:%M:%S')
end=$((SECONDS + 2400))
pid=""
while [ $SECONDS -lt $end ]; do
  if [ -z "$pid" ] || [ ! -r /proc/$pid/io ]; then
    pid=$(journalctl -u llama-swap --since "$SINCE" -o cat --no-pager 2>/dev/null \
      | grep -o 'PleOffloadWorker pid=[0-9]*' | tail -1 | cut -d= -f2)
  fi
  state=$(curl -s --max-time 3 http://127.0.0.1:11435/running \
    | grep -o '"state":"[a-z]*"' | head -1 | cut -d'"' -f4)
  cached=$(awk '/^Cached:/{printf "%d", $2/1024}' /proc/meminfo)
  if [ -n "$pid" ] && [ -r /proc/$pid/io ]; then
    rb=$(awk '/^read_bytes/{printf "%d", $2/1048576}' /proc/$pid/io)
    mf=$(awk '{print $12}' /proc/$pid/stat)
    echo "$(date +%H:%M:%S),$pid,$rb,$mf,$cached,$state" >> "$OUT"
  else
    echo "$(date +%H:%M:%S),,,,$cached,$state" >> "$OUT"
  fi
  sleep 5
done
