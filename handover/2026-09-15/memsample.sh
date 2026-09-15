#!/bin/bash
# Samples host memory + vLLM process memory + GPU memory once per second during a cold start.
out="$1"
echo "t,mem_avail_mib,mem_free_mib,cached_mib,anon_mib,swap_used_mib,pswpout,pswpin,vllm_anon_mib,vllm_file_mib,vllm_swap_mib,vllm_procs,gpu0,gpu1,gpu2,gpu3,gpu4" > "$out"
end=$((SECONDS + 1500))
while [ $SECONDS -lt $end ]; do
  t=$(date +%T)
  m=$(awk '/^MemAvailable:/{a=$2}/^MemFree:/{f=$2}/^Cached:/{c=$2}/^AnonPages:/{an=$2}/^SwapTotal:/{st=$2}/^SwapFree:/{sf=$2}END{printf "%d,%d,%d,%d,%d",a/1024,f/1024,c/1024,an/1024,(st-sf)/1024}' /proc/meminfo)
  s=$(awk '/^pswpout /{o=$2}/^pswpin /{i=$2}END{printf "%d,%d",o,i}' /proc/vmstat)
  pa=0; pf=0; ps_=0; n=0
  for p in $(pgrep -f "^VLLM::|^/home/mp/vllm/venv/bin/python -m vllm.entrypoints"); do
    if [ -r /proc/$p/status ]; then
      read a f w < <(awk '/^RssAnon:/{a=$2}/^RssFile:/{f=$2}/^VmSwap:/{w=$2}END{print a+0, f+0, w+0}' /proc/$p/status 2>/dev/null)
      pa=$((pa + a)); pf=$((pf + f)); ps_=$((ps_ + w)); n=$((n + 1))
    fi
  done
  g=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ' | paste -sd,)
  echo "$t,$m,$s,$((pa/1024)),$((pf/1024)),$((ps_/1024)),$n,$g" >> "$out"
  if journalctl -u llama-swap --since "-3 sec" --no-pager 2>/dev/null | grep -q "Application startup complete"; then
    echo "# startup complete at $t" >> "$out"; sleep 30; fi
  sleep 1
done
