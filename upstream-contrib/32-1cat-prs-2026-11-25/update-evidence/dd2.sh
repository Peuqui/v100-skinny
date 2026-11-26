#!/bin/bash
# usage: dd2.sh oldbase oldhead newbase newhead [paths...] -> diff of only the added/removed lines (context ignored), file headers kept
cd /home/mp/Projekte/vllm-research/1Cat-vLLM
ob=$1; oh=$2; nb=$3; nh=$4; shift 4
f(){ git diff -U0 "$1" "$2" -- "$@" 2>/dev/null; }
only(){ grep -E '^(diff --git|\+|-)' | grep -v -E '^(\+\+\+|---)' ; }
diff <(git diff -U0 $ob $oh -- "$@" | only) <(git diff -U0 $nb $nh -- "$@" | only)
