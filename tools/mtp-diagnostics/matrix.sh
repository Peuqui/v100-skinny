#!/usr/bin/env bash
# Messmatrix zum MTP-Fix (sm75-GDN-Builder bekommt die Spekulations-Metadaten).
# Ein Lauf nach dem anderen — die Karten werden exklusiv gebraucht.
#
# Kartenlage (CUDA_DEVICE_ORDER=PCI_BUS_ID):
#   0 = Quadro RTX 8000 (sm75)   1 = Tesla V100 (sm70)
#   2 = Quadro RTX 8000 (sm75)   3 = Tesla V100 (sm70)   4 = Tesla V100 (sm70)
#
# Bei TP2/PP2 bilden Rang 0+1 die erste Stufe. DEVS=0,2,1,3 ergibt deshalb
# eine reine Turing-Stufe und eine reine Volta-Stufe (Flash-Next-Struktur);
# DEVS=0,1,2,3 wuerde heterogene TP-Gruppen bilden und ist sinnlos.
set -uo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/probe.sh"

run() {  # name  env...
  local name=$1; shift
  echo "################ $name ################"
  env "$@" "$P" "$name" 2>&1 | tail -18
  echo
}

case "${1:-all}" in
  piecewise)  # war laut Handover "Kauderwelsch" — traegt der Fix auch hier?
    run rtx_piecewise DEVS=0,2 TP=2 PP=1 CGMODE=PIECEWISE ;;
  eager)      # Kontrolle: war schon vor dem Fix korrekt
    run rtx_eager DEVS=0,2 TP=2 PP=1 CGMODE=NONE ;;
  rtx_pp2)    # PP auf Turing allein
    run rtx_pp2_tp1 DEVS=0,2 TP=1 PP=2 ;;
  volta_pp2)  # PP auf Volta allein (anderer GDN-Pfad: FlashQLA-SM70)
    run volta_pp2_tp1 DEVS=1,3 TP=1 PP=2 ;;
  volta_tp2)  # Volta-Kontrolle ohne PP
    run volta_tp2_pp1 DEVS=1,3 TP=2 PP=1 ;;
  mixed)      # Zielfall: Turing-Stufe + Volta-Stufe
    run mixed_tp2_pp2 DEVS=0,2,1,3 TP=2 PP=2 ;;
  k1)         # zweite Spekulationstiefe
    run rtx_k1 DEVS=0,2 TP=2 PP=1 K=1 ;;
  *) echo "Aufruf: matrix.sh {piecewise|eager|rtx_pp2|volta_pp2|volta_tp2|mixed|k1}" ;;
esac
