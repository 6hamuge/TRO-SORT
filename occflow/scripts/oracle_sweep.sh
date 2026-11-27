#!/usr/bin/env bash
# 오라클 실험: 관측 공백(gap)과 박스 잡음(noise)에 따라 flow vs motion 정확도 비교 (어려운 프레임만)
#   bash oracle_sweep.sh                       # 기본: 0026, 0079
#   SEQS=all bash oracle_sweep.sh              # val 전체 (val_seqmap.txt)
# 시퀀스별 원본 출력: $OUT/oracle_<noise>_<gap>.txt
set -euo pipefail
OCC_ROOT="${OCC_ROOT:-$HOME/occ}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEQS="${SEQS:-dancetrack0026 dancetrack0079}"
GAPS="${GAPS:-1 3 5 10 15}"
NOISES="${NOISES:-0 0.03}"
OUT="${OUT:-$OCC_ROOT/results/oracle}"
if [[ "$SEQS" == all ]]; then
  SEQS="$(tail -n +2 "$OCC_ROOT/data/val_seqmap.txt" | tr -d '\r' | xargs)"
fi
mkdir -p "$OUT"
for n in $NOISES; do
  for g in $GAPS; do
    f="$OUT/oracle_${n}_${g}.txt"
    "$OCC_ROOT/venv/bin/python" "$(dirname "$HERE")/tools/oracle_flow.py" --gt_root "$OCC_ROOT/data/val" \
      --seqs $SEQS --gap "$g" --noise "$n" > "$f"
    line=$(grep -E '^ALL +hard' "$f" | sed -E 's/^ALL +hard +//')
    printf 'noise=%-5s gap=%-3s %s\n' "$n" "$g" "$line"
  done
done
