#!/usr/bin/env bash
# 판단 보류 후처리: run_local.sh 결과(runs/<mode>)에 적용 -> runs/<mode>_defer<TAG>, TrackEval, 실패 분석
#   bash run_defer.sh
#   BASE="oa_sort hybrid" DEFER_SET="ratio=0.6 margin=0.4" TAG=_r06 bash run_defer.sh
set -euo pipefail
OCC_ROOT="${OCC_ROOT:-$HOME/occ}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OCCFLOW="$(dirname "$HERE")"
PY="$OCC_ROOT/venv/bin/python"
EVAL_PY="$OCC_ROOT/trackeval_venv/bin/python"
VAL="$OCC_ROOT/data/val"
OUT="$OCC_ROOT/results"
SEQS="${SEQS:-dancetrack0026 dancetrack0079}"
BASE="${BASE:-oa_sort hybrid}"
TAG="${TAG:-}"
DEFER_SET="${DEFER_SET:-}"

NAMES=()
for m in $BASE; do
  echo "== defer: $m"
  args=(--res "$OUT/runs/$m" --out "$OUT/runs/${m}_defer$TAG" --gt_root "$VAL" --seqs $SEQS)
  [[ -n "$DEFER_SET" ]] && args+=(--set $DEFER_SET)
  "$PY" "$OCCFLOW/tools/defer_resolve.py" "${args[@]}"
  NAMES+=("$m" "${m}_defer$TAG")
done

echo "== TrackEval"
EVAL="$OUT/eval_defer$TAG"
rm -rf "$EVAL"; mkdir -p "$EVAL/gt"
printf "name\n" > "$EVAL/seqmap.txt"
for seq in $SEQS; do
  mkdir -p "$EVAL/gt/$seq/gt"
  cp "$VAL/$seq/seqinfo.ini" "$EVAL/gt/$seq/"
  cp "$VAL/$seq/gt/gt.txt" "$EVAL/gt/$seq/gt/"
  echo "$seq" >> "$EVAL/seqmap.txt"
  for n in "${NAMES[@]}"; do
    mkdir -p "$EVAL/trackers/$n/data"
    cp "$OUT/runs/$n/$seq.txt" "$EVAL/trackers/$n/data/"
  done
done
(PYTHONPATH= "$EVAL_PY" "$OCCFLOW/occflow/integration/evaluate_modes.py" "$EVAL" "$(IFS=,; echo "${NAMES[*]}")") \
  > "$OUT/logs/eval_defer$TAG.log" 2>&1 || { tail -20 "$OUT/logs/eval_defer$TAG.log"; exit 1; }
"$PY" - "$EVAL/metrics.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
keys = ["HOTA", "AssA", "DetA", "IDF1", "MOTA", "IDSW", "FN", "Frag"]
for s in next(iter(r.values())):
    print("\n[%s]" % ("COMBINED" if s == "COMBINED_SEQ" else s))
    print("%-20s" % "mode" + "".join("%8s" % k for k in keys))
    for m, v in r.items():
        print("%-20s" % m + "".join(("%8.2f" if isinstance(v[s][k], float) else "%8d") % v[s][k] for k in keys))
EOF

echo "== 실패 유형 분석"
for m in $BASE; do
  "$PY" "$OCCFLOW/tools/analyze_failures.py" --gt_root "$VAL" --res "$m=$OUT/runs/$m" \
    "${m}_defer$TAG=$OUT/runs/${m}_defer$TAG" --det_root "$OUT/det_cache" --seqs $SEQS \
    --out "$OUT/analysis_defer$TAG" | grep -A4 "차이"
done
