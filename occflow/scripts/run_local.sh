#!/usr/bin/env bash
# setup_wsl.sh 이후 실행. colab_cells.py 의 셀 B~E 와 같은 순서.
#   bash run_local.sh                                   # 기본: 0026, 0079
#   SEQS="dancetrack0026" MODES="oa_sort,oa_null,oa_flow" bash run_local.sh
#   SEQS=all bash run_local.sh                          # val 전체 (val_seqmap.txt)
#   TAG=_w05 FLOW_SET="flow.weight=0.5" bash run_local.sh
# 결과: $OCC_ROOT/results  (det_cache / runs / eval / analysis)
set -euo pipefail

OCC_ROOT="${OCC_ROOT:-$HOME/occ}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OCCFLOW="$(dirname "$HERE")"
REPO="$OCC_ROOT/HybridSORT"
PY="$OCC_ROOT/venv/bin/python"
EVAL_PY="$OCC_ROOT/trackeval_venv/bin/python"
CODE_DIR="$OCC_ROOT/code_dir"
VAL="$OCC_ROOT/data/val"
CK="$REPO/pretrained/ocsort_dance_model.pth.tar"
EXP="$REPO/exps/example/mot/yolox_dancetrack_val_hybrid_sort.py"
OUT="$OCC_ROOT/results"
SEQS="${SEQS:-dancetrack0026 dancetrack0079}"
MODES="${MODES:-hybrid,oa_sort,oa_null,oa_flow,oa_rec,oa_audit,oa_flow_rec,oa_flow_audit,oa_full}"
TAG="${TAG:-}"
FLOW_SET="${FLOW_SET:-}"
if [[ "$SEQS" == all ]]; then
  SEQS="$(tail -n +2 "$OCC_ROOT/data/val_seqmap.txt" | tr -d '\r' | xargs)"
fi
mkdir -p "$OUT/det_cache" "$OUT/runs" "$OUT/logs"
export PYTHONPATH="$REPO" MPLBACKEND=Agg OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 YOLOX_DATADIR="$REPO/datasets"

echo "== 추적 ($MODES)"
for seq in $SEQS; do
  args=(--repo "$REPO" --code_dir "$CODE_DIR" --det_cache "$OUT/det_cache" --output "$OUT/runs"
        --sequence "$seq" --modes "$MODES" --frames "$VAL/$seq/img1" --weights "$CK" --config "$EXP" --fp16)
  [[ -n "$TAG" ]] && args+=(--tag "$TAG")
  [[ -n "$FLOW_SET" ]] && args+=(--set $FLOW_SET)
  echo "-- $seq"
  (cd "$REPO" && "$PY" "$OCCFLOW/occflow/integration/run_oa_flow.py" "${args[@]}") > "$OUT/logs/$seq$TAG.log" 2>&1 \
    || { tail -30 "$OUT/logs/$seq$TAG.log"; exit 1; }
  grep -E "^(hybrid|oa_)" "$OUT/logs/$seq$TAG.log" || true
done

# 모드 이름 (flow 계열은 TAG 접미사)
IFS=, read -ra M <<< "$MODES"
NAMES=()
for m in "${M[@]}"; do
  if [[ "$m" == oa_* && "$m" != oa_sort && "$m" != oa_null ]]; then NAMES+=("$m$TAG"); else NAMES+=("$m"); fi
done

echo "== 동작 검증: oa_null == oa_sort"
if [[ " ${NAMES[*]} " == *" oa_null "* && " ${NAMES[*]} " == *" oa_sort "* ]]; then
  "$PY" "$OCCFLOW/tools/check_equivalence.py" "$OUT/runs/oa_sort" "$OUT/runs/oa_null" || echo "!! 동일하지 않음"
fi

echo "== TrackEval"
EVAL="$OUT/eval$TAG"
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
if [[ -x "$EVAL_PY" ]]; then
  (PYTHONPATH= "$EVAL_PY" "$OCCFLOW/occflow/integration/evaluate_modes.py" "$EVAL" "$(IFS=,; echo "${NAMES[*]}")") \
    > "$OUT/logs/eval$TAG.log" 2>&1 || { tail -20 "$OUT/logs/eval$TAG.log"; exit 1; }
  "$PY" - "$EVAL/metrics.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
keys = ["HOTA", "AssA", "DetA", "IDF1", "MOTA", "IDSW", "FN", "Frag"]
seqs = list(next(iter(r.values())))
for s in seqs:
    print("\n[%s]" % ("COMBINED" if s == "COMBINED_SEQ" else s))
    print("%-16s" % "mode" + "".join("%8s" % k for k in keys))
    for m, v in r.items():
        print("%-16s" % m + "".join(("%8.2f" if isinstance(v[s][k], float) else "%8d") % v[s][k] for k in keys))
EOF
else
  echo "(TrackEval venv 없음: STEP=trackeval bash setup_wsl.sh)"
fi

echo "== 실패 유형 분석 (oa_sort 대비)"
RES=()
for n in oa_sort "${NAMES[@]}"; do
  [[ "$n" == oa_null ]] && continue
  [[ " ${RES[*]} " == *" $n=$OUT/runs/$n "* ]] && continue
  RES+=("$n=$OUT/runs/$n")
done
"$PY" "$OCCFLOW/tools/analyze_failures.py" --gt_root "$VAL" --res "${RES[@]}" \
  --det_root "$OUT/det_cache" --seqs $SEQS --out "$OUT/analysis$TAG" | tee "$OUT/logs/analysis$TAG.log"
echo "done -> $OUT"
