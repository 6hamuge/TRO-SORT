"""ID 변경 이벤트(analyze_failures 의 events.csv)가 어떤 상황에서 생겼는지 분류한다.

  pair   : 이벤트 전후 window 안에서 관련 예측 트랙이 '제3자 없는 두 트랙 교차' 구간에 있음
  group  : 관련 트랙이 같은 시간에 둘 이상의 상대와 교차 (3명 이상 뭉침)
  none   : 관련 트랙이 교차 구간에 없음 (단절/재생성 등)
교차 구간 정의는 occflow.deferred 와 같다 (IoU >= enter_iou).

  python tools/locate_errors.py --events OUT/analysis/oa_sort_events.csv --res RUNS/oa_sort --seqs ...
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from occflow.deferred import DeferParams, find_episodes, tracks_from_rows  # noqa: E402
from occflow.io_utils import find_result_file, read_mot_txt  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--window", type=int, default=10)
    a = ap.parse_args()
    p = DeferParams()
    ev = [r for r in csv.DictReader(open(a.events, encoding="utf-8-sig")) if r["seq"] in a.seqs]
    stat = Counter()
    by_cat = defaultdict(Counter)
    for seq in a.seqs:
        T = tracks_from_rows(read_mot_txt(find_result_file(a.res, seq)))
        eps = find_episodes(T, p)
        per_track = defaultdict(list)
        for e in eps:
            per_track[e["a"]].append((e["t0"], e["t1"], e["b"]))
            per_track[e["b"]].append((e["t0"], e["t1"], e["a"]))
        for r in (x for x in ev if x["seq"] == seq):
            f = int(r["frame"])
            lo, hi = f - a.window, f + a.window
            ids = {int(r["prev_pred"]), int(r["new_pred"])}
            partners = set()
            for tid in ids:
                for t0, t1, other in per_track.get(tid, []):
                    if t0 <= hi and t1 >= lo:
                        partners.add((tid, other))
            if not partners:
                where = "none"
            else:
                per = Counter(t for t, _ in partners)
                where = "group" if max(per.values()) >= 2 else "pair"
            stat[where] += 1
            by_cat[r["category"]][where] += 1
    n = sum(stat.values())
    print("events=%d  " % n + "  ".join("%s=%d (%.0f%%)" % (k, v, 100 * v / max(n, 1)) for k, v in stat.most_common()))
    for c in sorted(by_cat):
        print("  category %s: %s" % (c, dict(by_cat[c])))


if __name__ == "__main__":
    main()
