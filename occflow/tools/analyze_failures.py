"""실패 유형 분석 + 추적기 비교.

예) 재현 결과에서 0026, 0079 의 Hybrid-SORT vs OA-SORT 비교
  python tools/analyze_failures.py --gt_root BENCH/gt \
      --res hybrid=BENCH/trackers/hybrid/data oa_sort=BENCH/trackers/oa_sort/data \
      --det_root BENCH/occflow/det_cache --seqs dancetrack0026 dancetrack0079 --out out/analysis

출력 (out/):
  <name>_events.csv   ID 변경 이벤트 (category 1/2/R, persistent, 끊김 구간 검출 유무, 가림 상대, 교집합 흐름)
  <name>_gts.csv      GT별 FN / IDSW / Frag / 틀린 ID 프레임 수 (AssA 하락 원인 추적용)
  <name>_summary.csv  시퀀스별 + COMBINED 요약
  compare_<a>_vs_<b>_events.csv / _summary.csv   두 추적기 이벤트 대응 (only_a / only_b / both)
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from occflow.analysis import AnalysisParams, analyze_sequence, combine, compare_events  # noqa: E402
from occflow.io_utils import (find_det_file, find_result_file, list_sequences, load_det_frames,  # noqa: E402
                              load_gt, read_mot_txt)


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys or ["empty"], restval="")
        w.writeheader()
        w.writerows(rows)


SHOW = ("seq", "IDSW", "Frag", "FN", "FP", "IDF1", "cat1_immediate", "cat2_break", "cat3_persistent",
        "wrong_id_frames", "fn_det_high", "fn_no_det")


def print_table(title, rows):
    print("\n== %s" % title)
    print(" | ".join("%15s" % k if k == "seq" else "%8s" % k[:8] for k in SHOW))
    for r in rows:
        print(" | ".join("%15s" % r[k] if k == "seq" else "%8s" % r[k] for k in SHOW))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt_root", required=True, help="<seq>/gt/gt.txt 가 있는 폴더 (DanceTrack val 또는 BENCH/gt)")
    ap.add_argument("--res", nargs="+", required=True, help="name=결과폴더 (여러 개 가능, 첫 번째가 비교 기준)")
    ap.add_argument("--det_root", default=None, help="검출 캐시 (<seq>/det.txt) - 끊김 구간 검출 유무 판단")
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--out", default="out/analysis")
    ap.add_argument("--swap_max_gap", type=int, default=2)
    ap.add_argument("--swap_window", type=int, default=10)
    ap.add_argument("--persist_min", type=int, default=30)
    ap.add_argument("--det_high", type=float, default=0.6)
    ap.add_argument("--det_low", type=float, default=0.1)
    ap.add_argument("--top", type=int, default=8, help="시퀀스별로 출력할 긴 지속 오류 이벤트 수")
    a = ap.parse_args()

    params = AnalysisParams(swap_max_gap=a.swap_max_gap, swap_window=a.swap_window,
                            persist_min=a.persist_min, det_high=a.det_high, det_low=a.det_low)
    seqs = a.seqs or list_sequences(a.gt_root)
    trackers = [tuple(x.split("=", 1)) for x in a.res]
    gts = {s: load_gt(os.path.join(a.gt_root, s, "gt", "gt.txt")) for s in seqs}
    dets = {}
    if a.det_root:
        for s in seqs:
            path = find_det_file(a.det_root, s)
            dets[s] = load_det_frames(path) if path else None
            if path is None:
                print("[warn] 검출 파일 없음:", s)

    results = {}
    for name, rdir in trackers:
        ev_all, gt_all, sums = [], [], []
        for s in seqs:
            path = find_result_file(rdir, s)
            if path is None:
                print("[warn] %s: 결과 없음 %s" % (name, s))
                continue
            r = analyze_sequence(s, gts[s], read_mot_txt(path), dets.get(s), params)
            ev_all += r["events"]
            gt_all += r["gts"]
            sums.append(r["summary"])
        if not sums:
            continue
        sums.append(combine(sums))
        results[name] = dict(events=ev_all, summary=sums)
        write_csv(os.path.join(a.out, "%s_events.csv" % name), ev_all)
        write_csv(os.path.join(a.out, "%s_gts.csv" % name), gt_all)
        write_csv(os.path.join(a.out, "%s_summary.csv" % name), sums)
        print_table(name, sums)
        persist = sorted((e for e in ev_all if e["persistent"]), key=lambda e: -e["run_len"])[: a.top * len(seqs)]
        if persist:
            print("  -- 지속 오류 상위 (seq gt frame cat/kind run gap_fn partner flow)")
            for e in persist:
                print("   %s g%-3d f%-5d %s/%-8s run=%-4d gap=%-3d partner=g%-3d %s" % (
                    e["seq"], e["gt_id"], e["frame"], e["category"], e["kind"], e["run_len"],
                    e["gap_fn"], e["partner_gt"], e["overlap_flow"]))

    names = list(results)
    for other in names[1:]:
        ref = names[0]
        rows = compare_events(results[ref]["events"], results[other]["events"])
        write_csv(os.path.join(a.out, "compare_%s_vs_%s_events.csv" % (ref, other)), rows)
        diff = []
        for sa, sb in zip(results[ref]["summary"], results[other]["summary"]):
            if sa["seq"] != sb["seq"]:
                continue
            d = {"seq": sa["seq"]}
            for k in SHOW[1:]:
                d[k] = round(sb[k] - sa[k], 3)
            diff.append(d)
        write_csv(os.path.join(a.out, "compare_%s_vs_%s_summary.csv" % (ref, other)), diff)
        print_table("%s - %s (차이)" % (other, ref), diff)
        c = {}
        for r in rows:
            c[r["status"]] = c.get(r["status"], 0) + 1
        print("  이벤트 대응: both=%d only_%s=%d only_%s=%d" % (c.get("both", 0), ref, c.get("only_a", 0),
                                                             other, c.get("only_b", 0)))


if __name__ == "__main__":
    main()
