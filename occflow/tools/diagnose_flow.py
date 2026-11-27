"""흐름 비용 보정이 바꾼 결정을 GT로 판정한다.

assoc.csv 에서 흐름 항이 적용된 행(active=1)마다
  base_choice  = 경합 후보 중 원래 비용(base_cost) 최소 검출
  flow_choice  = 실제로 선택된 검출 (chosen=1), 없으면 final_cost 최소
가 다르면 '뒤집힘'으로 본다. 그 트랙이 직전 window 프레임 동안 따라가던 GT(다수결)의 현재 박스와
IoU 가 가장 큰 검출을 정답으로 보고 판정한다.
  fixed   : flow_choice 가 정답, base_choice 는 아님
  broke   : base_choice 가 정답, flow_choice 는 아님
  unclear : 둘 다 아님 / 트랙의 GT 를 정할 수 없음

  python tools/diagnose_flow.py --mode_dir RUNS/oa_flow --debug_dir RUNS/oa_flow_debug \
      --det_cache RES/det_cache --gt_root VAL --seqs dancetrack0026 dancetrack0079 --out RES/diagnose
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from occflow.geometry import iou_matrix  # noqa: E402
from occflow.io_utils import find_result_file, load_gt, read_mot_txt, to_frame_dict  # noqa: E402


def det_thresh_from_spec(path):
    with open(path, "rb") as f:
        args, kwargs = pickle.load(f)
    if "det_thresh" in kwargs:
        return float(kwargs["det_thresh"])
    return float(args[1])


def high_dets(cache_dir, thr):
    data = np.load(os.path.join(cache_dir, "detections.npz"))
    dets = np.split(data["boxes"], np.cumsum(data["counts"])[:-1])
    return {i + 1: d[d[:, 4] > thr] for i, d in enumerate(dets)}


def follow_gt(pred, gt, tid, f, window, thr=0.5):
    votes = Counter()
    for x in range(f - window, f):
        if x not in pred or x not in gt:
            continue
        pi, pb, _ = pred[x]
        sel = np.flatnonzero(pi == tid)
        if not len(sel):
            continue
        gi, gb, _ = gt[x]
        if not len(gi):
            continue
        ious = iou_matrix(pb[sel[:1]], gb)[0]
        k = int(np.argmax(ious))
        if ious[k] >= thr:
            votes[int(gi[k])] += 1
    return votes.most_common(1)[0][0] if votes else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode_dir", required=True)
    ap.add_argument("--debug_dir", required=True)
    ap.add_argument("--det_cache", required=True)
    ap.add_argument("--gt_root", required=True)
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=10)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows_out = []
    for seq in a.seqs:
        cache = os.path.join(a.det_cache, seq)
        D = high_dets(cache, det_thresh_from_spec(os.path.join(cache, "tracker_spec.pkl")))
        G = to_frame_dict(load_gt(os.path.join(a.gt_root, seq, "gt", "gt.txt")))
        P = to_frame_dict(read_mot_txt(find_result_file(a.mode_dir, seq)))
        groups = defaultdict(list)
        with open(os.path.join(a.debug_dir, "%s_assoc.csv" % seq), encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["active"] == "1":
                    groups[(int(r["frame"]), int(r["track_id"]))].append(r)
        for (f, tid), rs in sorted(groups.items()):
            base_c = min(rs, key=lambda r: float(r["base_cost"]))
            chosen = [r for r in rs if r.get("chosen") == "1"]
            flow_c = chosen[0] if chosen else min(rs, key=lambda r: float(r["final_cost"]))
            jb, jf = int(base_c["det_idx"]), int(flow_c["det_idx"])
            if jb == jf:
                continue
            g = follow_gt(P, G, tid, f, a.window)
            verdict, best, best_iou = "unclear", -1, 0.0
            dets = D.get(f)
            if g is not None and f in G and dets is not None and len(dets):
                gi, gb, _ = G[f]
                sel = np.flatnonzero(gi == g)
                if len(sel):
                    ious = iou_matrix(gb[sel[:1]], dets[:, :4])[0]
                    best = int(np.argmax(ious))
                    best_iou = float(ious[best])
                    if best_iou >= 0.5:
                        if best == jf and best != jb:
                            verdict = "fixed"
                        elif best == jb and best != jf:
                            verdict = "broke"
            rows_out.append(dict(
                seq=seq, frame=f, track_id=tid, partner_id=int(rs[0]["partner_id"]), gt=g if g is not None else -1,
                verdict=verdict, base_det=jb, flow_det=jf, gt_det=best, gt_iou=round(best_iou, 3),
                chosen_known=int(bool(chosen)), n_cand=len(rs),
                base_gap=round(float(flow_c["base_cost"]) - float(base_c["base_cost"]), 4),
                mismatch_base=round(float(base_c["mismatch"]), 3), mismatch_flow=round(float(flow_c["mismatch"]), 3),
                adj_base=round(float(base_c["adj"]), 4), reliability=round(float(rs[0]["reliability"]), 3),
                consistency=round(float(rs[0]["consistency"]), 3),
            ))

    path = os.path.join(a.out, "flow_flips.csv")
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        if rows_out:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
            w.writeheader()
            w.writerows(rows_out)
    print("flips:", len(rows_out), "->", path)
    for seq in a.seqs + ["ALL"]:
        rs = [r for r in rows_out if seq == "ALL" or r["seq"] == seq]
        c = Counter(r["verdict"] for r in rs)
        print("%-15s fixed=%-4d broke=%-4d unclear=%-4d" % (seq, c["fixed"], c["broke"], c["unclear"]))
    keys = ("base_gap", "mismatch_base", "mismatch_flow", "reliability", "consistency", "n_cand")
    print("\n특징 평균 (fixed vs broke)")
    for v in ("fixed", "broke", "unclear"):
        rs = [r for r in rows_out if r["verdict"] == v]
        if rs:
            print("  %-8s n=%-4d " % (v, len(rs)) + "  ".join("%s=%.3f" % (k, np.mean([r[k] for r in rs])) for k in keys))


if __name__ == "__main__":
    main()
