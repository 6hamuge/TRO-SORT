"""판단 보류 후처리를 추적기 결과에 적용 (+ 선택: GT 로 결정 판정).

  python tools/defer_resolve.py --res RUNS/oa_sort --out RUNS/oa_sort_defer --gt_root VAL \
      --seqs dancetrack0026 dancetrack0079 [--set mode=edges ratio=0.7]
  python tools/defer_resolve.py --oracle --gt_root VAL --seqs ...   # GT 트랙에 적용 (모든 swap = 오판)
출력: OUT/<seq>.txt, OUT/_defer_log.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from dataclasses import fields

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from occflow.deferred import DeferParams, find_episodes, judge_with_gt, link_cost, resolve, tracks_from_rows  # noqa: E402
from occflow.deferred import _window  # noqa: E402
from occflow.io_utils import find_result_file, list_sequences, load_gt, read_mot_txt, to_frame_dict  # noqa: E402


def parse_params(items):
    p = DeferParams()
    types = {f.name: f.type for f in fields(DeferParams)}
    for kv in items:
        k, v = kv.split("=", 1)
        cur = getattr(p, k)
        if isinstance(cur, bool):
            v = v.lower() in ("1", "true", "yes")
        elif isinstance(cur, int):
            v = int(v)
        elif isinstance(cur, float):
            v = float(v)
        setattr(p, k, v)
    return p


def oracle(gt_arr, p, noise, rng):
    """GT 트랙(정답 ID)에서 교차 구간마다 keep 이 swap 보다 싼지: edges(양방향) vs fwd(앞 구간만)."""
    arr = gt_arr.copy()
    if noise > 0:
        arr[:, 2:6] += rng.normal(0, noise, (len(arr), 4)) * np.tile(arr[:, 4:6], 2)
    T = tracks_from_rows(arr)
    res = Counter()
    for e in find_episodes(T, p):
        a, b, t0, t1 = e["a"], e["b"], e["t0"], e["t1"]
        if t1 - t0 + 1 > p.max_episode:
            continue
        pre_frames = range(t0 - 1, t0 - 1 - 3 * p.k - p.max_lag, -1)
        post_frames = range(t1 + 1, t1 + 1 + p.max_lag)
        segs = [_window(T, a, b, pre_frames, p)[::-1], _window(T, b, a, pre_frames, p)[::-1],
                _window(T, a, b, post_frames, p), _window(T, b, a, post_frames, p)]
        if min(len(s) for s in segs) < p.min_pts:
            continue
        L = t1 - t0 + 1
        bucket = "len<=5" if L <= 5 else ("len<=15" if L <= 15 else "len>15")
        for mode in ("edges", "fwd"):
            q = DeferParams(**{**p.__dict__, "mode": mode})
            keep = link_cost(segs[0], segs[2], q) + link_cost(segs[1], segs[3], q)
            swap = link_cost(segs[0], segs[3], q) + link_cost(segs[1], segs[2], q)
            ok = keep < swap
            fs = swap < keep * q.ratio and keep - swap > q.margin
            for bk in (bucket, "all"):
                res[(mode, bk, "n")] += 1
                res[(mode, bk, "ok")] += int(ok)
                res[(mode, bk, "false_swap")] += int(fs)
    return res


def oracle_group(gt_arr, p, noise, rng):
    """GT 트랙에서 뭉침 구간마다 Hungarian 이 '그대로(identity)'를 고르는지: edges vs fwd.
    또 임계값(ratio/margin)을 넘겨 실제로 순열을 바꿨을 오판 수도 센다."""
    from occflow.deferred import evaluate_group, find_groups
    arr = gt_arr.copy()
    if noise > 0:
        arr[:, 2:6] += rng.normal(0, noise, (len(arr), 4)) * np.tile(arr[:, 4:6], 2)
    T = tracks_from_rows(arr)
    res = Counter()
    for g in find_groups(T, p):
        for mode in ("edges", "fwd"):
            q = DeferParams(**{**p.__dict__, "mode": mode})
            r = evaluate_group(T, g, q)
            if r["decision"] == "skip" and mode == "edges":
                res[("skip", r["reason"])] += 1
            if r["decision"] == "skip":
                break
            n = len(r["judged"].split())
            bk = "n=2" if n == 2 else ("n=3" if n == 3 else "n>=4")
            ok = r["perm"] == ""
            for b in (bk, "all"):
                res[(mode, b, "n")] += 1
                res[(mode, b, "ok")] += int(ok)
                res[(mode, b, "false_swap")] += int(r["decision"] == "swap")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--res", help="추적기 결과 폴더")
    ap.add_argument("--out", help="후처리 결과 폴더")
    ap.add_argument("--gt_root", required=True)
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--noise", type=float, default=0.0)
    a = ap.parse_args()
    p = parse_params(a.set)
    seqs = a.seqs or list_sequences(a.gt_root)

    if a.oracle:
        tot = Counter()
        rng = np.random.RandomState(0)
        fn = oracle_group if p.unit == "group" else oracle
        for s in seqs:
            tot += fn(load_gt(os.path.join(a.gt_root, s, "gt", "gt.txt")), p, a.noise, rng)
        if p.unit == "group":
            print("skip:", {k[1]: v for k, v in tot.items() if k[0] == "skip"})
        for bk in ("len<=5", "len<=15", "len>15", "n=2", "n=3", "n>=4", "all"):
            n = tot[("edges", bk, "n")]
            if n:
                print("%-8s n=%-5d  양방향 acc=%.3f 오판swap=%d | 앞구간만 acc=%.3f 오판swap=%d" % (
                    bk, n, tot[("edges", bk, "ok")] / n, tot[("edges", bk, "false_swap")],
                    tot[("fwd", bk, "ok")] / n, tot[("fwd", bk, "false_swap")]))
        return

    os.makedirs(a.out, exist_ok=True)
    all_logs = []
    for s in seqs:
        path = find_result_file(a.res, s)
        if path is None:
            print("[skip] no result", s)
            continue
        G = to_frame_dict(load_gt(os.path.join(a.gt_root, s, "gt", "gt.txt")))
        arr = read_mot_txt(path)
        new, logs = resolve(arr, p, judge=judge_with_gt(G))
        with open(os.path.join(a.out, s + ".txt"), "w", encoding="utf-8") as f:
            for r in new:
                f.write("%d,%d,%.4f,%.4f,%.4f,%.4f,1,-1,-1,-1\n" % (r[0], r[1], r[2], r[3], r[4], r[5]))
        for lg in logs:
            lg["seq"] = s
        all_logs += logs
        c = Counter(lg["decision"] for lg in logs)
        v = Counter(lg.get("verdict", "") for lg in logs if lg["decision"] != "skip")
        print("%s episodes=%d swap=%d keep=%d skip=%d | fixed=%d false_swap=%d missed=%d ok_keep=%d unknown=%d" % (
            s, len(logs), c["swap"], c["keep"], c["skip"], v["fixed"], v["false_swap"], v["missed"],
            v["ok_keep"], v["unknown"]))
    keys = []
    for lg in all_logs:
        for k in lg:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(a.out, "_defer_log.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(all_logs)
    sk = Counter(lg["reason"] for lg in all_logs if lg["decision"] == "skip")
    lags = [lg["lag"] for lg in all_logs if lg["decision"] != "skip"]
    print("skip reasons:", dict(sk), "| decision lag frames: median=%s max=%s" % (
        int(np.median(lags)) if lags else "-", max(lags) if lags else "-"))


if __name__ == "__main__":
    main()
