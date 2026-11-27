"""판단 보류 (deferred decision): 교차 구간에서는 ID 확정을 미루고, 두 트랙이 다시 떨어진 뒤
교차 '이전' 구간과 '이후' 구간을 양방향으로 이어 보아 원래대로 둘지(keep) 맞바꿀지(swap) 정한다.

추적기 출력(MOT 행)에 적용하는 고정 지연 후처리다. 결정은 교차 종료 후 k 프레임(최대 max_lag) 안에
내려지므로, 출력이 그만큼 늦게 확정되는 준온라인 방식으로 볼 수 있다 (결정별 지연을 기록).

연결 비용 C(pre_a, post_b):
  mode='edges' : pre 구간 1차 회귀를 post 첫 프레임으로 연장한 값 vs post 실제 박스  +
                 post 구간 1차 회귀를 pre 마지막 프레임으로 역연장한 값 vs pre 실제 박스  (양방향 평균)
  mode='fwd'   : pre 구간만 앞으로 연장 (기존 온라인 추적기와 같은 정보, 비교 기준)
  거리 = 중심 거리 / 평균 높이 + size_w * |log(높이 비)|
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .geometry import iou_matrix


@dataclass
class DeferParams:
    enter_iou: float = 0.3     # 이 이상 겹치면 교차(판단 보류) 구간
    exit_iou: float = 0.1      # pre/post 구간은 두 트랙 IoU 가 이 미만인 프레임만 사용
    merge_gap: int = 2         # 교차 프레임 사이 공백이 이 이하면 같은 구간
    k: int = 5                 # pre/post 구간 길이 (회귀에 쓰는 프레임 수)
    min_pts: int = 3
    max_episode: int = 60      # 이보다 긴 교차(계속 붙어 있는 대형)는 판단하지 않음
    max_lag: int = 30          # post 구간은 교차 종료 후 이 프레임 안에서 찾음 (= 최대 출력 지연)
    ratio: float = 0.7         # swap 비용 < keep 비용 * ratio
    margin: float = 0.3        # keep - swap > margin
    size_w: float = 1.0
    mode: str = "edges"
    skip_multi: bool = True    # (pair 단위) 같은 시간에 제3자와도 교차하면 판단하지 않음
    unit: str = "group"        # 'group' = 겹치는 트랙들을 묶어 순열로 판단 / 'pair' = 두 트랙 단위
    max_group: int = 5         # 이보다 많은 트랙이 뭉친 구간은 판단하지 않음


Box = np.ndarray


def tracks_from_rows(arr: np.ndarray) -> Dict[int, Dict[int, Box]]:
    """MOT 배열(frame,id,x,y,w,h,...) -> {id: {frame: tlbr}}"""
    T: Dict[int, Dict[int, Box]] = defaultdict(dict)
    for r in arr:
        T[int(r[1])][int(r[0])] = np.array([r[2], r[3], r[2] + r[4], r[3] + r[5]], dtype=float)
    return T


def find_episodes(T: Dict[int, Dict[int, Box]], p: DeferParams):
    by_frame: Dict[int, List[Tuple[int, Box]]] = defaultdict(list)
    for tid, fr in T.items():
        for f, b in fr.items():
            by_frame[f].append((tid, b))
    pair_frames: Dict[Tuple[int, int], List[Tuple[int, float]]] = defaultdict(list)
    for f, items in by_frame.items():
        if len(items) < 2:
            continue
        ids = [t for t, _ in items]
        M = iou_matrix(np.stack([b for _, b in items]), np.stack([b for _, b in items]))
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                if M[a, b] >= p.enter_iou:
                    key = (min(ids[a], ids[b]), max(ids[a], ids[b]))
                    pair_frames[key].append((f, float(M[a, b])))
    eps = []
    for (a, b), fl in pair_frames.items():
        fl.sort()
        start = prev = fl[0][0]
        peak = fl[0]
        for f, v in fl[1:] + [(10 ** 9, 0.0)]:
            if f - prev > p.merge_gap + 1:
                eps.append(dict(a=a, b=b, t0=start, t1=prev, peak=peak[0], peak_iou=peak[1]))
                start, peak = f, (f, v)
            elif v > peak[1]:
                peak = (f, v)
            prev = f
    eps.sort(key=lambda e: (e["t1"], e["t0"]))
    return eps


def _pair_iou(T, a, b, f) -> float:
    A, B = T[a].get(f), T[b].get(f)
    if A is None or B is None:
        return 0.0
    return float(iou_matrix(A[None], B[None])[0, 0])


def _window(T, a, b, frames, p: DeferParams):
    out = []
    for f in frames:
        if f in T[a] and _pair_iou(T, a, b, f) < p.exit_iou:
            out.append((f, T[a][f]))
            if len(out) >= p.k:
                break
    return out


def _fit(seg):
    """[(frame, box)] -> f(t) = (cx, cy, h)"""
    fs = np.array([f for f, _ in seg], dtype=float)
    vals = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[3] - b[1]] for _, b in seg])
    if len(fs) >= 2 and fs.max() > fs.min():
        A = np.vstack([fs, np.ones_like(fs)]).T
        coef = np.linalg.lstsq(A, vals, rcond=None)[0]
        return lambda t: coef[0] * t + coef[1]
    v = vals.mean(0)
    return lambda t: v


def _dist(x, y, size_w):
    h = max(0.5 * (x[2] + y[2]), 1e-6)
    return float(np.hypot(x[0] - y[0], x[1] - y[1]) / h + size_w * abs(np.log(max(x[2], 1e-6) / max(y[2], 1e-6))))


def _obs(b):
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[3] - b[1]])


def link_cost(pre, post, p: DeferParams) -> float:
    fpre, fpost = _fit(pre), _fit(post)
    t_post, t_pre = post[0][0], pre[-1][0]
    fwd = _dist(fpre(t_post), _obs(post[0][1]), p.size_w)
    if p.mode == "fwd":
        return fwd
    bwd = _dist(fpost(t_pre), _obs(pre[-1][1]), p.size_w)
    return 0.5 * (fwd + bwd)


def evaluate_episode(T, e, p: DeferParams, busy=None):
    """반환: dict(decision='keep'|'swap'|'skip', reason, keep, swap, pre/post 구간 프레임)"""
    a, b, t0, t1 = e["a"], e["b"], e["t0"], e["t1"]
    out = dict(e)
    if t1 - t0 + 1 > p.max_episode:
        return dict(out, decision="skip", reason="too_long")
    if p.skip_multi and busy is not None:
        for tid in (a, b):
            for (s0, s1, other) in busy.get(tid, []):
                if other not in (a, b) and s0 <= t1 and s1 >= t0:
                    return dict(out, decision="skip", reason="multi")
    pre_frames = range(t0 - 1, t0 - 1 - 3 * p.k - p.max_lag, -1)
    post_frames = range(t1 + 1, t1 + 1 + p.max_lag)
    pre_a = _window(T, a, b, pre_frames, p)[::-1]
    pre_b = _window(T, b, a, pre_frames, p)[::-1]
    post_a = _window(T, a, b, post_frames, p)
    post_b = _window(T, b, a, post_frames, p)
    if min(len(pre_a), len(pre_b)) < p.min_pts:
        return dict(out, decision="skip", reason="short_pre")
    if min(len(post_a), len(post_b)) < p.min_pts:
        return dict(out, decision="skip", reason="short_post")
    keep = link_cost(pre_a, post_a, p) + link_cost(pre_b, post_b, p)
    swap = link_cost(pre_a, post_b, p) + link_cost(pre_b, post_a, p)
    decided_at = max(post_a[-1][0], post_b[-1][0])
    out.update(keep=round(keep, 4), swap=round(swap, 4), decided_at=decided_at, _segs=(pre_a, pre_b, post_a, post_b),
               lag=decided_at - e["peak"], pre_a=(pre_a[0][0], pre_a[-1][0]),
               post_a=(post_a[0][0], post_a[-1][0]), pre_b=(pre_b[0][0], pre_b[-1][0]),
               post_b=(post_b[0][0], post_b[-1][0]))
    if swap < keep * p.ratio and keep - swap > p.margin:
        return dict(out, decision="swap", reason="")
    return dict(out, decision="keep", reason="")


def _busy(eps):
    busy = defaultdict(list)
    for e in eps:
        busy[e["a"]].append((e["t0"], e["t1"], e["b"]))
        busy[e["b"]].append((e["t0"], e["t1"], e["a"]))
    return busy


def swap_tracks(T, a, b, ts):
    """프레임 ts 이후 a, b 의 박스를 맞바꾼다 (= 출력 ID 교환)."""
    A = {f: v for f, v in T[a].items() if f >= ts}
    B = {f: v for f, v in T[b].items() if f >= ts}
    for f in A:
        del T[a][f]
    for f in B:
        del T[b][f]
    T[a].update(B)
    T[b].update(A)


def judge_with_gt(gt_frames):
    """GT {frame: (ids, boxes, _)} -> 구간 [(f, box)] 의 다수결 GT id (IoU>=0.5), 없으면 -1."""
    def judge(seg):
        votes = defaultdict(int)
        for f, b in seg:
            g = gt_frames.get(f)
            if g is None or not len(g[0]):
                continue
            ious = iou_matrix(b[None], g[1])[0]
            k = int(np.argmax(ious))
            if ious[k] >= 0.5:
                votes[int(g[0][k])] += 1
        return max(votes, key=votes.get) if votes else -1
    return judge


def _verdict(ga, gb, ha, hb, decision):
    """ga/gb: pre 구간 GT, ha/hb: post 구간 GT. 필요한 동작과 실제 결정 비교."""
    if -1 in (ga, gb, ha, hb) or ga == gb:
        return "unknown"
    if ga == ha and gb == hb:
        need = "keep"
    elif ga == hb and gb == ha:
        need = "swap"
    else:
        return "unknown"
    if need == "swap":
        return "fixed" if decision == "swap" else "missed"
    return "false_swap" if decision == "swap" else "ok_keep"


def find_groups(T, p: DeferParams):
    """두 트랙 교차 구간들을 '트랙을 공유하고 시간이 겹치면' 하나로 묶는다 -> 뭉침 구간."""
    eps = find_episodes(T, p)
    parent = list(range(len(eps)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    per_track = defaultdict(list)
    for i, e in enumerate(eps):
        per_track[e["a"]].append(i)
        per_track[e["b"]].append(i)
    for idx in per_track.values():
        idx.sort(key=lambda i: eps[i]["t0"])
        cur, end = idx[0], eps[idx[0]]["t1"]
        for i in idx[1:]:
            if eps[i]["t0"] <= end + p.merge_gap + 1:
                parent[root(i)] = root(cur)
                end = max(end, eps[i]["t1"])
            else:
                cur, end = i, eps[i]["t1"]
    groups = defaultdict(list)
    for i in range(len(eps)):
        groups[root(i)].append(eps[i])
    out = []
    for es in groups.values():
        S = sorted({e["a"] for e in es} | {e["b"] for e in es})
        out.append(dict(tracks=S, t0=min(e["t0"] for e in es), t1=max(e["t1"] for e in es), n_pairs=len(es)))
    out.sort(key=lambda g: (g["t1"], g["t0"]))
    return out


def _window_group(T, a, S, frames, p):
    out = []
    for f in frames:
        if f not in T[a]:
            continue
        if all(_pair_iou(T, a, o, f) < p.exit_iou for o in S if o != a):
            out.append((f, T[a][f]))
            if len(out) >= p.k:
                break
    return out


def evaluate_group(T, g, p: DeferParams):
    from scipy.optimize import linear_sum_assignment
    S, t0, t1 = g["tracks"], g["t0"], g["t1"]
    out = dict(tracks=" ".join(map(str, S)), t0=t0, t1=t1, n=len(S), n_pairs=g["n_pairs"])
    if t1 - t0 + 1 > p.max_episode:
        return dict(out, decision="skip", reason="too_long")
    if len(S) > p.max_group:
        return dict(out, decision="skip", reason="too_many")
    pre_frames = range(t0 - 1, t0 - 1 - 3 * p.k - p.max_lag, -1)
    post_frames = range(t1 + 1, t1 + 1 + p.max_lag)
    pre = {a: _window_group(T, a, S, pre_frames, p)[::-1] for a in S}
    post = {a: _window_group(T, a, S, post_frames, p) for a in S}
    # 앞뒤 구간이 모두 있는 트랙만 판단 대상 (나머지는 그대로 둔다)
    U = [a for a in S if len(pre[a]) >= p.min_pts and len(post[a]) >= p.min_pts]
    if len(U) < 2:
        return dict(out, decision="skip", reason="short_window")
    C = np.array([[link_cost(pre[i], post[j], p) for j in U] for i in U])
    r, c = linear_sum_assignment(C)
    ident = float(np.trace(C))
    best = float(C[r, c].sum())
    perm = {U[i]: U[j] for i, j in zip(r, c)}           # pre 트랙 i 의 정체성은 post 트랙 j 로 이어짐
    decided_at = max(post[a][-1][0] for a in U)
    out.update(judged=" ".join(map(str, U)), keep=round(ident, 4), best=round(best, 4),
               decided_at=decided_at, lag=decided_at - (t0 + t1) // 2,
               perm=" ".join("%d>%d" % (i, j) for i, j in perm.items() if i != j),
               _segs=({a: pre[a] for a in U}, {a: post[a] for a in U}), _perm=perm)
    if all(i == j for i, j in perm.items()):
        return dict(out, decision="keep", reason="")
    if best < ident * p.ratio and ident - best > p.margin:
        return dict(out, decision="swap", reason="")
    return dict(out, decision="keep", reason="weak")


def permute_tracks(T, perm, ts):
    """프레임 ts 이후: 라벨 i 는 원래 트랙 perm[i] 의 박스를 갖는다."""
    tail = {j: {f: v for f, v in T[j].items() if f >= ts} for j in perm.values()}
    for j in perm.values():
        for f in tail[j]:
            del T[j][f]
    for i, j in perm.items():
        T[i].update(tail[j])


def _verdict_group(gpre, gpost, perm, decision):
    """gpre/gpost: {track: gt}. 필요한 순열(정답)과 결정 비교."""
    vals = list(gpre.values())
    if -1 in vals or -1 in gpost.values() or len(set(vals)) != len(vals):
        return "unknown"
    inv = {g: j for j, g in gpost.items()}
    need = {i: inv.get(g) for i, g in gpre.items()}
    if None in need.values():
        return "unknown"
    need_change = any(i != j for i, j in need.items())
    if decision == "swap":
        if need == perm:
            return "fixed"
        return "false_swap" if not need_change else "wrong_perm"
    return "missed" if need_change else "ok_keep"


def resolve(arr: np.ndarray, p: Optional[DeferParams] = None, judge=None):
    p = p or DeferParams()
    if p.unit == "pair":
        return resolve_pairs(arr, p, judge)
    T = tracks_from_rows(arr)
    logs, done = [], set()
    while True:
        applied = False
        for g in find_groups(T, p):
            key = (tuple(g["tracks"]), g["t0"], g["t1"])
            if key in done:
                continue
            done.add(key)
            r = evaluate_group(T, g, p)
            segs, perm = r.pop("_segs", None), r.pop("_perm", None)
            if judge is not None and segs is not None:
                gpre = {a: judge(s) for a, s in segs[0].items()}
                gpost = {a: judge(s) for a, s in segs[1].items()}
                r["gt_pre"] = " ".join("%d:%d" % kv for kv in gpre.items())
                r["gt_post"] = " ".join("%d:%d" % kv for kv in gpost.items())
                r["verdict"] = _verdict_group(gpre, gpost, perm, r["decision"])
            logs.append(r)
            if r["decision"] == "swap":
                permute_tracks(T, perm, (g["t0"] + g["t1"]) // 2)
                applied = True
                break
        if not applied:
            break
    return _rows(T), logs


def _rows(T):
    rows = []
    for tid, fr in T.items():
        for f, b in fr.items():
            rows.append([f, tid, b[0], b[1], b[2] - b[0], b[3] - b[1], 1.0, -1, -1, -1])
    rows.sort(key=lambda r: (r[0], r[1]))
    return np.array(rows, dtype=float).reshape(-1, 10)


def resolve_pairs(arr: np.ndarray, p: Optional[DeferParams] = None, judge=None):
    """MOT 배열 -> (수정된 MOT 배열, 결정 로그). 결정은 교차 종료 순서대로 적용한다.
    swap 을 적용하면 이후 교차 구간을 다시 찾는다 (ID 가 바뀌었으므로).
    judge(seg)->gt_id 를 주면 결정마다 GT 판정(verdict)을 기록한다 (결정에는 쓰지 않음)."""
    p = p or DeferParams()
    T = tracks_from_rows(arr)
    score = {(int(r[0]), int(r[1])): float(r[6]) if len(r) > 6 else 1.0 for r in arr}
    logs = []
    done = set()
    while True:
        eps = find_episodes(T, p)
        busy = _busy(eps)
        applied = False
        for e in eps:
            key = (e["a"], e["b"], e["t0"], e["t1"])
            if key in done:
                continue
            done.add(key)
            r = evaluate_episode(T, e, p, busy)
            segs = r.pop("_segs", None)
            if judge is not None and segs is not None:
                ga, gb, ha, hb = (judge(s) for s in segs)
                r.update(gt_pre_a=ga, gt_pre_b=gb, gt_post_a=ha, gt_post_b=hb,
                         verdict=_verdict(ga, gb, ha, hb, r["decision"]))
            logs.append(r)
            if r["decision"] == "swap":
                swap_tracks(T, e["a"], e["b"], e["peak"])
                applied = True
                break
        if not applied:
            break
    return _rows(T), logs
