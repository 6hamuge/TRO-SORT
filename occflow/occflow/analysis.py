"""GT 기준 실패 유형 분석.

ID 변경 이벤트(CLEAR IDSW 와 같은 정의: GT가 직전에 매칭됐던 예측 ID와 다른 ID에 매칭)를 분류한다.

  category 1  즉시 교환   : 새 ID가 '직전까지 다른 GT의 ID'였고, 끊김(FN) 프레임이 swap_max_gap 이하
                            kind=swap(상대 GT도 내 이전 ID를 가져감) / takeover(한쪽만)
  category 2  단절 후 다른 ID : 그 외 (끊김이 길거나, 처음 보는 새 ID로 이어짐)
                            kind=newid / reuse(오래전 다른 GT의 ID) / swap,takeover(끊김이 긴 경우)
  category R  이전 ID 복귀 : 과거에 이 GT가 쓰던 ID로 돌아옴 (IDSW로 세지지만 '회복')
  persistent  오연결 지속 : 새 ID가 persist_min 프레임 이상 또는 GT 끝까지 유지 (1/2와 독립 플래그)

함께 기록: 끊김 구간에서 검출이 있었는지(det_high/det_low/no_det), 가림 상대 GT,
교집합 위치 흐름(GT 박스 기준, 예: 'LB>CB>RB'), 하단 y 기준 앞뒤.
HOTA/AssA 는 TrackEval 로 계산하고, 여기서는 IDF1 전역 배정 기준 '틀린 ID 프레임 수'를 같이 낸다.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import intersection_box, iou_matrix, position_bin, relative_center
from .io_utils import to_frame_dict

EMPTY = (np.zeros(0, int), np.zeros((0, 4)), np.zeros(0))


@dataclass
class AnalysisParams:
    iou_thr: float = 0.5
    swap_max_gap: int = 2      # 이 이하 FN 프레임이면 '즉시'
    swap_window: int = 10      # 새 ID가 다른 GT 소유였던 시점이 이 이내면 교환/탈취
    persist_min: int = 30      # 이 이상 유지되면 '지속'
    ctx_before: int = 15       # 가림 상대 탐색 구간 (이전 매칭 프레임 기준)
    ctx_after: int = 5
    det_high: float = 0.6      # 재현 Hybrid-SORT det_thresh 와 맞출 것
    det_low: float = 0.1


def _compress(tokens: List[str]) -> str:
    out: List[str] = []
    for t in tokens:
        if not out or out[-1] != t:
            out.append(t)
    return ">".join(out)


def det_status(det_frames: Optional[Dict[int, np.ndarray]], frame: int, box, p: AnalysisParams) -> str:
    if det_frames is None:
        return ""
    d = det_frames.get(frame)
    if d is None or len(d) == 0:
        return "no_det"
    ious = iou_matrix(np.asarray(box)[None], d[:, :4])[0]
    ok = ious >= p.iou_thr
    if (ok & (d[:, 4] > p.det_high)).any():
        return "det_high"
    if (ok & (d[:, 4] > p.det_low)).any():
        return "det_low"
    return "no_det"


def analyze_sequence(seq: str, gt_arr: np.ndarray, pr_arr: np.ndarray,
                     det_frames: Optional[Dict[int, np.ndarray]] = None,
                     params: Optional[AnalysisParams] = None) -> dict:
    p = params or AnalysisParams()
    G = to_frame_dict(gt_arr)
    P = to_frame_dict(pr_arr)
    frames = sorted(set(G) | set(P))

    gt_box: Dict[int, Dict[int, np.ndarray]] = defaultdict(dict)
    for f, (ids, boxes, _) in G.items():
        for i, b in zip(ids, boxes):
            gt_box[int(i)][f] = b

    prev_ts: Dict[int, int] = {}                 # 직전 프레임 매칭 (CLEAR 매칭 보너스용)
    last_ever: Dict[int, tuple] = {}             # GT -> (마지막 매칭 pred, frame)
    match: Dict[int, Dict[int, int]] = {}
    fn_frames: Dict[int, List[int]] = defaultdict(list)
    gt_frames: Dict[int, List[int]] = defaultdict(list)
    pred_owner: Dict[int, List[tuple]] = defaultdict(list)   # pred -> [(frame, gt)]
    potential: Counter = Counter()
    segs: Counter = Counter()
    prev_tracked: set = set()
    idsw = []
    tp = fn = fp = 0
    n_gt = n_pr = 0
    fn_pred_overlap = 0

    for f in frames:
        gi, gb, _ = G.get(f, EMPTY)
        pi, pb, _ = P.get(f, EMPTY)
        n_gt += len(gi)
        n_pr += len(pi)
        cur: Dict[int, int] = {}
        iou = None
        if len(gi) and len(pi):
            iou = iou_matrix(gb, pb)
            for a, b in zip(*np.nonzero(iou >= p.iou_thr)):
                potential[(int(gi[a]), int(pi[b]))] += 1
            prevp = np.array([prev_ts.get(int(x), -(10 ** 9)) for x in gi])
            score = np.where(iou >= p.iou_thr - 1e-9, iou + 1000.0 * (prevp[:, None] == pi[None, :]), 0.0)
            r, c = linear_sum_assignment(-score)
            for a, b in zip(r, c):
                if score[a, b] > 0:
                    cur[int(gi[a])] = int(pi[b])
        for k, g in enumerate(gi):
            g = int(g)
            gt_frames[g].append(f)
            if g in cur:
                pid = cur[g]
                if g in last_ever and last_ever[g][0] != pid:
                    idsw.append((g, f, last_ever[g][0], last_ever[g][1], pid))
                last_ever[g] = (pid, f)
                if g not in prev_tracked:
                    segs[g] += 1
                pred_owner[pid].append((f, g))
            else:
                fn_frames[g].append(f)
                if iou is not None and (iou[k] >= p.iou_thr).any():
                    fn_pred_overlap += 1
        prev_tracked = set(cur)
        prev_ts = cur
        match[f] = cur
        tp += len(cur)
        fn += len(gi) - len(cur)
        fp += len(pi) - len(cur)

    # ---- IDF1 전역 배정 (TrackEval Identity 와 같은 방식)
    gids = sorted(gt_frames)
    pids = sorted({k[1] for k in potential} | {int(x) for f in P for x in P[f][0]})
    idtp = 0
    global_map: Dict[int, int] = {}
    if gids and pids and potential:
        gi_of = {g: i for i, g in enumerate(gids)}
        pi_of = {q: i for i, q in enumerate(pids)}
        M = np.zeros((len(gids), len(pids)))
        for (g, q), c in potential.items():
            M[gi_of[g], pi_of[q]] = c
        r, c = linear_sum_assignment(-M)
        for a, b in zip(r, c):
            if M[a, b] > 0:
                global_map[gids[a]] = pids[b]
                idtp += int(M[a, b])

    # ---- GT 별 표
    gts = []
    wrong_total = 0
    for g in gids:
        mf = [(f, match[f][g]) for f in gt_frames[g] if g in match[f]]
        gm = global_map.get(g)
        wrong = sum(1 for _, q in mf if q != gm)
        wrong_total += wrong
        gts.append(dict(seq=seq, gt_id=g, frames=len(gt_frames[g]), matched=len(mf),
                        fn=len(fn_frames[g]), idsw=sum(1 for e in idsw if e[0] == g),
                        frag=max(segs[g] - 1, 0), global_pred=gm if gm is not None else -1,
                        wrong_id_frames=wrong, n_pred_ids=len({q for _, q in mf})))

    # ---- FN 프레임의 검출 유무
    fn_det = Counter()
    if det_frames is not None:
        for g, fl in fn_frames.items():
            for f in fl:
                fn_det[det_status(det_frames, f, gt_box[g][f], p)] += 1

    # ---- 이벤트 분류
    events_by_gt = defaultdict(list)
    for e in idsw:
        events_by_gt[e[0]].append(e)
    events = []
    for g, f, p0, f0, p1 in idsw:
        fn_between = [x for x in fn_frames[g] if f0 < x < f]
        gap_fn = len(fn_between)
        owners_before = [(ff, gg) for ff, gg in pred_owner[p1] if ff < f]
        other_owner = [(ff, gg) for ff, gg in owners_before if gg != g]
        used_by_me_before = any(gg == g for _, gg in owners_before)
        p1_first = pred_owner[p1][0][0] if pred_owner[p1] else f
        prev_owner = other_owner[-1] if other_owner else None
        reciprocal = False
        if prev_owner is not None and f - prev_owner[0] <= p.swap_window:
            g2 = prev_owner[1]
            reciprocal = any(e[4] == p0 and abs(e[1] - f) <= p.swap_window for e in events_by_gt.get(g2, []))
            kind = "swap" if reciprocal else "takeover"
        elif used_by_me_before:
            kind = "return"
        elif prev_owner is not None:
            kind = "reuse"
        else:
            kind = "newid"
        if kind == "return":
            category = "R"
        elif kind in ("swap", "takeover") and gap_fn <= p.swap_max_gap:
            category = "1"
        else:
            category = "2"

        # 지속 길이
        later = [(x, match[x][g]) for x in gt_frames[g] if x >= f and g in match[x]]
        run_len, run_end, changed = 0, f, False
        for x, q in later:
            if q != p1:
                changed = True
                break
            run_len += 1
            run_end = x
        to_end = not changed
        returned_prev = any(q == p0 for x, q in later)

        # 가림 상대 (GT 기준) 와 교집합 위치 흐름
        lo, hi = f0 - p.ctx_before, f + p.ctx_after
        w_frames = [x for x in gt_frames[g] if lo <= x <= hi]
        acc = Counter()
        for x in w_frames:
            ids, boxes, _ = G[x]
            ious = iou_matrix(gt_box[g][x][None], boxes)[0]
            for oid, v in zip(ids, ious):
                if int(oid) != g and v > 0:
                    acc[int(oid)] += float(v)
        partner, partner_iou, flow, front = -1, 0.0, "", ""
        if acc:
            partner = acc.most_common(1)[0][0]
            toks, maxv = [], 0.0
            for x in w_frames:
                pb_ = gt_box[partner].get(x)
                if pb_ is None:
                    continue
                tb = gt_box[g][x]
                maxv = max(maxv, float(iou_matrix(tb[None], pb_[None])[0, 0]))
                inter = intersection_box(tb, pb_)
                toks.append("--" if inter is None else position_bin(*relative_center(tb, inter)))
            partner_iou, flow = maxv, _compress(toks)
            near = [x for x in (f0, f) if x in gt_box[partner]]
            if near:
                x = near[0]
                front = "partner_front" if gt_box[partner][x][3] > gt_box[g][x][3] else "self_front"
        partner_before = match.get(f0, {}).get(partner, -1) if partner >= 0 else -1
        partner_after = -1
        if partner >= 0:
            for x in gt_frames[partner]:
                if x >= f and partner in match[x]:
                    partner_after = match[x][partner]
                    break

        gap_det = Counter(det_status(det_frames, x, gt_box[g][x], p) for x in fn_between) if det_frames else Counter()
        events.append(dict(
            seq=seq, gt_id=g, frame=f, prev_frame=f0, gap_fn=gap_fn, prev_pred=p0, new_pred=p1,
            category=category, kind=kind, new_pred_first_frame=p1_first,
            new_pred_prev_gt=prev_owner[1] if prev_owner else -1,
            new_pred_prev_gt_frame=prev_owner[0] if prev_owner else -1,
            persistent=int(to_end or run_len >= p.persist_min), run_len=run_len, run_end=run_end,
            to_end=int(to_end), returned_to_prev=int(returned_prev),
            gap_det_high=gap_det.get("det_high", 0), gap_det_low=gap_det.get("det_low", 0),
            gap_no_det=gap_det.get("no_det", 0),
            partner_gt=partner, partner_max_iou=round(partner_iou, 3), overlap_flow=flow, depth=front,
            partner_pred_before=partner_before, partner_pred_after=partner_after,
        ))

    cat = Counter(e["category"] for e in events)
    kinds = Counter(e["kind"] for e in events)
    summary = dict(
        seq=seq, gt_dets=n_gt, pred_dets=n_pr, TP=tp, FN=fn, FP=fp, IDSW=len(idsw),
        Frag=sum(max(v - 1, 0) for v in segs.values()),
        MOTA=round(100 * (1 - (fn + fp + len(idsw)) / max(n_gt, 1)), 3),
        IDTP=idtp, IDF1=round(100 * 2 * idtp / max(n_gt + n_pr, 1), 3),
        cat1_immediate=cat.get("1", 0), cat2_break=cat.get("2", 0), catR_return=cat.get("R", 0),
        cat3_persistent=sum(e["persistent"] for e in events),
        persistent_frames=sum(e["run_len"] for e in events if e["persistent"]),
        swap=kinds.get("swap", 0), takeover=kinds.get("takeover", 0), newid=kinds.get("newid", 0),
        reuse=kinds.get("reuse", 0), wrong_id_frames=wrong_total,
        fn_det_high=fn_det.get("det_high", 0), fn_det_low=fn_det.get("det_low", 0),
        fn_no_det=fn_det.get("no_det", 0), fn_pred_overlap=fn_pred_overlap,
    )
    return dict(summary=summary, events=events, gts=gts)


SUM_KEYS = ("gt_dets", "pred_dets", "TP", "FN", "FP", "IDSW", "Frag", "IDTP", "cat1_immediate",
            "cat2_break", "catR_return", "cat3_persistent", "persistent_frames", "swap", "takeover",
            "newid", "reuse", "wrong_id_frames", "fn_det_high", "fn_det_low", "fn_no_det", "fn_pred_overlap")


def combine(summaries: List[dict]) -> dict:
    tot = {k: sum(s[k] for s in summaries) for k in SUM_KEYS}
    tot["seq"] = "COMBINED"
    tot["MOTA"] = round(100 * (1 - (tot["FN"] + tot["FP"] + tot["IDSW"]) / max(tot["gt_dets"], 1)), 3)
    tot["IDF1"] = round(100 * 2 * tot["IDTP"] / max(tot["gt_dets"] + tot["pred_dets"], 1), 3)
    return tot


def compare_events(ev_a: List[dict], ev_b: List[dict], tol: int = 5) -> List[dict]:
    """같은 (seq, gt_id) 에서 프레임 차이 tol 이내 이벤트를 짝지어 only_a / only_b / both 표시."""
    def by_key(evs):
        d = defaultdict(list)
        for e in evs:
            d[(e["seq"], e["gt_id"])].append(e)
        return d
    A, B = by_key(ev_a), by_key(ev_b)
    rows = []
    for key in sorted(set(A) | set(B)):
        la, lb = sorted(A.get(key, []), key=lambda e: e["frame"]), sorted(B.get(key, []), key=lambda e: e["frame"])
        used = set()
        for ea in la:
            best, bd = None, tol + 1
            for j, eb in enumerate(lb):
                if j in used:
                    continue
                dd = abs(eb["frame"] - ea["frame"])
                if dd < bd:
                    best, bd = j, dd
            if best is not None:
                used.add(best)
                eb = lb[best]
                rows.append(dict(status="both", seq=key[0], gt_id=key[1], frame_a=ea["frame"], frame_b=eb["frame"],
                                 cat_a=ea["category"], cat_b=eb["category"], kind_a=ea["kind"], kind_b=eb["kind"],
                                 run_a=ea["run_len"], run_b=eb["run_len"], overlap_flow=ea["overlap_flow"]))
            else:
                rows.append(dict(status="only_a", seq=key[0], gt_id=key[1], frame_a=ea["frame"], frame_b="",
                                 cat_a=ea["category"], cat_b="", kind_a=ea["kind"], kind_b="",
                                 run_a=ea["run_len"], run_b="", overlap_flow=ea["overlap_flow"]))
        for j, eb in enumerate(lb):
            if j not in used:
                rows.append(dict(status="only_b", seq=key[0], gt_id=key[1], frame_a="", frame_b=eb["frame"],
                                 cat_a="", cat_b=eb["category"], kind_a="", kind_b=eb["kind"],
                                 run_a="", run_b=eb["run_len"], overlap_flow=eb["overlap_flow"]))
    return rows
