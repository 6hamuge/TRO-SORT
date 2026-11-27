"""오라클 실험: GT 박스만으로 가설을 검증한다 (추적기 잡음 없음).

겹쳐 있거나 막 갈라진 GT 쌍 (a, b) 에 대해, 프레임 f 에서
  정답 배정 (a->A, b->B) 과 교환 배정 (a->B, b->A) 중 어느 쪽이 맞는지를
  - flow   : 직전 프레임까지 GT 로 만든 가림 이력의 흐름 불일치 합 (occflow.hook 과 같은 계산)
  - motion : 각자 자기 중심 궤적을 1차 회귀로 연장한 예측 위치와의 거리 합 (박스 크기 정규화)
로 고르게 하고 정확도를 비교한다. '어려운 프레임' = 두 박스 IoU 가 hard_iou 이상.

  python tools/oracle_flow.py --gt_root VAL --seqs dancetrack0026 dancetrack0079
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from occflow.config import FlowConfig  # noqa: E402
from occflow.geometry import center, iou_matrix, wh  # noqa: E402
from occflow.history import flow_mismatch  # noqa: E402
from occflow.hook import OcclusionFlowHook  # noqa: E402
from occflow.io_utils import list_sequences, load_gt, to_frame_dict  # noqa: E402


def motion_pred(track, f, k):
    last = track[-1][0] if track else f
    pts = [(x, center(b)) for x, b in track if last - k < x <= last]
    if len(pts) < 3:
        return None
    xs = np.array([p[0] for p in pts], float)
    out = []
    for d in (0, 1):
        ys = np.array([p[1][d] for p in pts])
        A = np.vstack([xs, np.ones_like(xs)]).T
        s, c = np.linalg.lstsq(A, ys, rcond=None)[0]
        out.append(s * f + c)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt_root", required=True)
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--hard_iou", type=float, default=0.1)
    ap.add_argument("--k", type=int, default=6, help="motion 예측에 쓰는 과거 프레임 수 (flow fit_len 과 같게)")
    ap.add_argument("--set", nargs="*", default=[], help="FlowConfig 덮어쓰기, 예: fit_len=4 sigma_floor=0.05")
    ap.add_argument("--gap", type=int, default=1, help="마지막 관측과 판단 시점 사이 프레임 수 (가림 공백 모사)")
    ap.add_argument("--noise", type=float, default=0.0, help="박스 좌표 잡음 (박스 크기 대비 표준편차)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.RandomState(a.seed)

    # 오라클: GT 관측은 모두 이력에 넣고, 공백만큼 오래된 이력도 쓰게 한다
    cfg = FlowConfig(trust_mismatch=1.01, max_stale=max(5, a.gap + 1), exit_memory=max(10, a.gap + 5))
    for kv in a.set:
        k, v = kv.split("=", 1)
        cur = getattr(cfg, k)
        setattr(cfg, k, type(cur)(v) if not isinstance(cur, bool) else v.lower() in ("1", "true"))
    a.k = cfg.fit_len if a.k is None else a.k

    total = defaultdict(lambda: np.zeros(6, int))   # n, flow_ok, motion_ok, disagree, flow_wins, motion_wins
    for seq in a.seqs or list_sequences(a.gt_root):
        G = to_frame_dict(load_gt(os.path.join(a.gt_root, seq, "gt", "gt.txt")))
        if a.noise > 0:                             # 관측 잡음 (판단 프레임 박스에도 동일 적용)
            for f in G:
                ids, boxes, sc = G[f]
                w = (boxes[:, 2] - boxes[:, 0])[:, None]
                h = (boxes[:, 3] - boxes[:, 1])[:, None]
                G[f] = (ids, boxes + rng.normal(0, a.noise, boxes.shape) * np.hstack([w, h, w, h]), sc)
        hook = OcclusionFlowHook(cfg)
        tracks = defaultdict(list)
        stat = defaultdict(lambda: np.zeros(6, int))
        frames = sorted(G)
        for f in frames:
            # 이력/궤적은 f-gap 까지만 관측된 상태로 판단
            fo_ = f - a.gap
            if fo_ in G:
                oi, ob, _ = G[fo_]
                obs = {int(i): b for i, b in zip(oi, ob)}
                hook.observe(fo_, [(i, b, True) for i, b in obs.items()])
                for i, b in obs.items():
                    tracks[i].append((fo_, b))
            ids, boxes, _ = G[f]
            bx = {int(i): b for i, b in zip(ids, boxes)}
            seen = set()
            for i in bx:
                h = hook.histories.get(i)
                if h is None or h.partner_id is None or h.partner_id not in bx:
                    continue
                j = h.partner_id
                key = (min(i, j), max(i, j))
                hj = hook.histories.get(j)
                if key in seen or hj is None or hj.partner_id != i:
                    continue
                seen.add(key)
                A, B = bx[i], bx[j]
                pi, pj = h.predict(f, cfg), hj.predict(f, cfg)
                mi, mj = motion_pred(tracks[i], f, a.k), motion_pred(tracks[j], f, a.k)
                if pi is None or pj is None or mi is None or mj is None:
                    continue
                cur = flow_mismatch(pi, A, B) + flow_mismatch(pj, B, A)
                swp = flow_mismatch(pi, B, A) + flow_mismatch(pj, A, B)
                if abs(cur - swp) < 1e-6:
                    continue                        # 흐름이 구분 못 함 (동률) -> 판단 보류
                sa, sb = np.mean(wh(A)), np.mean(wh(B))
                cA, cB = np.array(center(A)), np.array(center(B))
                mcur = np.linalg.norm(mi - cA) / sa + np.linalg.norm(mj - cB) / sb
                mswp = np.linalg.norm(mi - cB) / sa + np.linalg.norm(mj - cA) / sb
                iou = float(iou_matrix(A[None], B[None])[0, 0])
                tag = "hard" if iou >= a.hard_iou else "easy"
                fo, mo = cur < swp, mcur < mswp
                for t in (tag, "all"):
                    s = stat[t]
                    s += np.array([1, fo, mo, fo != mo, fo and not mo, mo and not fo])
        for t in ("hard", "easy", "all"):
            s = stat[t]
            total[t] += s
            if s[0]:
                print("%-15s %-4s n=%-6d flow_acc=%.3f motion_acc=%.3f disagree=%d (flow wins %d / motion wins %d)"
                      % (seq, t, s[0], s[1] / s[0], s[2] / s[0], s[3], s[4], s[5]))
    print()
    for t in ("hard", "easy", "all"):
        s = total[t]
        if s[0]:
            print("%-15s %-4s n=%-6d flow_acc=%.3f motion_acc=%.3f disagree=%d (flow wins %d / motion wins %d)"
                  % ("ALL", t, s[0], s[1] / s[0], s[2] / s[0], s[3], s[4], s[5]))


if __name__ == "__main__":
    main()
