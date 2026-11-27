"""추적기에 끼우는 가림 흐름 훅. 트랙 ID 기준으로 이력을 관리하므로 어떤 SORT 계열에도 붙일 수 있다.

호출 순서 (프레임마다):
  1) adjust()   : 1차 매칭 비용 행렬 보정            -> 실패 유형 1 (교차 직후 즉시 교환)
  2) recover()  : 새 ID 생성 전, 가림 중 놓친 트랙 재연결 -> 실패 유형 2 (끊겼다가 다른 ID)
  3) audit()    : 매칭 후 교차 쌍의 ID 사후 점검        -> 실패 유형 3 (틀린 ID가 끝까지 감)
  4) observe()  : 신뢰할 수 있는 관측만 이력에 추가
  5) prune()    : 사라진 트랙 이력 삭제
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import AuditConfig, FlowConfig, RecoveryConfig
from .geometry import center, coverage, iou_matrix, wh
from .history import OcclusionHistory, flow_mismatch
from .occlusion import observe_pair


class OcclusionFlowHook:
    def __init__(self, flow_cfg: Optional[FlowConfig] = None,
                 rec_cfg: Optional[RecoveryConfig] = None,
                 audit_cfg: Optional[AuditConfig] = None):
        self.cfg = flow_cfg or FlowConfig()
        self.rec = rec_cfg or RecoveryConfig()
        self.aud = audit_cfg or AuditConfig()
        self.histories: Dict[int, OcclusionHistory] = {}
        self._audit_count: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self._audit_cooldown: Dict[Tuple[int, int], int] = {}
        self.stats = Counter()

    # ------------------------------------------------------------------ utils
    def history(self, tid: int) -> OcclusionHistory:
        h = self.histories.get(tid)
        if h is None:
            h = OcclusionHistory(self.cfg.hist_len)
            self.histories[tid] = h
        return h

    def prune(self, alive_ids: Iterable[int]) -> None:
        alive = set(int(x) for x in alive_ids)
        for tid in list(self.histories):
            if tid not in alive:
                del self.histories[tid]
        for d in (self._audit_count, self._audit_cooldown):
            for key in list(d):
                if key[0] not in alive or key[1] not in alive:
                    del d[key]

    def _skip(self, logs: list, base: dict, reason: str) -> None:
        self.stats["skip_" + reason] += 1
        if self.cfg.log_all:
            logs.append(dict(base, det_idx=-1, base_cost=np.nan, mismatch=np.nan, adj=0.0,
                             final_cost=np.nan, reliability=np.nan, consistency=np.nan,
                             active=0, reason=reason))

    def _near_best(self, row: np.ndarray) -> List[int]:
        finite = np.isfinite(row)
        if not finite.any():
            return []
        best = row[finite].min()
        return [int(j) for j in np.flatnonzero(finite & (row <= best + self.cfg.ambig_margin))]

    @staticmethod
    def _next_best(row: np.ndarray, exclude: Sequence[int]) -> Optional[int]:
        for j in np.argsort(row, kind="stable"):
            if np.isfinite(row[j]) and int(j) not in exclude:
                return int(j)
        return None

    @staticmethod
    def _partner_box(pr: int, j: int, masked: np.ndarray, det_boxes: np.ndarray,
                     track_boxes: np.ndarray):
        """후보 j를 대상 트랙이 가져간다고 가정할 때 상대 트랙이 있을 위치.
        상대의 (j를 제외한) 최선 후보 검출을 쓰고, 없으면 상대의 예측 박스를 신뢰도 0.5로 쓴다."""
        row = masked[pr].copy()
        row[j] = np.inf
        if np.isfinite(row).any():
            return det_boxes[int(np.argmin(row))], 1.0
        return track_boxes[pr], 0.5

    # ----------------------------------------------------------------- adjust
    def adjust(self, frame: int, track_ids: Sequence[int], track_boxes, det_boxes,
               cost, valid=None):
        """비용 행렬 (트랙 x 검출, 낮을수록 좋음)을 보정한다.

        - 이력이 있고, 경합 후보가 2개 이상이고, 흐름 신뢰도가 충분한 트랙 행에만 적용.
        - 행 안에서 '가장 흐름에 맞는 후보' 대비 상대적인 불일치만 더한다
          (이력 유무 자체로 우대/불이익이 생기지 않도록).
        - 조건이 안 맞으면 원래 비용 그대로 = 기존 추적기 판단.
        """
        cost = np.asarray(cost, dtype=float)
        out = cost.copy()
        logs: List[dict] = []
        if not self.cfg.enable or cost.size == 0:
            return out, logs
        n_t, n_d = cost.shape
        if valid is None:
            valid = np.isfinite(cost)
        masked = np.where(valid, cost, np.inf)
        track_ids = [int(t) for t in track_ids]
        row_of = {t: i for i, t in enumerate(track_ids)}
        det_boxes = np.asarray(det_boxes, dtype=float).reshape(-1, 4)
        track_boxes = np.asarray(track_boxes, dtype=float).reshape(-1, 4)
        cands = [self._near_best(masked[i]) for i in range(n_t)]

        for i, tid in enumerate(track_ids):
            h = self.histories.get(tid)
            if h is None or h.partner_id is None:
                continue
            base = dict(frame=frame, track_id=tid, partner_id=h.partner_id, row=i)
            pred = h.predict(frame, self.cfg)
            if pred is None:
                self._skip(logs, base, "hist_" + h.unusable_reason(frame, self.cfg))
                continue
            pr = row_of.get(h.partner_id)
            if pr is None:
                self._skip(logs, base, "partner_absent")
                continue
            C = list(cands[i])
            if len(C) == 1 and C[0] in cands[pr]:
                # 상대와 같은 검출을 다투는 경우: 대상의 차선 후보까지 비교
                extra = self._next_best(masked[i], C)
                if extra is not None:
                    C.append(extra)
            if len(C) < 2:
                self._skip(logs, base, "unambiguous")
                continue
            rel = pred.consistency
            if self.cfg.use_depth:
                rel *= 0.5 + 0.5 * pred.depth_conf
            mism = []
            partner_w = 1.0
            for j in C:
                pbox, pw = self._partner_box(pr, j, masked, det_boxes, track_boxes)
                partner_w = min(partner_w, pw)
                mism.append(flow_mismatch(pred, det_boxes[j], pbox))
            rel *= partner_w
            if rel < self.cfg.min_reliability:
                self._skip(logs, base, "low_reliability")
                continue
            fmin = min(mism)
            self.stats["rows_applied"] += 1
            for j, m in zip(C, mism):
                a = self.cfg.weight * rel * (m - fmin)
                out[i, j] += a
                logs.append(dict(base, det_idx=int(j), base_cost=float(cost[i, j]),
                                 mismatch=float(m), adj=float(a), final_cost=float(out[i, j]),
                                 reliability=float(rel), consistency=float(pred.consistency),
                                 active=1, reason="applied"))
        return out, logs

    # ---------------------------------------------------------------- observe
    def _within_exit(self, h: OcclusionHistory, frame: int) -> bool:
        return h.last_overlap_frame is not None and frame - h.last_overlap_frame <= self.cfg.exit_memory

    def _select_partner(self, tid: int, box, boxes: Dict[int, np.ndarray],
                        h: OcclusionHistory, frame: int) -> Optional[int]:
        """가장 크게 겹치는 상대. 현재 상대는 흔들림 방지를 위해 우선 유지하고,
        겹침이 막 끝난 상대도 exit_memory 동안 유지한다 (갈라진 직후가 가장 중요한 순간)."""
        best, best_cov = None, 0.0
        for oid, ob in boxes.items():
            if oid == tid:
                continue
            c = coverage(box, ob)
            if c > best_cov:
                best, best_cov = oid, c
        cur = h.partner_id
        if cur is not None and cur in boxes:
            cur_cov = coverage(box, boxes[cur])
            if cur_cov >= self.cfg.min_cov:
                if (best != cur and best_cov >= self.cfg.min_cov
                        and best_cov > cur_cov * self.cfg.partner_switch_ratio):
                    return best
                return cur
            if best_cov < self.cfg.min_cov and self._within_exit(h, frame):
                return cur
        return best if best_cov >= self.cfg.min_cov else None

    def observe(self, frame: int, entries: Iterable[Tuple[int, object, bool]]) -> None:
        """entries: (track_id, 이번 프레임 관측 박스, trusted). 매칭 확정 후 프레임당 1회 호출.

        - trusted 가 아닌 관측(저신뢰 검출, 재연결 직후 등)은 넣지 않는다.
        - 상대도 trusted 로 관측된 경우에만 쌍 관측을 만든다 (예측 위치를 관측처럼 쌓지 않음).
        - 기존 흐름과 크게 어긋나는 관측은 보류(pending)하고, 오래 쌓이면 새 흐름으로 재시작.
        """
        entries = [(int(t), np.asarray(b, dtype=float), bool(ok)) for t, b, ok in entries]
        boxes = {t: b for t, b, _ in entries}
        trusted = {t for t, _, ok in entries if ok}
        for tid, box, ok in entries:
            if not ok:
                continue
            h = self.history(tid)
            partner = self._select_partner(tid, box, boxes, h, frame)
            if partner is None:
                if h.partner_id is not None and h.partner_id in boxes and not self._within_exit(h, frame):
                    h.reset()
                continue
            if partner not in trusted:
                continue
            obs = observe_pair(frame, box, boxes[partner], partner, self.cfg.depth_margin)
            if partner == h.partner_id:
                pred = h.predict(frame, self.cfg)
                if pred is not None and flow_mismatch(pred, box, boxes[partner]) > self.cfg.trust_mismatch:
                    h.pending += 1
                    self.stats["obs_rejected"] += 1
                    if h.pending > self.cfg.max_pending:
                        h.reset(partner)
                        h.add(obs)
                        self.stats["hist_restart"] += 1
                    continue
            h.add(obs)

    # ---------------------------------------------------------------- recover
    def recover(self, frame: int, lost: Sequence[Tuple[int, object, int]], det_boxes,
                matched_boxes: Dict[int, object]) -> List[Tuple[int, int, float]]:
        """가림 상대와 겹친 채 놓친 트랙을, 상대 기준 흐름이 예상하는 위치의 미매칭 검출과 재연결.

        lost          : (track_id, 마지막 관측 박스, 놓친 프레임 수)
        det_boxes     : 이번 프레임 미매칭 고신뢰 검출 (K, 4)
        matched_boxes : 이번 프레임 이미 매칭된 트랙 {id: box} (상대 위치로 사용)
        반환          : (lost 인덱스, det 인덱스, 불일치)
        """
        if not self.rec.enable or not len(lost) or len(det_boxes) == 0:
            return []
        det_boxes = np.asarray(det_boxes, dtype=float).reshape(-1, 4)
        cost = np.full((len(lost), len(det_boxes)), np.inf)
        st = self.stats
        for r, (tid, last_box, tsu) in enumerate(lost):
            if tsu > self.rec.max_lost:
                st["rec_skip_too_old"] += 1
                continue
            h = self.histories.get(int(tid))
            if h is None or h.partner_id is None or h.last_overlap_frame is None or not h.entries:
                st["rec_skip_no_history"] += 1
                continue
            if h.entries[-1].frame - h.last_overlap_frame > self.cfg.exit_memory:
                st["rec_skip_not_occlusion"] += 1
                continue  # 가림 상황에서 놓친 트랙만 대상
            pbox = matched_boxes.get(h.partner_id)
            if pbox is None:
                st["rec_skip_partner_unmatched"] += 1
                continue
            pred = h.predict(frame, self.cfg, max_extrap=self.rec.max_extrap, max_stale=self.rec.max_lost)
            if pred is None:
                st["rec_skip_history_unusable"] += 1
                continue
            if pred.consistency < self.rec.min_consistency:
                st["rec_skip_low_consistency"] += 1
                continue
            st["rec_considered"] += 1
            last_box = np.asarray(last_box, dtype=float)
            lw, lh = wh(last_box)
            pcx, pcy = center(pbox)
            ex, ey = pcx - pred.dx * lw, pcy - pred.dy * lh
            expected = np.array([ex - lw / 2, ey - lh / 2, ex + lw / 2, ey + lh / 2])
            ious = iou_matrix(expected[None], det_boxes)[0]
            for c, d in enumerate(det_boxes):
                dh = wh(d)[1]
                if max(dh / lh, lh / dh) > self.rec.size_ratio or ious[c] < self.rec.min_expected_iou:
                    continue
                m = flow_mismatch(pred, d, pbox)
                if m <= self.rec.max_mismatch:
                    cost[r, c] = m
        if not np.isfinite(cost).any():
            return []
        rr, cc = linear_sum_assignment(np.where(np.isfinite(cost), cost, 1e6))
        out = [(int(a), int(b), float(cost[a, b])) for a, b in zip(rr, cc) if np.isfinite(cost[a, b])]
        self.stats["recovered"] += len(out)
        return out

    # ------------------------------------------------------------------ audit
    def audit(self, frame: int, matched_boxes: Dict[int, object]) -> List[Tuple[int, int, int, float, float]]:
        """서로를 가림 상대로 가진 두 트랙에 대해, 현재 배정과 ID를 맞바꾼 배정 중
        어느 쪽이 교차 이전 흐름과 맞는지 비교한다. 교환 쪽이 confirm_frames 연속으로
        확실히 맞으면 교환을 제안한다. (적어도 한쪽 이력이 '보류' 상태일 때만 = 최근 관측이 흐름과 어긋남)

        반환: (id_a, id_b, 오류 시작 추정 프레임, 현재 불일치, 교환 불일치)
        """
        if not self.aud.enable:
            return []
        swaps = []
        seen = set()
        stale = self.cfg.max_stale + self.aud.confirm_frames
        mb = {int(k): np.asarray(v, dtype=float) for k, v in matched_boxes.items()}
        for a in sorted(mb):
            ha = self.histories.get(a)
            if ha is None or ha.partner_id is None:
                continue
            b = ha.partner_id
            key = (min(a, b), max(a, b))
            if key in seen or b not in mb:
                continue
            seen.add(key)
            hb = self.histories.get(b)
            if hb is None or hb.partner_id != a:
                self._audit_count.pop(key, None)
                continue
            if self._audit_cooldown.get(key, -1) >= frame:
                continue
            if ha.pending == 0 and hb.pending == 0:
                self._audit_count.pop(key, None)
                continue
            pa = ha.predict(frame, self.cfg, max_stale=stale)
            pb = hb.predict(frame, self.cfg, max_stale=stale)
            if pa is None or pb is None or min(pa.consistency, pb.consistency) < self.cfg.min_consistency:
                self._audit_count.pop(key, None)
                continue
            A, B = mb[a], mb[b]
            cur = flow_mismatch(pa, A, B) + flow_mismatch(pb, B, A)
            swp = flow_mismatch(pa, B, A) + flow_mismatch(pb, A, B)
            if cur - swp > self.aud.margin:
                start = min(ha.last_frame, hb.last_frame) + 1
                cnt, start0 = self._audit_count.get(key, (0, start))
                cnt += 1
                if cnt >= self.aud.confirm_frames:
                    swaps.append((a, b, start0, float(cur), float(swp)))
                    self._audit_count.pop(key, None)
                    self._audit_cooldown[key] = frame + self.aud.cooldown
                    ha.pending = hb.pending = 0
                    self.stats["audit_swaps"] += 1
                else:
                    self._audit_count[key] = (cnt, start0)
            else:
                self._audit_count.pop(key, None)
        return swaps
