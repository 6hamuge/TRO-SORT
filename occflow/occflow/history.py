"""트랙별 가림 이력과 흐름 예측.

원칙 (계획서 '불확실한 가림 정보에 대한 대응'):
- 가림 상대가 바뀌면 이전 이력을 이어 쓰지 않고 새 흐름으로 시작
- 미검출 기간의 예측 위치는 이력에 넣지 않음 (확정 관측만 add)
- 이력이 오래되었거나(max_stale) 겹침이 끝난 지 오래면(exit_memory) 활용 보류
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .geometry import intersection_box, relative_center, relative_offset
from .occlusion import OverlapObservation, OverlapState


@dataclass
class FlowPrediction:
    dx: float
    dy: float
    sx: float
    sy: float
    u: Optional[float]
    v: Optional[float]
    su: float
    sv: float
    consistency: float   # 흐름 방향 일관성 x 이동량 강도 (0~1)
    depth_conf: float    # 이력 구간의 평균 앞뒤 신뢰도
    n: int
    gap: int             # 마지막 확정 관측 이후 프레임 수
    last_frame: int


def _linfit(x: np.ndarray, y: np.ndarray, t: float) -> Tuple[float, float]:
    """1차 최소제곱으로 t 시점 값 예측. 점이 부족하면 마지막 값 유지."""
    n = len(x)
    if n >= 3 and (x.max() - x.min()) >= 2:
        xm, ym = x.mean(), y.mean()
        sxx = ((x - xm) ** 2).sum()
        slope = ((x - xm) * (y - ym)).sum() / sxx
        fit = ym + slope * (x - xm)
        resid = np.sqrt(((y - fit) ** 2).sum() / max(n - 2, 1))
        return float(ym + slope * (t - xm)), float(resid)
    return float(y[-1]), (float(y.std()) if n >= 2 else 0.0)


class OcclusionHistory:
    def __init__(self, max_len: int = 10):
        self.entries = deque(maxlen=max_len)
        self.partner_id: Optional[int] = None
        self.last_overlap_frame: Optional[int] = None
        self.pending = 0  # 이력과 어긋나 보류된 연속 관측 수

    def reset(self, partner_id: Optional[int] = None) -> None:
        self.entries.clear()
        self.partner_id = partner_id
        self.last_overlap_frame = None
        self.pending = 0

    def add(self, obs: OverlapObservation) -> None:
        if self.partner_id is not None and obs.partner_id != self.partner_id:
            self.reset(obs.partner_id)
        self.partner_id = obs.partner_id
        self.entries.append(obs)
        if obs.state == OverlapState.OVERLAP:
            self.last_overlap_frame = obs.frame
        self.pending = 0

    @property
    def last_frame(self) -> Optional[int]:
        return self.entries[-1].frame if self.entries else None

    @property
    def n_overlap(self) -> int:
        return sum(1 for e in self.entries if e.state == OverlapState.OVERLAP)

    def unusable_reason(self, frame: int, cfg, max_stale: Optional[int] = None) -> str:
        """predict() 가 None 인 이유 (진단 로그용)."""
        if self.partner_id is None:
            return "no_partner"
        if len(self.entries) < cfg.min_hist:
            return "hist_short"
        if self.n_overlap < cfg.min_overlap_entries:
            return "few_overlap"
        gap = frame - self.entries[-1].frame
        if gap < 0 or gap > (cfg.max_stale if max_stale is None else max_stale):
            return "stale"
        if self.last_overlap_frame is None or self.entries[-1].frame - self.last_overlap_frame > cfg.exit_memory:
            return "overlap_ended"
        return "ok"

    def predict(self, frame: int, cfg, max_extrap: Optional[int] = None,
                max_stale: Optional[int] = None) -> Optional[FlowPrediction]:
        """frame 시점의 상대 위치를 이력 흐름으로 예측. 쓸 수 없으면 None (= 기존 추적기 판단을 따름)."""
        if self.partner_id is None or len(self.entries) < cfg.min_hist:
            return None
        if self.n_overlap < cfg.min_overlap_entries:
            return None
        last = self.entries[-1].frame
        gap = frame - last
        stale = cfg.max_stale if max_stale is None else max_stale
        if gap < 0 or gap > stale:
            return None
        if self.last_overlap_frame is None or last - self.last_overlap_frame > cfg.exit_memory:
            return None

        E = list(self.entries)[-cfg.fit_len:]
        t = float(frame if max_extrap is None else min(frame, last + max_extrap))
        fs = np.array([e.frame for e in E], dtype=float)
        dxs = np.array([e.dx for e in E])
        dys = np.array([e.dy for e in E])
        dx, rx = _linfit(fs, dxs, t)
        dy, ry = _linfit(fs, dys, t)
        grow = 1.0 + cfg.sigma_growth * max(gap, 0)
        sx = max(cfg.sigma_floor, rx) * grow
        sy = max(cfg.sigma_floor, ry) * grow

        ov = [e for e in E if e.state == OverlapState.OVERLAP]
        u = v = None
        su = sv = 0.0
        if len(ov) >= 2:
            fu = np.array([e.frame for e in ov], dtype=float)
            u, ru = _linfit(fu, np.array([e.u for e in ov]), t)
            v, rv = _linfit(fu, np.array([e.v for e in ov]), t)
            u = float(np.clip(u, -0.25, 1.25))
            v = float(np.clip(v, -0.25, 1.25))
            su = max(cfg.sigma_floor, ru) * grow
            sv = max(cfg.sigma_floor, rv) * grow

        pts = np.stack([dxs, dys], axis=1)
        consistency = 0.0
        if len(pts) >= 2:
            tot = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
            net = float(np.linalg.norm(pts[-1] - pts[0]))
            c = net / tot if tot > 1e-9 else 0.0
            strength = min(1.0, net / max(cfg.min_disp, 1e-9))
            consistency = c * strength
        all_ov = [e for e in self.entries if e.state == OverlapState.OVERLAP]
        depth_conf = float(np.mean([e.depth_conf for e in all_ov])) if all_ov else 0.0
        return FlowPrediction(dx, dy, sx, sy, u, v, su, sv, consistency, depth_conf,
                              len(E), gap, last)


def flow_mismatch(pred: FlowPrediction, cand_box, partner_box) -> float:
    """후보 박스가 가림 흐름과 얼마나 어긋나는지 (0=완전 일치, 1=크게 어긋남).

    - 항상: 상대 중심 오프셋 (갈라진 직후에도 정의됨)
    - 후보가 상대와 겹치면: 교집합 중심의 후보 박스 내 상대 위치도 함께 비교
    """
    dx, dy = relative_offset(cand_box, partner_box)
    z2 = ((dx - pred.dx) / pred.sx) ** 2 + ((dy - pred.dy) / pred.sy) ** 2
    if pred.u is not None:
        inter = intersection_box(cand_box, partner_box)
        if inter is not None:
            u, v = relative_center(cand_box, inter)
            zu = ((u - pred.u) / pred.su) ** 2 + ((v - pred.v) / pred.sv) ** 2
            z2 = 0.5 * (z2 + zu)
    return float(1.0 - np.exp(-0.25 * z2))
