"""가림 관측: 두 박스의 교집합을 가림 대리 영역으로 삼아 대상 박스 기준 위치를 기록."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .geometry import (area, intersection_box, position_bin, relative_center,
                       relative_offset)


class OverlapState:
    OVERLAP = "overlap"          # 상대와 겹침 관측됨
    NO_OVERLAP = "no_overlap"    # 상대는 관측되지만 겹침 없음 (갈라진 직후 등)
    UNOBSERVED = "unobserved"    # 관측 자체가 없음 (미검출) - 이력에 넣지 않음


def depth_order(target, other, margin: float = 0.1) -> Tuple[int, float]:
    """OA-SORT식 앞뒤 추정: 박스 하단(y2)이 더 아래인 쪽이 카메라에 가깝다고 가정.

    반환 (sign, conf): sign=+1 이면 other 가 앞(대상이 가려짐), -1 이면 대상이 앞.
    conf 는 하단 차이 / 평균 높이 를 margin 으로 나눠 0~1로 자른 값.
    하체 가림·점프(논문 §5.4) 상황에서는 하단 차이가 작거나 틀릴 수 있어 conf 로 의존도를 낮춘다.
    """
    h = 0.5 * ((target[3] - target[1]) + (other[3] - other[1]))
    d = (other[3] - target[3]) / max(h, 1e-6)
    sign = 1 if d > 0 else (-1 if d < 0 else 0)
    conf = min(1.0, abs(d) / margin) if margin > 0 else 1.0
    return sign, float(conf)


@dataclass
class OverlapObservation:
    frame: int
    partner_id: int
    state: str
    dx: float                    # 상대 중심의 상대 오프셋 (대상 폭 단위)
    dy: float
    u: Optional[float] = None    # 교집합 중심의 대상 박스 내 상대 위치 (겹칠 때만)
    v: Optional[float] = None
    coverage: float = 0.0        # 교집합 / 대상 면적
    bin: str = "--"              # 'LB','CB','RB' ... (겹침 없음 '--')
    depth_sign: int = 0
    depth_conf: float = 0.0


def observe_pair(frame: int, target_box, partner_box, partner_id: int,
                 depth_margin: float = 0.1) -> OverlapObservation:
    dx, dy = relative_offset(target_box, partner_box)
    sign, conf = depth_order(target_box, partner_box, depth_margin)
    inter = intersection_box(target_box, partner_box)
    if inter is None:
        return OverlapObservation(frame, int(partner_id), OverlapState.NO_OVERLAP, dx, dy,
                                  depth_sign=sign, depth_conf=conf)
    u, v = relative_center(target_box, inter)
    cov = float(area(inter) / max(float(area(target_box)), 1e-9))
    return OverlapObservation(frame, int(partner_id), OverlapState.OVERLAP, dx, dy, u, v, cov,
                              position_bin(u, v), sign, conf)
