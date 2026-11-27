"""박스 기하 함수. 박스는 모두 tlbr (x1, y1, x2, y2)."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def as_boxes(b) -> np.ndarray:
    return np.asarray(b, dtype=float).reshape(-1, 4)


def area(b) -> np.ndarray:
    b = np.asarray(b, dtype=float)
    return np.maximum(0.0, b[..., 2] - b[..., 0]) * np.maximum(0.0, b[..., 3] - b[..., 1])


def inter_matrix(a, b) -> np.ndarray:
    a = as_boxes(a)
    b = as_boxes(b)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    return np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)


def iou_matrix(a, b) -> np.ndarray:
    a = as_boxes(a)
    b = as_boxes(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    inter = inter_matrix(a, b)
    union = area(a)[:, None] + area(b)[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def intersection_box(a, b) -> Optional[np.ndarray]:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return np.array([x1, y1, x2, y2], dtype=float)


def coverage(target, other) -> float:
    """대상 박스 면적 중 other 와 겹치는 비율 (가림 정도의 대리값)."""
    inter = intersection_box(target, other)
    if inter is None:
        return 0.0
    return float(area(inter) / max(float(area(target)), 1e-9))


def center(b) -> Tuple[float, float]:
    return (0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3]))


def wh(b) -> Tuple[float, float]:
    return (max(b[2] - b[0], 1e-6), max(b[3] - b[1], 1e-6))


def relative_center(target, region) -> Tuple[float, float]:
    """region 중심의 target 박스 내부 상대 좌표 (0~1). 이동/크기 변화와 무관하게 비교하기 위함."""
    w, h = wh(target)
    cx, cy = center(region)
    return ((cx - target[0]) / w, (cy - target[1]) / h)


def relative_offset(target, other) -> Tuple[float, float]:
    """other 중심이 target 중심에서 얼마나 떨어져 있는지 (target 폭/높이로 정규화).
    겹침이 사라진 뒤(갈라진 직후)에도 정의되므로 교집합 위치 흐름을 경계 너머로 이어준다."""
    w, h = wh(target)
    tx, ty = center(target)
    ox, oy = center(other)
    return ((ox - tx) / w, (oy - ty) / h)


H_LABELS = ("L", "C", "R")
V_LABELS = ("T", "B")


def position_bin(u: float, v: float) -> str:
    """왼쪽/중앙/오른쪽 x 위/아래 구간 라벨. 예: 'LB' = 왼쪽 아래."""
    hi = 0 if u < 1.0 / 3 else (1 if u < 2.0 / 3 else 2)
    vi = 0 if v < 0.5 else 1
    return H_LABELS[hi] + V_LABELS[vi]


def tlwh_to_tlbr(a) -> np.ndarray:
    a = np.asarray(a, dtype=float).copy()
    a[..., 2] += a[..., 0]
    a[..., 3] += a[..., 1]
    return a


def tlbr_to_tlwh(a) -> np.ndarray:
    a = np.asarray(a, dtype=float).copy()
    a[..., 2] -= a[..., 0]
    a[..., 3] -= a[..., 1]
    return a
