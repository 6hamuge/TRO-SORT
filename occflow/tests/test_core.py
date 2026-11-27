"""핵심 모듈 단위 테스트.  python tests/test_core.py  (또는 pytest)"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.optimize import linear_sum_assignment  # noqa: E402

from occflow.analysis import AnalysisParams, analyze_sequence  # noqa: E402
from occflow.config import AuditConfig, FlowConfig, RecoveryConfig, build_config  # noqa: E402
from occflow.geometry import position_bin, relative_center, intersection_box  # noqa: E402
from occflow.hook import OcclusionFlowHook  # noqa: E402


def box(x, y=100, w=50, h=150):
    return np.array([x, y, x + w, y + h], dtype=float)


def crossing_hook(**flow_kw):
    """A(id 1)는 x=100에 정지, B(id 2)는 왼쪽에서 오른쪽으로 10px/프레임 통과 (프레임 1..6 관측).
    A 기준 교집합 위치: 왼쪽 -> 중앙 으로 이동 중."""
    hook = OcclusionFlowHook(FlowConfig(**flow_kw), RecoveryConfig(enable=True), AuditConfig(enable=True))
    for f in range(1, 7):
        hook.observe(f, [(1, box(100), True), (2, box(40 + 10 * (f - 1)), True)])
    return hook


def test_relative_position_bins():
    a = box(100)
    inter = intersection_box(a, box(60))
    u, v = relative_center(a, inter)
    assert abs(u - 0.1) < 1e-9 and abs(v - 0.5) < 1e-9
    assert position_bin(u, v) == "LB"
    assert position_bin(0.5, 0.2) == "CT"


def test_history_prediction():
    hook = crossing_hook()
    h = hook.histories[1]
    assert h.partner_id == 2 and len(h.entries) == 4          # 겹침이 시작된 프레임 3부터
    pred = h.predict(8, hook.cfg)
    assert pred is not None
    assert abs(pred.dx - 0.2) < 1e-6 and abs(pred.u - 0.6) < 1e-6
    assert pred.consistency > 0.99
    assert pred.depth_conf == 0.0                            # 하단 y 동일 -> 앞뒤 불분명


def test_adjust_prefers_flow_consistent_candidate():
    """프레임 8: d0=A 자리(x=100), d1=B 자리(x=110). 기본 비용은 '교환' 쪽이 약간 유리하게 설정.
    흐름 보정 후에는 A->d0, B->d1 이 선택되어야 한다."""
    hook = crossing_hook()
    dets = np.stack([box(100), box(110)])
    tracks = np.stack([box(100), box(110)])
    cost = np.array([[0.50, 0.45],     # A: d1 이 약간 유리 (잘못된 쪽)
                     [0.45, 0.50]])    # B: d0 이 약간 유리 (잘못된 쪽)
    r, c = linear_sum_assignment(cost)
    assert list(c) == [1, 0]                                 # 보정 전: 교환
    out, logs = hook.adjust(8, [1, 2], tracks, dets, cost, np.ones_like(cost, bool))
    r, c = linear_sum_assignment(out)
    assert list(c) == [0, 1], out                            # 보정 후: 올바른 연결
    assert any(lg["active"] == 1 for lg in logs)


def test_adjust_is_noop_without_ambiguity_or_weight():
    hook = crossing_hook()
    dets = np.stack([box(100), box(400)])
    cost = np.array([[0.1, 0.9], [0.9, 0.1]])
    out, _ = hook.adjust(8, [1, 2], dets, dets, cost, np.ones_like(cost, bool))
    assert np.array_equal(out, cost)                         # 경합 없음 -> 그대로
    hook0 = crossing_hook(weight=0.0)
    cost2 = np.array([[0.50, 0.45], [0.45, 0.50]])
    out2, _ = hook0.adjust(8, [1, 2], dets, np.stack([box(100), box(110)]), cost2, np.ones_like(cost2, bool))
    assert np.array_equal(out2, cost2)                       # 가중치 0 -> 그대로 (null 동작 검증의 근거)


def test_observe_rejects_inconsistent_and_audit_swaps():
    """프레임 7부터 A/B 박스가 뒤바뀐 채 관측(잘못된 연결이 이어지는 상황).
    이력에는 추가되지 않고(pending), 사후 점검이 confirm_frames 후 교환을 제안해야 한다."""
    hook = crossing_hook()
    swaps = []
    for f in range(7, 12):
        a_true, b_true = box(100), box(40 + 10 * (f - 1))
        mb = {1: b_true, 2: a_true}                          # ID 가 뒤바뀐 관측
        swaps += hook.audit(f, mb)
        if swaps:
            break
        hook.observe(f, [(1, b_true, True), (2, a_true, True)])
    assert hook.histories[1].pending >= 1 or swaps
    assert len(swaps) == 1 and set(swaps[0][:2]) == {1, 2}, swaps


def test_recover_lost_track_on_expected_side():
    """A(정지)가 B 뒤에서 가려져 놓침. B는 계속 오른쪽으로 이동. A가 다시 나타난 검출을 A로 재연결."""
    hook = crossing_hook()
    f = 9
    b_box = box(40 + 10 * (f - 1))                          # B: x=120
    cands = np.stack([box(100), box(400)])                   # A 재등장 후보 + 무관한 검출
    res = hook.recover(f, [(1, box(100), 3)], cands, {2: b_box})
    assert res and res[0][:2] == (0, 0), res


def test_presets():
    assert not build_config("off").any_enabled
    n = build_config("null")
    assert n.flow.weight == 0.0 and n.recovery.max_mismatch < 0 and n.audit.margin > 1e8
    c = build_config("flow", ["flow.weight=0.5", "flow.use_depth=false"])
    assert c.flow.weight == 0.5 and c.flow.use_depth is False


def _mot(rows):
    return np.array([[f, i, x, 100, 50, 150, 1, -1, -1, -1] for f, i, x in rows], dtype=float)


def test_analysis_categories():
    """g1,g2 프레임 21에서 서로 교환 (끝까지) / g3 는 16~20 끊긴 뒤 새 ID 9 로 이어짐."""
    gt, pr = [], []
    for f in range(1, 41):
        x1, x2, x3 = 100 + 5 * f, 400 - 5 * f, 900
        gt += [(f, 1, x1), (f, 2, x2), (f, 3, x3)]
        if f <= 20:
            pr += [(f, 1, x1), (f, 2, x2)]
        else:
            pr += [(f, 2, x1), (f, 1, x2)]
        if f <= 15:
            pr.append((f, 3, x3))
        elif f >= 21:
            pr.append((f, 9, x3))
    dets = {f: np.array([[900, 100, 950, 250, 0.9]]) for f in (16, 17, 18)}
    r = analyze_sequence("syn", _mot(gt), _mot(pr), dets, AnalysisParams(persist_min=15))
    ev = {e["gt_id"]: e for e in r["events"]}
    assert ev[1]["category"] == "1" and ev[1]["kind"] == "swap" and ev[1]["to_end"] == 1
    assert ev[2]["category"] == "1" and ev[2]["kind"] == "swap"
    assert ev[3]["category"] == "2" and ev[3]["kind"] == "newid" and ev[3]["gap_fn"] == 5
    assert ev[3]["gap_det_high"] == 3 and ev[3]["gap_no_det"] == 2
    s = r["summary"]
    assert s["IDSW"] == 3 and s["cat3_persistent"] == 3 and s["Frag"] == 1
    # 전역(IDF1) 배정 기준 틀린 ID 프레임: g1/g2 각 20 + g3 의 앞부분 15 (g3 -> 9 로 배정)
    assert s["wrong_id_frames"] == 55


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
