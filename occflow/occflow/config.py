"""실험 설정. 모든 임계값은 초기값이며 실험으로 정한다 (계획서: '판단 기준은 실험을 통해 정함')."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional


@dataclass
class FlowConfig:
    """가림 위치 흐름 비용 (실패 유형 1: 교차 직후 후보 선택)."""
    enable: bool = True
    weight: float = 0.3          # 비용 결합 가중치 (1차 매칭 비용 = -(HMIoU + 방향/점수 항) 스케일)
    ambig_margin: float = 0.15   # 최저 비용 + margin 이내 후보만 '경합 후보'로 봄
    hist_len: int = 10           # 저장 이력 길이
    fit_len: int = 6             # 흐름 예측에 쓰는 최근 이력 수
    min_hist: int = 3            # 이력 최소 길이
    min_overlap_entries: int = 2 # 실제 겹침 관측 최소 개수
    max_stale: int = 5           # 마지막 확정 관측 이후 허용 프레임 수 (넘으면 활용 보류)
    exit_memory: int = 10        # 겹침이 끝난 뒤(갈라진 직후) 이력을 유지하는 프레임 수
    min_cov: float = 0.05        # 가림 상대로 인정할 최소 피복률 (교집합 / 대상 박스 면적)
    partner_switch_ratio: float = 1.5  # 다른 상대가 이만큼 더 크게 겹쳐야 상대 교체 (흔들림 방지)
    min_consistency: float = 0.4 # 흐름 일관성 하한 (사후 점검에서 사용)
    min_disp: float = 0.1        # '흐름'으로 볼 최소 이동량 (대상 박스 크기 정규화 단위)
    min_reliability: float = 0.25
    sigma_floor: float = 0.08    # 예측 불확실성 하한 (정규화 단위)
    sigma_growth: float = 0.15   # 관측 공백 1프레임당 불확실성 증가율
    use_depth: bool = True       # 하단 y 기반 앞뒤 신뢰도로 가중치 조절
    depth_margin: float = 0.1    # 하단 y 차이 / 박스 높이 가 이 값 이상이면 앞뒤 확신 1
    trust_min_score: float = 0.1 # 이력에 넣을 관측의 최소 검출 점수. 가려지는 쪽은 점수가 떨어져
                                 # BYTE(저신뢰) 단계로 매칭되므로 기본값은 BYTE 하한과 같게 둔다
    trust_min_hits: int = 2      # 이력에 넣을 트랙의 최소 매칭 횟수
    trust_mismatch: float = 0.6  # 이력과 이 이상 어긋나는 관측은 이력에 넣지 않음 (오연결 누적 방지)
    max_pending: int = 5         # 연속 불일치가 이만큼 쌓이면 새 흐름으로 이력 재시작
    log_all: bool = False        # 비활성 사유까지 모두 로그


@dataclass
class RecoveryConfig:
    """가림 중 놓친 트랙 재연결 (실패 유형 2: 끊겼다가 다른 ID로 이어짐)."""
    enable: bool = False
    max_lost: int = 15           # 이 프레임 이상 놓친 트랙은 대상 아님 (긴 공백을 연속 이동으로 단정하지 않음)
    max_extrap: int = 5          # 흐름 외삽 최대 프레임 (이후는 마지막 추세 고정)
    max_mismatch: float = 0.35
    min_consistency: float = 0.4
    size_ratio: float = 1.4      # 허용 높이 비율
    min_expected_iou: float = 0.1  # 흐름으로 예상한 위치 박스와 후보의 최소 IoU


@dataclass
class AuditConfig:
    """교차 후 ID 사후 점검 (실패 유형 3: 한 번 틀린 ID가 끝까지 감)."""
    enable: bool = False
    margin: float = 0.5          # (현재 배정 불일치 합 - 교환 배정 불일치 합) 이 이보다 커야 증거
    confirm_frames: int = 3      # 연속 증거 프레임 수
    cooldown: int = 30           # 같은 쌍 재교환 금지 기간
    rewrite_past: bool = False   # True면 과거 출력까지 교정 (오프라인 변형 - 온라인 비교 시 False)
    max_rewrite: int = 30


@dataclass
class ExperimentConfig:
    flow: FlowConfig = field(default_factory=FlowConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def any_enabled(self) -> bool:
        return self.flow.enable or self.recovery.enable or self.audit.enable


_ALL_OFF = {"flow.enable": False, "recovery.enable": False, "audit.enable": False}

PRESETS = {
    # 모든 모듈 꺼짐 = 재현 OA-SORT 그대로
    "off": dict(_ALL_OFF),
    # 세 모듈의 코드 경로를 모두 타지만 결과에는 영향이 없도록 설정.
    # -> 재현 OA-SORT(oa_sort)와 출력이 완전히 같아야 한다 (계획서 '동작 검증').
    "null": {"flow.enable": True, "flow.weight": 0.0,
             "recovery.enable": True, "recovery.max_mismatch": -1.0,
             "audit.enable": True, "audit.margin": 1e9},
    "flow": dict(_ALL_OFF, **{"flow.enable": True}),
    "flow_rec": dict(_ALL_OFF, **{"flow.enable": True, "recovery.enable": True}),
    "flow_audit": dict(_ALL_OFF, **{"flow.enable": True, "audit.enable": True}),
    "rec": dict(_ALL_OFF, **{"recovery.enable": True}),
    "audit": dict(_ALL_OFF, **{"audit.enable": True}),
    "full": {"flow.enable": True, "recovery.enable": True, "audit.enable": True},
}


def _parse_value(text: str, current):
    if isinstance(current, bool):
        return text.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(current, int):
        return int(text)
    if isinstance(current, float):
        return float(text)
    return text


def set_value(cfg: ExperimentConfig, key: str, value) -> None:
    section, name = key.split(".", 1)
    sub = getattr(cfg, section)
    if not hasattr(sub, name):
        raise KeyError("unknown config key: %s" % key)
    cur = getattr(sub, name)
    if isinstance(value, str) and not isinstance(cur, str):
        value = _parse_value(value, cur)
    setattr(sub, name, value)


def build_config(preset: str = "off", overrides: Iterable[str] = (),
                 json_path: Optional[str] = None) -> ExperimentConfig:
    """preset -> json 파일 -> 'section.key=value' 덮어쓰기 순으로 적용."""
    if preset not in PRESETS:
        raise KeyError("unknown preset %s (choose from %s)" % (preset, list(PRESETS)))
    cfg = ExperimentConfig()
    for k, v in PRESETS[preset].items():
        set_value(cfg, k, v)
    if json_path:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for section, values in data.items():
            for k, v in values.items():
                set_value(cfg, "%s.%s" % (section, k), v)
    for item in overrides or ():
        k, v = item.split("=", 1)
        set_value(cfg, k.strip(), v.strip())
    return cfg
