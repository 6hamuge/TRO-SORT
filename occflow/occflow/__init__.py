"""Occlusion-flow association for MOT (DanceTrack experiments).

바운딩 박스 기반 가림 영역(교집합)의 시간적 위치 변화를 연결 보조 단서로 쓰는 모듈.

- geometry / occlusion : 교집합 영역의 박스 내 상대 위치, 앞뒤 추정(하단 y 기준)
- history              : 트랙별 가림 상대 + 관측 위치 이력, 흐름 예측
- hook                 : 추적기에 끼우는 훅 (비용 보정 / 재연결 / ID 사후 점검)
- integration          : 재현 OA-SORT(baseline 노트북)에 훅을 붙인 추적기와 실행기
- analysis             : GT 기준 실패 유형(즉시 교환 / 단절 후 다른 ID / 오연결 지속) 분석
"""

from .config import (AuditConfig, ExperimentConfig, FlowConfig, PRESETS,
                     RecoveryConfig, build_config)
from .hook import OcclusionFlowHook

__all__ = [
    "AuditConfig", "ExperimentConfig", "FlowConfig", "PRESETS", "RecoveryConfig",
    "build_config", "OcclusionFlowHook",
]
