# occflow — 가림 영역의 시간적 위치 변화를 활용한 연결 보조 (DanceTrack)

재현한 OA-SORT(`baseline/OA-SORT_DanceTrack_Val_Colab.ipynb`) 위에 **바운딩 박스 교집합(가림 대리 영역)의
박스 내 상대 위치 흐름**을 보조 단서로 얹는 실험 코드다. baseline 노트북은 수정하지 않는다.

## 세 가지 실패 유형과 대응

| 실패 유형 | 분석 (`analysis.py`) | 대응 모듈 (`hook.py`) | 설정 |
|---|---|---|---|
| 1. 두 ID가 즉시 맞바뀜 | `category=1` (`kind=swap/takeover`) | **흐름 비용 보정** `adjust()`: 1차 매칭에서 경합 후보 중 가림 흐름과 이어지는 후보를 우대 | `flow.*` |
| 2. 끊겼다가 다른 ID로 이어짐 | `category=2` (`kind=newid/reuse`), 끊김 구간 검출 유무 | **가림 중 놓친 트랙 재연결** `recover()`: 새 ID 생성 직전, 상대 기준 흐름이 예상하는 위치의 검출을 기존 트랙에 연결 | `recovery.*` |
| 3. 틀린 ID가 끝까지 감 | `persistent=1` (`run_len`, `to_end`), GT별 `wrong_id_frames` | **ID 사후 점검** `audit()`: 교차 쌍에서 '교환한 배정'이 교차 전 흐름과 연속으로 더 잘 맞으면 ID 교환 | `audit.*` |

## 폴더 구조

```
occflow/
  occflow/
    geometry.py      교집합, 박스 내 상대 위치(u,v), 상대 오프셋, L/C/R x T/B 구간 라벨
    occlusion.py     쌍 관측 (겹침/겹침 없음 구분, 하단 y 기반 앞뒤 + 신뢰도)
    history.py       트랙별 가림 이력, 흐름 예측(최근 이력 1차 회귀), 흐름 불일치 점수
    hook.py          OcclusionFlowHook: adjust / recover / audit / observe / prune
    config.py        모든 임계값 + 프리셋 (off, null, flow, rec, audit, flow_rec, flow_audit, full)
    analysis.py      GT 기준 실패 유형 분류, CLEAR/IDF1 근사, 추적기 간 이벤트 대응
    io_utils.py      MOT 입출력
    integration/
      oa_flow.py        재현 OASort 를 상속한 OAFlowSort (재현 코드와 같은 AST 방식으로 훅 삽입)
      run_oa_flow.py    시퀀스 1개: 검출 캐시(1회 YOLOX) -> 여러 모드 재생 -> MOT 결과 + 디버그 로그
      evaluate_modes.py TrackEval (HOTA/AssA/IDF1/IDSW/Frag), 모드 수 제한 없음
      colab_cells.py    baseline 노트북 뒤에 붙여 실행할 셀 A~E
  tools/
    analyze_failures.py   실패 유형 분석 + 추적기 비교 CSV
    visualize_event.py    이벤트 구간 영상 (GT, 추적기별 ID, 교집합 강조)
    check_equivalence.py  두 결과 폴더가 같은지 검사
  tests/
    test_core.py          단위 테스트 (교차 직후 후보 선택, 이력 보류, 교환 감지, 재연결, 분석 분류)
    test_integration.py   실제 Hybrid-SORT + 재현 OA-SORT 위에서 전체 파이프라인 + oa_null==oa_sort 검사
    synthetic.py          합성 교차 장면
```

## 동작 원리 (계획서 항목 대응)

**가림 위치 파악** (`occlusion.observe_pair`)
- 두 박스의 교집합 중심을 대상 박스 기준 상대 좌표 `(u, v)`로 기록 (이동·크기 변화와 무관). 구간 라벨 `LB, CB, RB…`.
- 상대 중심 오프셋 `(dx, dy)`(대상 폭/높이 단위)도 함께 기록한다. 교집합 위치와 같은 방향으로 움직이고,
  **두 사람이 갈라진 직후(겹침 0)에도 정의**되므로 흐름을 겹침 경계 너머로 이어준다.
- 앞뒤는 OA-SORT처럼 하단 y로 추정하되 차이가 작으면 신뢰도를 낮춰 가중치를 줄인다 (§5.4 하체 가림/점프 대비).
- 상태 구분: `overlap` / `no_overlap`(상대는 보이지만 안 겹침) / 미관측(이력에 넣지 않음).

**시간순 이력** (`history.OcclusionHistory`)
- 트랙마다 가림 상대 1명과 최근 `hist_len`개 관측(프레임, u/v, dx/dy, 피복률, 앞뒤)을 저장.
- 상대가 바뀌면 이력을 새로 시작한다. 겹침이 끝난 뒤 `exit_memory` 프레임 동안 유지하고, 마지막 관측 후 `max_stale`을 넘으면 사용하지 않는다.
- 흐름 예측: 최근 `fit_len`개 이력에 1차 회귀를 맞춘다. 불확실성은 관측 공백만큼 키운다.
  흐름 일관성 = |순이동| / 이동 거리 합 × 이동량 강도. 정지된 겹침은 흐름이 아니므로 0이 된다.
- **신뢰 관측만 추가**: 점수 `trust_min_score` 이상, 매칭 `trust_min_hits`회 이상, 재연결 직후 제외.
  상대도 같은 프레임에 관측돼야 하고, 기존 흐름과 크게 어긋나면(`trust_mismatch`) 보류(pending)한다.
  보류가 `max_pending`을 넘으면 새 흐름으로 재시작한다.

**연결 보조** (`hook.adjust`): 1차 매칭 비용 `-(HMIoU + 방향/점수 항)` + OAO(재현식 그대로)에 더한다.
- 적용 조건: 이력 사용 가능 + 상대 트랙이 같은 매칭에 있음 + 경합 후보 2개 이상(최저 비용 + `ambig_margin` 이내,
  또는 상대와 같은 검출을 다투는 경우) + 신뢰도 ≥ `min_reliability`. 조건이 맞지 않으면 **기존 추적기 판단을 그대로 따른다**.
- 후보 j의 흐름 불일치는 '대상이 j를 가져가면 상대는 남은 최선 후보에 있다'는 가정으로 계산한다
  (두 사람의 배정을 쌍으로 비교하는 셈). OAO 보정값은 행 전체에 같지만, 이 항은 **후보마다 다르다**.
- 행 안에서 가장 잘 맞는 후보 대비 상대 차이만 더한다: `weight × 신뢰도 × (불일치_j − 불일치_min)`.
  이력이 있다는 사실만으로 우대나 불이익이 생기지 않는다.

## 실행 (Colab)

1. baseline 노트북의 1~7번 셀을 실행한다 (이미 돌린 결과 폴더가 있으면 같은 `RESULTS_ROOT`).
2. 이 `occflow` 폴더를 Colab에 올린다 (예: `/content/occflow`).
3. `occflow/integration/colab_cells.py`의 셀 A~E를 노트북 뒤에 붙여 실행한다.
   - A: 설정 (`RUN_SEQS = FOCUS` = 0026, 0079부터. 전체는 `SEQUENCES`)
   - B: 시퀀스별 YOLOX 검출을 **처음 한 번만** 계산해 `det_cache/<seq>/`에 저장하고, 같은 검출로 모든 모드를 재생한다.
     다음부터는 GPU 없이 재생만 한다.
   - C: 동작 검증 (아래)
   - D: TrackEval
   - E: 실패 유형 분석 (oa_sort 대비)

모드: `hybrid`, `oa_sort`(재현 그대로), `oa_null`(검증용), `oa_flow`, `oa_rec`, `oa_audit`,
`oa_flow_rec`, `oa_flow_audit`, `oa_full`. 하이퍼파라미터 탐색은 `FLOW_SET=["flow.weight=0.5"]`, `TAG="_w05"`처럼 지정한다.

### 동작 검증 (계획서 '이력 모듈 비활성화 시 기준 추적기와 동일')
- `oa_null`은 세 모듈의 코드 경로를 모두 타지만 효과를 0으로 둔다(가중치 0, 재연결 임계 음수, 점검 margin 무한대).
  **`oa_null` 출력은 `oa_sort`와 완전히 같아야 한다** (`check_equivalence.py`, 셀 C).
- 셀 C의 첫 줄은 재생한 `oa_sort`와 노트북의 `oa_sort` 결과를 비교한다.
  FP16 YOLOX 추론 차이로 검출이 미세하게 다르면 여기서는 차이가 날 수 있다. 그래도 같은 캐시를 쓰는 모드끼리의 비교는 공정하다.

## 실패 사례 분석 (리뷰 의견 대응)

```
python tools/analyze_failures.py --gt_root BENCH/gt \
  --res oa_sort=RUNS/oa_sort hybrid=RUNS/hybrid oa_full=RUNS/oa_full \
  --det_root BENCH/occflow/det_cache --seqs dancetrack0026 dancetrack0079 --out out/analysis
```
- `<name>_events.csv`: ID 변경 이벤트마다 시퀀스, 프레임 범위(`prev_frame`→`frame`, 끊김 `gap_fn`), 전후 ID
  (`prev_pred`→`new_pred`), 가림 상대 GT와 상대의 전후 ID, `category`(1/2/R), `persistent`/`run_len`/`to_end`,
  **끊김 구간 검출 유무**(`gap_det_high/low/no_det`), 교집합 위치 흐름 `overlap_flow`(GT 기준, 예 `LB>CB>RB>--`), 앞뒤(`depth`)를 남긴다.
  → "두 사람 모두 다시 검출됐는데 ID가 뒤바뀐 경우"(cat 1, gap_det_high>0)와
  "검출이나 추적 출력이 비어 끊긴 경우"(cat 2, gap_no_det)를 바로 구분할 수 있다.
- `<name>_summary.csv`: `fn_det_high`(검출이 있었는데 출력이 빠진 FN), `fn_no_det`(검출 자체가 없던 FN),
  `fn_pred_overlap`(출력 박스는 있었지만 다른 GT에 배정), 유형별 개수.
- `<name>_gts.csv`: GT별 `wrong_id_frames`(IDF1 전역 배정 기준 틀린 ID 프레임 수). IDSW는 1건만 늘었는데
  AssA가 떨어진 0079처럼, **잘못 연결된 ID가 오래 유지되는 경우**를 GT 단위로 찾는 데 쓴다.
- `compare_oa_sort_vs_<other>_events.csv`: 같은 GT·근접 프레임 이벤트 대응 (`only_a`/`only_b`/`both`).
- 구간 확인: `tools/visualize_event.py --seq_dir <val>/dancetrack0026 --res ... --start --end --focus_gt <g> --video`.
- IDSW/Frag/IDF1은 TrackEval과 같은 규칙으로 근사한다. 보고용 수치(HOTA/AssA 포함)는 TrackEval(셀 D)을 쓴다.

## 디버그 로그 (`runs/<mode>_debug/`)
- `<seq>_assoc.csv`: 흐름 항이 적용된 행의 후보별 `base_cost, mismatch, adj, final_cost, reliability, chosen`.
  `flow.log_all=true`로 두면 적용하지 않은 사유도 기록한다.
- `<seq>_events.csv`: `recover`(재연결)와 `audit_swap`(ID 교환) 이벤트.
- `<seq>_summary.json`의 `flow_stats`: 적용·미적용 사유별 횟수
  (`rows_applied`, `skip_unambiguous`, `skip_hist_stale`, `rec_skip_*`, `obs_rejected` 등). **먼저 이 값으로 모듈이 실제로 작동했는지 확인한다.**

## 로컬 테스트
```
python tests/test_core.py
OCCFLOW_REPO=<HybridSORT> OCCFLOW_CODE_DIR=<노트북 CODE_DIR> python tests/test_integration.py
```
Colab에서는 노트북의 `PYTHON`, `REPO`, `CODE_DIR`로 그대로 실행할 수 있다.

## 실험 기록 (2026-11-26, WSL2 로컬, dancetrack0026 + 0079)

TrackEval 합산 (같은 YOLOX 검출 캐시 사용, `scripts/run_local.sh`):

| 모드 | HOTA | AssA | IDF1 | IDSW | FN |
|---|---:|---:|---:|---:|---:|
| hybrid | 68.49 | 58.55 | 74.51 | 158 | 1079 |
| oa_sort (재현) | 66.65 | 55.36 | 71.93 | 156 | 1085 |
| oa_flow (기본값) | 65.95 | 54.34 | 71.31 | 148 | 1090 |

- `oa_null` 출력 == `oa_sort` 출력 (두 시퀀스 모두 동일).
- 재연결(`rec`)과 ID 사후 점검(`audit`)은 기본값에서 한 번도 적용되지 않았다.
- 흐름 보정이 결정을 뒤집은 48건을 GT로 판정한 결과 (`tools/diagnose_flow.py`): 고침 19, 망침 18, 불명 11.
  신뢰도와 일관성으로는 고침과 망침이 구분되지 않는다.
- 오라클 실험 (`tools/oracle_flow.py`, `scripts/oracle_sweep.sh`): GT 박스만으로 겹친 쌍의 정답 배정과 교환 배정 중 하나를 고르게 했다.
  모든 공백(1~15프레임)과 잡음(0, 3%) 조건에서 흐름 단서가 각자 직선 운동 예측보다 약간 낮았다
  (공백 10: 0.948 vs 0.969). 운동 예측이 틀린 프레임에서 흐름이 맞힌 비율은 약 절반(동전 던지기 수준)이었다.
- 해석: 교집합 위치와 상대 오프셋은 두 박스로부터 결정되는 값이다. 따라서 그 시간 추세를 직선으로 외삽하면
  두 박스 궤적을 각각 외삽한 것과 거의 같은 정보가 된다. 지금 구현에서는 새 정보가 들어가지 않는다는 뜻이다.
  이득을 보려면 다른 정보원(앞뒤 관계, 외형/ReID, 비선형·장기 패턴)과 결합하거나,
  가림 맥락을 "언제 기존 판단을 믿지 말지" 정하는 데 써야 할 가능성이 크다.
- **val 전체 25개 시퀀스 오라클** (`SEQS=all bash scripts/oracle_sweep.sh`, 어려운 프레임 = 두 박스 IoU ≥ 0.1):

  | 잡음 | 공백 | n | 흐름 | 운동 | 갈린 경우 흐름 승 / 운동 승 | 흐름이 나은 시퀀스 |
  |---|---:|---:|---:|---:|---|---:|
  | 0 | 1 | 52148 | 0.994 | 0.996 | 71 / 181 | 1 / 25 |
  | 0 | 5 | 46960 | 0.952 | 0.966 | 533 / 1200 | 1 / 25 |
  | 0 | 10 | 42721 | 0.896 | 0.921 | 1213 / 2260 | 1 / 25 |
  | 0 | 15 | 39540 | 0.848 | 0.879 | 1779 / 3001 | 3 / 25 |
  | 0.03 | 10 | 42374 | 0.886 | 0.903 | 1549 / 2301 | 2 / 25 |
  | 0.03 | 15 | 39204 | 0.830 | 0.851 | 2149 / 2975 | 2 / 25 |

  모든 조건에서 운동 예측이 앞섰고, 흐름이 나은 시퀀스도 조건마다 바뀌었다(일관된 시퀀스 없음).
  운동 예측이 틀린 프레임 중 흐름이 맞힌 비율은 약 36~37%(잡음 0, 공백 10~15)로, 대체 신호로 쓰기에도 부족하다.
  두 방식 중 하나만 맞아도 정답으로 치는 상한도 공백 10에서 0.921 → 0.949 수준이다.
- **판단 보류 실험** (`occflow/deferred.py`, `tools/defer_resolve.py`, `scripts/run_defer.sh`):
  교차 중에는 ID 확정을 미루고, 흩어진 뒤 교차 전후 구간을 양방향 직선 외삽으로 이어 keep/swap(순열)을 정하는 고정 지연 후처리.
  - GT 오라클 (val 전체, 잡음 0): 두 트랙 단위 정확도 0.809(양방향) vs 0.748(앞 구간만).
    여러 명 묶음 단위는 0.793 vs 0.727이고, 3명 이상이면 0.50 vs 0.44였다. 양방향이 앞 구간만 보는 것보다 일관되게 낫다.
  - 실제 결과 적용 (0026 + 0079): 두 트랙 단위와 여러 명 묶음 단위 모두 GT 판정상 **고친 경우 0건**, 잘못 바꾼 경우 2~4건이었다.
    oa_sort HOTA는 66.65 → 65.45(두 트랙), 65.57(묶음)으로 떨어졌다. 판정 가능한 구간에서 교환이 필요했던 경우(missed)도 0건이었다.
  - 원인 (`tools/locate_errors.py`): 추적기 ID 변경 이벤트의 93~94%는 3명 이상이 겹친 상황에서 생기고, 교차와 무관한 오류는 0건이었다.
    판정 가능한 구간(멤버가 깨끗하게 흩어지는 경우)은 추적기가 이미 맞게 처리했다.
    오류가 나는 구간은 정해진 지연 안에 깨끗하게 흩어지지 않아 판단 대상에서 빠진다(`short_window`).
  - 결론: 박스 운동만으로 판단을 미루는 방식은 실제 오류 구간에 닿지 않는다. 흩어지지 않는 뭉침에서는
    외형(ReID) 같은 운동 외 단서가 필요하다.
- 별도 관찰: `oa_sort` FN 1085건 중 846건(78%)은 같은 위치에 고신뢰 검출이 있었다. 검출이 아니라 연결과 출력 단계에서 빠진 것이다.

## 주의 / 현재 상태
- 합성 장면 통합 테스트에서 검증한 것은 **파이프라인 동작과 `oa_null == oa_sort` 동일성**뿐이다. 성능 근거가 아니다.
  합성 장면에서는 가려진 사람의 검출이 빠지는 경우가 많아, 흐름 비용 보정은 거의 적용되지 않았다
  (`skip_unambiguous`가 대부분). 재연결은 1건 적용됐는데 틀린 연결이어서 기본값을 보수적으로 낮췄다
  (`max_lost` 15, `max_mismatch` 0.35).
- `flow.ambig_margin`, `flow.weight`, `recovery.*`, `audit.*`은 모두 초기값이다. 0026/0079에서 적용 횟수
  (`flow_stats`)와 이벤트 변화를 보고 정한다.
- `audit.rewrite_past=true`는 과거 출력까지 고치는 **오프라인** 변형이다. 온라인 추적기와 비교할 때는 `false`로 둔다.
- 사후 점검은 춤 동작 중 실제 방향 전환을 교차로 오인할 위험이 있다. `audit_swap` 이벤트를 GT와 대조해
  맞는 교환과 틀린 교환을 세어 본다.
- 여러 사람이 동시에 겹치면 현재는 가장 크게 겹치는 상대 1명만 추적한다 (상대별 이력 분리는 확장 과제).
