"""실제 Hybrid-SORT + 재현 OA-SORT(oa_sort_impl.py) 위에서 실행기 전체를 합성 검출로 점검.

  OCCFLOW_REPO=<HybridSORT 저장소>  OCCFLOW_CODE_DIR=<노트북 CODE_DIR>  python tests/test_integration.py
Colab 에서는 노트북의 REPO, CODE_DIR 를 넣고 같은 PYTHON 으로 실행하면 된다.

확인 항목
  1) oa_null (세 모듈 코드 경로를 모두 타지만 효과 0) 출력 == oa_sort 출력  <- 동작 검증
  2) 모든 flow 모드가 오류 없이 돌고 결과/로그를 남김
  3) 실패 유형 분석기가 결과에 대해 동작
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from occflow.integration.run_oa_flow import save_cache  # noqa: E402
from synthetic import make_scene  # noqa: E402

MODES = "hybrid,oa_sort,oa_null,oa_flow,oa_flow_rec,oa_flow_audit,oa_full"


def main():
    repo = os.environ.get("OCCFLOW_REPO")
    code = os.environ.get("OCCFLOW_CODE_DIR")
    if not repo or not code:
        raise SystemExit(__doc__)
    work = Path(os.environ.get("OCCFLOW_TEST_OUT") or tempfile.mkdtemp(prefix="occflow-it-"))
    seq = "synthetic0001"
    gt, dets = make_scene(n_people=6, n_frames=int(os.environ.get("OCCFLOW_TEST_FRAMES", "300")), seed=3)
    (work / "gt" / seq / "gt").mkdir(parents=True, exist_ok=True)
    np.savetxt(work / "gt" / seq / "gt" / "gt.txt", gt, fmt="%d,%d,%.2f,%.2f,%.2f,%.2f,%d,%d,%d")
    args = argparse.Namespace(track_thresh=0.6, TCM_first_step=True, TCM_first_step_weight=1.0,
                              TCM_byte_step=True, TCM_byte_step_weight=1.0, min_box_area=10,
                              use_nsa_kalman=False, kalman_GPR=False)
    spec = ((args, 0.6), dict(max_age=30, min_hits=3, iou_threshold=0.3, delta_t=3,
                              asso_func="Height_Modulated_IoU", inertia=0.05, use_byte=True))
    save_cache(work / "det_cache" / seq, dets, [(720, 1280)] * len(dets), spec)

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([repo, env.get("PYTHONPATH", "")])
    run = [sys.executable, str(ROOT / "occflow/integration/run_oa_flow.py"), "--repo", repo, "--code_dir", code,
           "--det_cache", str(work / "det_cache"), "--output", str(work / "runs"), "--sequence", seq,
           "--modes", MODES]
    subprocess.run(run, check=True, env=env, cwd=repo)

    eq = subprocess.run([sys.executable, str(ROOT / "tools/check_equivalence.py"),
                         str(work / "runs/oa_sort"), str(work / "runs/oa_null")], env=env)
    assert eq.returncode == 0, "oa_null != oa_sort : 훅이 기존 동작을 바꿈"

    for m in MODES.split(","):
        assert (work / "runs" / m / (seq + ".txt")).is_file(), m
    res = ["%s=%s" % (m, work / "runs" / m) for m in ("oa_sort", "hybrid", "oa_flow", "oa_flow_rec",
                                                      "oa_flow_audit", "oa_full")]
    subprocess.run([sys.executable, str(ROOT / "tools/analyze_failures.py"), "--gt_root", str(work / "gt"),
                    "--res", *res, "--det_root", str(work / "det_cache"), "--out", str(work / "analysis")],
                   check=True, env=env)
    print("\nOK - outputs in", work)


if __name__ == "__main__":
    main()
