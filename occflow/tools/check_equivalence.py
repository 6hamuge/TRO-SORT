"""두 결과 폴더의 MOT 출력이 완전히 같은지 확인 (계획서 '이력 모듈 비활성화 시 기준 추적기와 동일').

  python tools/check_equivalence.py OUT/oa_sort OUT/oa_null
  python tools/check_equivalence.py BENCH/trackers/oa_sort/data OUT/oa_sort   # 재현 노트북 결과와 재생 결과 비교
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from occflow.io_utils import read_mot_txt  # noqa: E402


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    a_dir, b_dir = sys.argv[1], sys.argv[2]
    tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-3
    files = sorted(f for f in os.listdir(a_dir) if f.endswith(".txt"))
    bad = 0
    for f in files:
        pb = os.path.join(b_dir, f)
        if not os.path.isfile(pb):
            print("[missing]", f)
            bad += 1
            continue
        A, B = read_mot_txt(os.path.join(a_dir, f)), read_mot_txt(pb)
        A = A[np.lexsort((A[:, 1], A[:, 0]))][:, :6] if len(A) else A[:, :6]
        B = B[np.lexsort((B[:, 1], B[:, 0]))][:, :6] if len(B) else B[:, :6]
        if A.shape != B.shape or (len(A) and np.abs(A - B).max() > tol):
            bad += 1
            first = None
            if A.shape == B.shape:
                rows = np.flatnonzero(np.abs(A - B).max(1) > tol)
                first = int(A[rows[0], 0]) if len(rows) else None
            print("[DIFF] %s rows %d vs %d, first differing frame=%s" % (f, len(A), len(B), first))
        else:
            print("[same] %s (%d rows)" % (f, len(A)))
    print("\n%d / %d files differ" % (bad, len(files)))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
