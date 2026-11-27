"""이벤트 구간 시각화: GT(흰색) + 추적기별 예측 ID, 대상 GT와 가림 상대의 교집합을 강조.

  python tools/visualize_event.py --seq_dir DATA/val/dancetrack0026 \
      --res hybrid=BENCH/trackers/hybrid/data oa_sort=BENCH/trackers/oa_sort/data oa_full=OUT/oa_full \
      --start 300 --end 360 --focus_gt 3 --out out/vis/0026_g3 --video

여러 추적기는 가로로 나란히 붙인다. opencv-python 필요.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from occflow.geometry import intersection_box, iou_matrix, position_bin, relative_center  # noqa: E402
from occflow.io_utils import find_result_file, load_gt, read_mot_txt, seq_info, to_frame_dict  # noqa: E402


def color(i: int):
    rng = np.random.RandomState(int(i) * 7919 % (2 ** 31))
    return tuple(int(c) for c in rng.randint(60, 255, 3))


def main():
    import cv2

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq_dir", required=True)
    ap.add_argument("--res", nargs="+", required=True, help="name=결과폴더")
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--focus_gt", type=int, nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--video", action="store_true")
    a = ap.parse_args()

    seq = os.path.basename(os.path.normpath(a.seq_dir))
    info = seq_info(a.seq_dir)
    G = to_frame_dict(load_gt(os.path.join(a.seq_dir, "gt", "gt.txt")))
    R = []
    for item in a.res:
        name, d = item.split("=", 1)
        path = find_result_file(d, seq)
        if path is None:
            raise SystemExit("결과 없음: %s %s" % (name, d))
        R.append((name, to_frame_dict(read_mot_txt(path))))
    os.makedirs(a.out, exist_ok=True)
    writer = None
    for f in range(a.start, a.end + 1):
        img_path = os.path.join(a.seq_dir, info["im_dir"], "%08d%s" % (f, info["im_ext"]))
        img = cv2.imread(img_path)
        if img is None:
            print("[skip] 이미지 없음", img_path)
            continue
        gi, gb, _ = G.get(f, (np.zeros(0, int), np.zeros((0, 4)), None))
        panels = []
        for name, P in R:
            canvas = img.copy()
            overlay = canvas.copy()
            for g, b in zip(gi, gb):
                if a.focus_gt and int(g) in a.focus_gt and len(gb) > 1:
                    ious = iou_matrix(b[None], gb)[0]
                    ious[list(gi).index(g)] = 0
                    k = int(np.argmax(ious))
                    inter = intersection_box(b, gb[k])
                    if inter is not None:
                        x1, y1, x2, y2 = inter.astype(int)
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
                        lab = position_bin(*relative_center(b, inter))
                        cv2.putText(canvas, "g%d<-g%d %s" % (g, gi[k], lab), (int(b[0]), int(b[3]) + 22),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                x1, y1, x2, y2 = b.astype(int)
                thick = 3 if int(g) in a.focus_gt else 1
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), thick)
                cv2.putText(canvas, "g%d" % g, (x1, y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            canvas = cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0)
            pi, pb, _ = P.get(f, (np.zeros(0, int), np.zeros((0, 4)), None))
            for q, b in zip(pi, pb):
                x1, y1, x2, y2 = b.astype(int)
                c = color(q)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), c, 2)
                cv2.putText(canvas, str(int(q)), (x1 + 3, y1 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, c, 2)
            cv2.putText(canvas, "%s  frame %d" % (name, f), (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
            panels.append(cv2.resize(canvas, None, fx=a.scale, fy=a.scale))
        frame_img = np.concatenate(panels, axis=1)
        cv2.imwrite(os.path.join(a.out, "%06d.jpg" % f), frame_img)
        if a.video:
            if writer is None:
                h, w = frame_img.shape[:2]
                writer = cv2.VideoWriter(os.path.join(a.out, "clip.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
            writer.write(frame_img)
    if writer is not None:
        writer.release()
    print("saved:", a.out)


if __name__ == "__main__":
    main()
