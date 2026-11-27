"""합성 '댄스' 장면: 사람들이 좌우로 오가며 반복 교차. 뒤에 있는 사람은 가려지면 검출이 빠지거나 점수가 낮아진다."""
from __future__ import annotations

import numpy as np


def make_scene(n_people: int = 6, n_frames: int = 300, seed: int = 0, W: int = 1280, H: int = 720):
    rng = np.random.RandomState(seed)
    x0 = rng.uniform(0.3 * W, 0.7 * W, n_people)
    amp = rng.uniform(0.15 * W, 0.3 * W, n_people)
    period = rng.uniform(80, 160, n_people)
    phase = rng.uniform(0, 2 * np.pi, n_people)
    yb = rng.uniform(0.75 * H, 0.95 * H, n_people)
    gt_rows, dets = [], []
    for f in range(1, n_frames + 1):
        t = 2 * np.pi * f / period + phase
        cx = x0 + amp * np.sin(t)
        bottom = yb + 12 * np.sin(0.5 * t)
        h = 0.32 * bottom
        w = 0.42 * h
        boxes = np.stack([cx - w / 2, bottom - h, cx + w / 2, bottom], axis=1)
        frame_dets = []
        for i in range(n_people):
            b = boxes[i]
            gt_rows.append([f, i + 1, b[0], b[1], w[i], h[i], 1, 1, 1])
            occ = 0.0
            for j in range(n_people):
                if j == i or boxes[j, 3] <= b[3]:
                    continue
                ix = max(0.0, min(b[2], boxes[j, 2]) - max(b[0], boxes[j, 0]))
                iy = max(0.0, min(b[3], boxes[j, 3]) - max(b[1], boxes[j, 1]))
                occ = max(occ, ix * iy / (w[i] * h[i]))
            if rng.rand() < 0.03 or (occ > 0.6 and rng.rand() < 0.7):
                continue
            score = rng.uniform(0.2, 0.55) if occ > 0.4 else rng.uniform(0.65, 0.95)
            jit = rng.normal(0, 0.02, 4) * np.array([w[i], h[i], w[i], h[i]])
            frame_dets.append([*(b + jit), score])
        dets.append(np.array(frame_dets, dtype=np.float32).reshape(-1, 5))
    return np.array(gt_rows, dtype=float), dets
