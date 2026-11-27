"""MOT 형식 입출력 (frame,id,x,y,w,h,score,...)."""
from __future__ import annotations

import configparser
import os
from typing import Dict, Optional, Tuple

import numpy as np

from .geometry import tlwh_to_tlbr

FrameData = Tuple[np.ndarray, np.ndarray, np.ndarray]  # ids, boxes(tlbr), scores


def read_mot_txt(path: str) -> np.ndarray:
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return np.zeros((0, 10))
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",") if "," in line else line.split()
            rows.append([float(x) for x in parts])
    width = max(len(r) for r in rows)
    arr = np.full((len(rows), max(width, 7)), -1.0)
    for i, r in enumerate(rows):
        arr[i, :len(r)] = r
    return arr


def load_gt(path: str) -> np.ndarray:
    arr = read_mot_txt(path)
    if len(arr) and arr.shape[1] > 6:
        arr = arr[arr[:, 6] > 0]                    # 평가 대상 플래그
    if len(arr) and arr.shape[1] > 7 and (arr[:, 7] == 1).any():
        arr = arr[arr[:, 7] == 1]                   # pedestrian
    return arr


def to_frame_dict(arr: np.ndarray) -> Dict[int, FrameData]:
    out: Dict[int, FrameData] = {}
    if len(arr) == 0:
        return out
    frames = arr[:, 0].astype(int)
    order = np.argsort(frames, kind="stable")
    arr, frames = arr[order], frames[order]
    uniq, starts = np.unique(frames, return_index=True)
    for f, chunk in zip(uniq, np.split(arr, starts[1:])):
        out[int(f)] = (chunk[:, 1].astype(int), tlwh_to_tlbr(chunk[:, 2:6]), chunk[:, 6].copy())
    return out


def load_det_frames(path: str) -> Dict[int, np.ndarray]:
    """검출 파일 -> {frame: (N,5) tlbr+score}."""
    return {f: np.concatenate([b, s[:, None]], axis=1) for f, (_, b, s) in to_frame_dict(read_mot_txt(path)).items()}


def find_result_file(res_dir: str, seq: str) -> Optional[str]:
    for cand in (os.path.join(res_dir, seq + ".txt"), os.path.join(res_dir, "data", seq + ".txt")):
        if os.path.isfile(cand):
            return cand
    return None


def find_det_file(det_root: str, seq: str) -> Optional[str]:
    for cand in (os.path.join(det_root, seq, "det.txt"), os.path.join(det_root, seq + ".txt"),
                 os.path.join(det_root, seq, "det", "det.txt")):
        if os.path.isfile(cand):
            return cand
    return None


def seq_info(seq_dir: str) -> dict:
    cp = configparser.ConfigParser()
    cp.read(os.path.join(seq_dir, "seqinfo.ini"), encoding="utf-8-sig")
    s = cp["Sequence"] if "Sequence" in cp else {}
    return {"length": int(s.get("seqLength", 0)), "width": int(s.get("imWidth", 0)),
            "height": int(s.get("imHeight", 0)), "im_dir": s.get("imDir", "img1"),
            "im_ext": s.get("imExt", ".jpg")}


def list_sequences(gt_root: str):
    return sorted(d for d in os.listdir(gt_root)
                  if os.path.isfile(os.path.join(gt_root, d, "gt", "gt.txt")))
