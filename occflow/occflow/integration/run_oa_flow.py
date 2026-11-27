"""시퀀스 1개: YOLOX 검출을 (캐시가 없을 때만) 1회 계산 -> 같은 검출로 여러 모드를 재생.

baseline 노트북의 5번 셀까지 실행한 Colab 환경에서, 노트북과 같은 PYTHON / cwd=REPO 로 실행한다.

  python run_oa_flow.py --repo REPO --code_dir CODE_DIR --det_cache DET_CACHE \
      --frames FRAMES --weights CKPT --config REPO/VAL_CFG --fp16 \
      --output OUT --sequence dancetrack0026 \
      --modes oa_sort,oa_null,oa_flow,oa_flow_rec,oa_full

검출 캐시가 있으면 --frames/--weights/--config 없이도 재생만 한다 (CPU로도 가능).
결과: OUT/<mode>/<seq>.txt (MOT), OUT/<mode>_debug/<seq>_assoc.csv, <seq>_events.csv
"""
from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
import pickle
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]          # .../occflow (repo root)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 모드 이름 -> occflow.config.PRESETS 이름 (hybrid / oa_sort 는 재현 노트북 그대로)
FLOW_MODES = {
    "oa_null": "null",
    "oa_flow": "flow",
    "oa_rec": "rec",
    "oa_audit": "audit",
    "oa_flow_rec": "flow_rec",
    "oa_flow_audit": "flow_audit",
    "oa_full": "full",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------- det cache
def save_cache(folder: Path, detections, sizes, tracker_spec) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    counts = np.array([len(d) for d in detections], dtype=np.int64)
    boxes = (np.concatenate(detections, axis=0) if counts.sum() else np.zeros((0, 5), np.float32))
    np.savez_compressed(folder / "detections.npz", boxes=boxes.astype(np.float32), counts=counts,
                        sizes=np.asarray(sizes, dtype=np.int64))
    with (folder / "tracker_spec.pkl").open("wb") as f:
        pickle.dump(tracker_spec, f)
    # 다른 도구용 MOT 형식 사본 (frame,-1,x,y,w,h,score)
    with (folder / "det.txt").open("w", encoding="utf-8") as f:
        for fi, d in enumerate(detections, 1):
            for x1, y1, x2, y2, s in d:
                f.write("%d,-1,%.2f,%.2f,%.2f,%.2f,%.5f,-1,-1,-1\n" % (fi, x1, y1, x2 - x1, y2 - y1, s))


def load_cache(folder: Path):
    data = np.load(folder / "detections.npz")
    counts = data["counts"]
    splits = np.cumsum(counts)[:-1]
    detections = [d.astype(np.float32) for d in np.split(data["boxes"], splits)]
    sizes = [tuple(int(v) for v in s) for s in data["sizes"]]
    # 직접 만든 캐시 파일만 읽는다 (pickle 은 신뢰하는 파일에만 사용)
    with (folder / "tracker_spec.pkl").open("rb") as f:
        tracker_spec = pickle.load(f)
    return detections, sizes, tracker_spec


# ------------------------------------------------------------------- csv
def write_csv(path: Path, rows, preferred=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(preferred)
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", required=True, type=Path, help="HybridSORT 저장소 (노트북의 REPO)")
    p.add_argument("--code_dir", required=True, type=Path, help="oa_sort_impl.py 가 있는 노트북 CODE_DIR")
    p.add_argument("--det_cache", required=True, type=Path, help="검출 캐시 루트 (시퀀스별 하위 폴더)")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--sequence", required=True)
    p.add_argument("--frames", type=Path)
    p.add_argument("--weights", type=Path)
    p.add_argument("--config", type=Path, help="YOLOX exp 설정 (노트북의 REPO/VAL_CFG)")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--tau", type=float, default=0.15)
    p.add_argument("--depth-pixels", type=float, default=5)
    p.add_argument("--no-gaussian", action="store_true")
    p.add_argument("--modes", default="oa_sort,oa_null,oa_flow,oa_flow_rec,oa_full")
    p.add_argument("--tag", default="", help="모드 폴더 이름 뒤에 붙일 접미사 (하이퍼파라미터 탐색용)")
    p.add_argument("--cfg", default=None, help="occflow 설정 json (flow/recovery/audit 섹션)")
    p.add_argument("--set", nargs="*", default=[], help="예: flow.weight=0.5 audit.confirm_frames=4")
    p.add_argument("--no_debug", action="store_true")
    a = p.parse_args()

    sys.path.insert(0, str(a.code_dir.resolve()))
    import oa_sort_impl  # noqa: E402  (노트북이 생성한 재현 코드)
    from occflow.config import build_config  # noqa: E402
    from occflow.integration.oa_flow import make_flow_tracker  # noqa: E402

    cache = a.det_cache / a.sequence
    if (cache / "detections.npz").is_file():
        detections, sizes, tracker_spec = load_cache(cache)
        print("[occflow] detection cache:", cache, flush=True)
    else:
        if a.frames is None or a.weights is None or a.config is None:
            raise SystemExit("검출 캐시가 없습니다. --frames --weights --config 를 지정하세요.")
        import oa_sort_val_runner as runner  # noqa: E402
        frames = sorted(x.resolve() for x in a.frames.iterdir() if x.suffix.lower() in IMAGE_SUFFIXES)
        if not frames:
            raise SystemExit("no images in %s" % a.frames)
        detections, sizes, tracker_spec = runner.collect_detections(
            a.repo, frames, a.weights, a.fp16, a.config, a.output / "demo_tmp")
        save_cache(cache, detections, sizes, tracker_spec)
        print("[occflow] detections cached:", cache, flush=True)

    base = importlib.import_module("trackers.hybrid_sort_tracker.hybrid_sort")
    kf = importlib.import_module("trackers.hybrid_sort_tracker.kalmanfilter_score_new")
    tracker_args, tracker_kwargs = tracker_spec
    arguments = inspect.signature(base.Hybrid_Sort).bind(*tracker_args, **tracker_kwargs)
    arguments.apply_defaults()
    arguments.arguments["asso_func"] = "Height_Modulated_IoU"   # 재현 노트북 replay() 와 동일
    arguments.arguments["use_byte"] = True
    tracker_args, tracker_kwargs = arguments.args, arguments.kwargs

    oa_config = oa_sort_impl.OAConfig(tau=a.tau, depth_pixels=a.depth_pixels, gaussian=not a.no_gaussian)
    OASort = oa_sort_impl.make_oa_tracker(base, kf.KalmanFilterNew_score_new)
    OAFlowSort = make_flow_tracker(base, kf.KalmanFilterNew_score_new, OASort)

    summary = {}
    for mode in [m.strip() for m in a.modes.split(",") if m.strip()]:
        flow_cfg = None
        if mode == "hybrid":
            tracker = base.Hybrid_Sort(*tracker_args, **tracker_kwargs)
        elif mode == "oa_sort":
            tracker = OASort(*tracker_args, **tracker_kwargs,
                             oa_config=replace(oa_config, use_oao=True, use_bam=True))
        elif mode in FLOW_MODES:
            flow_cfg = build_config(FLOW_MODES[mode], a.set if mode != "oa_null" else (), a.cfg
                                    if mode != "oa_null" else None)
            tracker = OAFlowSort(*tracker_args, **tracker_kwargs,
                                 oa_config=replace(oa_config, use_oao=True, use_bam=True),
                                 flow_config=flow_cfg)
        else:
            raise SystemExit("unknown mode %s (hybrid, oa_sort, %s)" % (mode, ", ".join(FLOW_MODES)))

        name = mode + (a.tag if mode in FLOW_MODES and mode != "oa_null" else "")
        rows = []
        rewrite = flow_cfg is not None and flow_cfg.audit.enable and flow_cfg.audit.rewrite_past
        n_events = 0
        for frame_number, (image_hw, boxes) in enumerate(zip(sizes, detections), 1):
            tracks = np.array(tracker.update(boxes.copy(), image_hw, image_hw), dtype=float).reshape(-1, 5)
            rows.append([frame_number, tracks])
            if rewrite:
                events = tracker.flow_events
                for ev in events[n_events:]:
                    if ev["event"] != "audit_swap":
                        continue
                    lo = max(ev["start_frame"], frame_number - flow_cfg.audit.max_rewrite)
                    ia, ib = ev["track_id"], ev["other_id"]
                    for f, tr in rows:
                        if lo <= f < frame_number and len(tr):
                            col = tr[:, 4].copy()
                            tr[col == ia, 4] = ib
                            tr[col == ib, 4] = ia
                n_events = len(events)

        import oa_sort_val_runner as runner  # record_frame 재사용 (노트북과 같은 MOT 출력 형식)
        min_box_area = float(getattr(tracker.args, "min_box_area", 0))
        mode_dir = a.output / name
        mode_dir.mkdir(parents=True, exist_ok=True)
        total = 0
        with (mode_dir / ("%s.txt" % a.sequence)).open("w", encoding="utf-8") as fh:
            for f, tr in rows:
                total += runner.record_frame(tr, f, fh, min_box_area)

        info = {"track_rows": total, "oa_stats": dict(getattr(tracker, "oa_stats", Counter()))}
        if flow_cfg is not None:
            info["flow_config"] = flow_cfg.to_dict()
            info["flow_stats"] = dict(tracker.flow_hook.stats) if tracker.flow_hook else {}
            info["events"] = dict(Counter(e["event"] for e in tracker.flow_events))
            if not a.no_debug:
                dbg = a.output / (name + "_debug")
                write_csv(dbg / ("%s_assoc.csv" % a.sequence), tracker.flow_assoc_logs,
                          ["frame", "track_id", "partner_id", "det_idx", "chosen", "active", "reason",
                           "base_cost", "mismatch", "adj", "final_cost", "reliability", "consistency"])
                write_csv(dbg / ("%s_events.csv" % a.sequence), tracker.flow_events,
                          ["frame", "event", "track_id", "other_id", "value", "lost_frames", "start_frame"])
        summary[name] = info
        print("%s: rows=%d %s" % (name, total, info.get("events", "")), flush=True)

    report = {"sequence": a.sequence, "processed_frames": len(detections), "modes": summary}
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / ("%s_summary.json" % a.sequence)).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
