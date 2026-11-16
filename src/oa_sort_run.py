
"""Use the official demo to capture one set of detections, then replay it."""
import argparse
import ast
from collections import Counter
from dataclasses import replace
import importlib
import inspect
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from oa_sort_impl import OAConfig, make_oa_tracker


DANCE_CFG = "exps/example/mot/yolox_dancetrack_val_hybrid_sort.py"
MODE_FLAGS = {"hybrid": (False, False), "oao_only": (True, False),
              "bam_only": (False, True), "oa_sort": (True, True)}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def model_detections(outputs, ratio):
    """Return [x1,y1,x2,y2,score] per detection in original image pixels."""
    candidates = outputs[0]
    if candidates is None:
        return np.empty((0, 5), dtype=np.float32)
    if hasattr(candidates, "detach"):
        candidates = candidates.detach().float().cpu().numpy()
    candidates = np.asarray(candidates, dtype=np.float32)
    if candidates.ndim != 2 or candidates.shape[1] < 5:
        raise ValueError(f"Unexpected YOLOX detection tensor: {candidates.shape}")
    boxes = candidates[:, :5].copy()
    boxes[:, :4] /= ratio
    if candidates.shape[1] > 5:
        boxes[:, 4] *= candidates[:, 5]
    return boxes


class DetectionCapture:
    def __init__(self, frames):
        self.frames = frames
        self.detections = []
        self.tracker_spec = None

    def install(self, namespace):
        predictor_class = namespace.get("Predictor")
        upstream_tracker = namespace.get("Hybrid_Sort")
        if predictor_class is None or upstream_tracker is None:
            raise RuntimeError("Unexpected Hybrid-SORT demo API: expected Predictor and Hybrid_Sort.")

        def create_tracker(*args, **kwargs):
            if self.tracker_spec is not None:
                raise RuntimeError("Expected one Hybrid-SORT tracker in the official image demo.")
            self.tracker_spec = args, kwargs
            return DetectionOnlyTracker()

        original_inference = predictor_class.inference

        def capture_inference(predictor, image, *args, **kwargs):
            result = original_inference(predictor, image, *args, **kwargs)
            outputs, info = result[:2]
            index = len(self.detections)
            if index >= len(self.frames):
                raise RuntimeError("The demo received more images than expected.")
            if isinstance(image, (str, Path)) and Path(image).resolve() != self.frames[index]:
                raise RuntimeError(f"Unexpected input order at frame {index + 1}: {image}")
            if getattr(predictor, "test_size", None) is not None:
                test_h, test_w = predictor.test_size
                ratio = min(test_h / info["height"], test_w / info["width"])
            else:
                ratio = float(info["ratio"])
            if not np.isfinite(ratio) or ratio <= 0:
                raise RuntimeError(f"Invalid detector scale: {ratio}")
            self.detections.append(model_detections(outputs, ratio))
            return result

        predictor_class.inference = capture_inference
        namespace["Hybrid_Sort"] = create_tracker


class DetectionOnlyTracker:
    def update(self, *_args, **_kwargs):
        return np.empty((0, 5), dtype=np.float32)


def collect_detections(repo, frames, weights, fp16):
    """Enter the official demo with its original CLI, config and checkpoint loader."""
    checkpoint_path = weights.expanduser().resolve(strict=True)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint is not a file: {checkpoint_path}")

    script = repo / "tools/demo_track.py"
    tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))

    # Older FastReID uses collections.Mapping. Import it only when requested.
    lazy_reid = ast.parse(
        "def FastReIDInterface(*args, **kwargs):\n"
        "    from fast_reid.fast_reid_interfece import FastReIDInterface as Interface\n"
        "    return Interface(*args, **kwargs)\n"
    ).body[0]
    for index, node in enumerate(tree.body):
        if (isinstance(node, ast.ImportFrom)
                and node.module == "fast_reid.fast_reid_interfece"
                and [(alias.name, alias.asname) for alias in node.names]
                    == [("FastReIDInterface", None)]):
            tree.body[index] = ast.copy_location(lazy_reid, node)
            break

    guards = [
        node for node in tree.body
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__"
        and any(isinstance(value, ast.Constant) and value.value == "__main__"
                for value in node.test.comparators)
    ]
    if len(guards) != 1:
        raise RuntimeError("Could not locate the official Hybrid-SORT demo entry point.")
    entry = guards[0].body
    main_calls = [
        index for index, stmt in enumerate(entry)
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Name) and stmt.value.func.id == "main"
    ]
    if len(main_calls) != 1:
        raise RuntimeError("Could not locate the Hybrid-SORT demo main() call.")

    # The experiment config can override the CLI checkpoint before main().
    entry.insert(main_calls[0], ast.parse("args.ckpt = oa_checkpoint_path").body[0])
    entry.insert(0, ast.parse("install_detection_capture(globals())").body[0])

    # Bind the actual load to the same file even if main() rewrites ckpt_file.
    demo_main, = [node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == "main"]
    checkpoint_loads = [
        node for node in ast.walk(demo_main)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "torch.load"
        and node.args and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "ckpt_file"
    ]
    if len(checkpoint_loads) != 1:
        raise RuntimeError("Could not locate the Hybrid-SORT checkpoint load.")
    checkpoint_loads[0].args[0] = ast.parse("oa_checkpoint_path", mode="eval").body
    ast.fix_missing_locations(tree)

    capture = DetectionCapture(frames)
    original_argv = sys.argv
    original_path = sys.path.copy()
    # OpenCV GUI calls are unavailable in Colab.
    cv2.namedWindow = lambda *_args, **_kwargs: None
    cv2.imshow = lambda *_args, **_kwargs: None
    cv2.waitKey = lambda *_args, **_kwargs: -1
    cv2.destroyAllWindows = lambda *_args, **_kwargs: None
    # Upstream main initializes vis_folder only when --save_result is set.
    sys.argv = [str(script), "--demo_type", "image", "-f", DANCE_CFG,
                "-c", str(checkpoint_path), "--path", str(frames[0].parent), "--fuse", "--save_result"]
    if fp16:
        sys.argv.append("--fp16")
    sys.path.insert(0, str(script.parent))
    try:
        namespace = {"__name__": "__main__", "__file__": str(script),
                     "install_detection_capture": capture.install,
                     "oa_checkpoint_path": str(checkpoint_path)}
        print(f"[OA-SORT] Detector checkpoint: {checkpoint_path}", flush=True)
        exec(compile(tree, str(script), "exec"), namespace)
    finally:
        sys.argv = original_argv
        sys.path[:] = original_path
    if capture.tracker_spec is None or len(capture.detections) != len(frames):
        raise RuntimeError(
            f"Expected {len(frames)} captured frames and one tracker config, "
            f"got {len(capture.detections)} frames. Check the upstream demo log."
        )
    return capture.detections, capture.tracker_spec


def record_frame(image, tracks, mode, frame_number, frame_dir, text_file, min_box_area):
    vis = image.copy()
    cv2.putText(vis, f"{mode} | frame {frame_number}", (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 2, cv2.LINE_AA)
    rows = 0
    for x1, y1, x2, y2, track_id in tracks:
        width, height = x2 - x1, y2 - y1
        if not np.isfinite([x1, y1, x2, y2, track_id]).all():
            continue
        if width <= 0 or height <= 0 or width * height <= min_box_area:
            continue
        track_id = int(track_id)
        color = tuple(80 + (track_id * factor) % 176 for factor in (37, 61, 97))
        left, top, right, bottom = map(round, (x1, y1, x2, y2))
        cv2.rectangle(vis, (left, top), (right, bottom), color, 2)
        cv2.putText(vis, str(track_id), (max(0, left), max(18, top - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        text_file.write(f"{frame_number},{track_id},{x1:.4f},{y1:.4f},"
                        f"{width:.4f},{height:.4f},1,-1,-1,-1\n")
        rows += 1
    target = frame_dir / f"{frame_number:08d}.jpg"
    if not cv2.imwrite(str(target), vis):
        raise OSError(f"Could not save {target}")
    return rows


def replay(frames, detections, tracker_spec, base, kf_class, modes, config, dest, sequence):
    tracker_args, tracker_kwargs = tracker_spec
    arguments = inspect.signature(base.Hybrid_Sort).bind(*tracker_args, **tracker_kwargs)
    arguments.apply_defaults()
    # Eq. 8 and Eq. 9 use HMIoU for this configuration. Both sides of the
    # comparison also need the same low-score association.
    arguments.arguments["asso_func"] = "Height_Modulated_IoU"
    arguments.arguments["use_byte"] = True
    tracker_args, tracker_kwargs = arguments.args, arguments.kwargs
    oa_tracker = make_oa_tracker(base, kf_class) if any(mode != "hybrid" for mode in modes) else None
    summary = {}

    for mode in modes:
        use_oao, use_bam = MODE_FLAGS[mode]
        if mode == "hybrid":
            tracker = base.Hybrid_Sort(*tracker_args, **tracker_kwargs)
        else:
            tracker = oa_tracker(
                *tracker_args, **tracker_kwargs,
                oa_config=replace(config, use_oao=use_oao, use_bam=use_bam),
            )
        min_box_area = float(getattr(tracker.args, "min_box_area", 0))
        frame_dir = dest / mode / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        total_rows = 0
        with (frame_dir.parent / f"{sequence}.txt").open("w", encoding="utf-8") as text_file:
            for frame_number, (frame, boxes) in enumerate(zip(frames, detections, strict=True), 1):
                image = cv2.imread(str(frame))
                if image is None:
                    raise OSError(f"Could not read {frame}")
                image_hw = image.shape[:2]
                # Detector boxes were rescaled once during capture. A unit-scale
                # tracker call preserves the same coordinates in every mode.
                tracks = tracker.update(boxes.copy(), image_hw, image_hw)
                total_rows += record_frame(
                    image, tracks, mode, frame_number, frame_dir, text_file, min_box_area,
                )

        keys = ("predicted_occluded_tracks", "observed_occluded_tracks",
                "oao_solver_calls", "bam_updates", "bam_damped_updates")
        stats = getattr(tracker, "oa_stats", Counter())
        summary[mode] = {"track_rows": total_rows, **{key: stats[key] for key in keys}}
        print(f"{mode}: {summary[mode]}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--frames", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--modes", required=True, nargs="+", choices=MODE_FLAGS)
    parser.add_argument("--tau", type=float, default=0.15)
    parser.add_argument("--depth-pixels", type=float, default=5)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no-gaussian", action="store_true")
    options = parser.parse_args()

    if len(set(options.modes)) != len(options.modes):
        parser.error("--modes contains a duplicate.")
    frames = sorted(path.resolve() for path in options.frames.iterdir()
                    if path.suffix.lower() in IMAGE_SUFFIXES)
    if not frames:
        raise FileNotFoundError(f"No input images at {options.frames}")
    config = OAConfig(tau=options.tau, depth_pixels=options.depth_pixels,
                      gaussian=not options.no_gaussian)
    options.output.mkdir(parents=True, exist_ok=True)

    base = importlib.import_module("trackers.hybrid_sort_tracker.hybrid_sort")
    kf = importlib.import_module("trackers.hybrid_sort_tracker.kalmanfilter_score_new")
    detections, tracker_spec = collect_detections(options.repo, frames, options.weights, options.fp16)
    np.savez_compressed(
        options.output / "shared_detections.npz",
        frame_names=np.array([path.name for path in frames]),
        **{f"frame_{index:08d}": boxes for index, boxes in enumerate(detections, 1)},
    )
    summary = replay(
        frames, detections, tracker_spec, base, kf.KalmanFilterNew_score_new,
        options.modes, config, options.output, options.sequence,
    )
    report = {"spatial_metric": "Height_Modulated_IoU", "use_byte": True,
              "processed_frames": len(frames),
              "frames_without_detections": sum(len(boxes) == 0 for boxes in detections),
              "modes": summary}
    (options.output / "tracking_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
