
"""Paper reimplementation of OA-SORT on top of upstream Hybrid-SORT."""
import ast
from collections import Counter
from dataclasses import dataclass
import inspect
import math
from textwrap import dedent
from types import FunctionType

import numpy as np


@dataclass(frozen=True)
class OAConfig:
    use_oao: bool = True
    use_bam: bool = True
    gaussian: bool = True
    tau: float = 0.15
    depth_pixels: float = 5.0
    kx: float = 3 * math.sqrt(2)
    ky: float = 3.0

    def __post_init__(self):
        if not all(math.isfinite(value) for value in (self.tau, self.depth_pixels, self.kx, self.ky)):
            raise ValueError("OA-SORT parameters must be finite.")
        if not 0 <= self.tau <= 1 or min(self.depth_pixels, self.kx, self.ky) <= 0:
            raise ValueError("Require 0 <= tau <= 1 and positive depth_pixels, kx and ky.")


def occlusion_coefficients(boxes, image_hw, config):
    """OAM (Eq. 2–7). Boxes are N×4 xyxy in original image pixels."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    coefficients = np.zeros(len(boxes), dtype=np.float32)
    height, width = map(int, image_hw)
    if len(boxes) < 2 or min(height, width) <= 0:
        return coefficients

    finite = np.isfinite(boxes).all(axis=1)
    # Use floor/ceil for the image slices and floating-point bottom edges for depth.
    clipped = np.clip(np.where(finite[:, None], boxes, 0), 0, [width, height, width, height])
    starts = np.floor(clipped[:, :2]).astype(np.intp)
    ends = np.ceil(clipped[:, 2:]).astype(np.intp)
    valid = finite & (clipped[:, 2:] > clipped[:, :2]).all(axis=1)
    overlap_start = np.maximum(starts[:, None, :], starts[None, :, :])
    overlap_end = np.minimum(ends[:, None, :], ends[None, :, :])
    overlap = (overlap_end > overlap_start).all(axis=2) & valid[:, None] & valid[None, :]
    # Suppress pixel quantization overlaps between geometrically disjoint boxes.
    overlap &= (np.minimum(clipped[:, None, 2:], clipped[None, :, 2:])
                > np.maximum(clipped[:, None, :2], clipped[None, :, :2])).all(axis=2)
    np.fill_diagonal(overlap, False)
    bottoms = clipped[:, 3]
    occludes = overlap & ((bottoms[None, :] - bottoms[:, None]) >= config.depth_pixels)
    blocked = np.flatnonzero(occludes.any(axis=1))
    if not len(blocked):
        return coefficients

    if config.gaussian:
        gaussian_map = np.zeros((height, width), dtype=np.float32)
        for index in np.flatnonzero(overlap.any(axis=1)):
            left, top = starts[index]
            right, bottom = ends[index]
            center = (clipped[index, :2] + clipped[index, 2:]) * 0.5
            sigma = (clipped[index, 2:] - clipped[index, :2]) / [config.kx, config.ky]
            dx = (np.arange(left, right) + 0.5 - center[0]) / sigma[0]
            dy = (np.arange(top, bottom) + 0.5 - center[1]) / sigma[1]
            weights = np.exp(-0.5 * (dy[:, None] ** 2 + dx[None, :] ** 2))
            region = gaussian_map[top:bottom, left:right]
            np.maximum(region, weights, out=region)

    for index in blocked:
        left, top = starts[index]
        right, bottom = ends[index]
        mask = np.zeros((bottom - top, right - left), dtype=bool)
        for other in np.flatnonzero(occludes[index]):
            x1, y1 = overlap_start[index, other]
            x2, y2 = overlap_end[index, other]
            mask[y1 - top:y2 - top, x1 - left:x2 - left] = True
        covered = (gaussian_map[top:bottom, left:right][mask].sum(dtype=np.float64)
                   if config.gaussian else mask.sum())
        coefficients[index] = covered / mask.size
    return coefficients


def occlusion_aware_similarity(similarity, predicted_occlusion, tau):
    """OAO (Eq. 8). Columns represent predicted tracks."""
    return (1 - tau) * similarity + tau * (1 - predicted_occlusion[None, :])


def bias_aware_weight(spatial_similarity, last_observed_occlusion):
    """BAM (Eq. 9), with HMIoU following the paper's Hybrid-SORT experiment."""
    return float(np.clip(spatial_similarity, 0, 1) * (1 - last_observed_occlusion))


def damp_geometry_residual(residual, weight):
    # Hybrid-SORT measurement layout: x, y, area, confidence, aspect ratio.
    if residual.shape[0] != 5:
        raise RuntimeError(f"Expected the 5D Hybrid-SORT measurement, got {residual.shape}.")
    residual[[0, 1, 2, 4]] *= weight


def method_ast(function):
    return ast.parse(dedent(inspect.getsource(function)))


def compile_method(tree, original, **symbols):
    namespace = {**original.__globals__, **symbols}
    ast.fix_missing_locations(tree)
    exec(compile(tree, f"<OA integration: {original.__qualname__}>", "exec"), namespace)
    return namespace[original.__name__]


def build_kf_update(original):
    """Insert BAM after the KF innovation is computed, including after ORU."""
    tree = method_ast(original)
    method = tree.body[0]
    residuals = [
        index for index, node in enumerate(method.body)
        if isinstance(node, ast.Assign)
        and any(ast.unparse(target) == "self.y" for target in node.targets)
        and isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Sub)
    ]
    if len(residuals) != 1:
        raise RuntimeError("Hybrid-SORT KF update changed. Could not locate the innovation.")
    method.args.kwonlyargs.append(ast.arg(arg="oa_weight"))
    method.args.kw_defaults.append(None)
    method.body.insert(residuals[0] + 1, ast.parse("damp_geometry_residual(self.y, oa_weight)").body[0])
    return compile_method(tree, original, damp_geometry_residual=damp_geometry_residual)


def build_track_update(original, kf_update):
    """Keep the raw observation in Hybrid-SORT's motion and confidence histories."""
    tree = method_ast(original)
    method = tree.body[0]
    locations = [
        node for node in ast.walk(tree) if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "self.kf.update" and len(node.args) == 1
        and ast.unparse(node.args[0]) == "convert_bbox_to_z(bbox)"
    ]
    if len(locations) != 1:
        raise RuntimeError("Hybrid-SORT track update changed. Could not attach BAM.")
    method.args.kwonlyargs.append(ast.arg(arg="oa_weight"))
    method.args.kw_defaults.append(None)
    target = locations[0]
    target.func = ast.Name(id="oa_kf_update", ctx=ast.Load())
    target.args.insert(0, ast.parse("self.kf", mode="eval").body)
    target.keywords.append(ast.keyword(arg="oa_weight", value=ast.Name(id="oa_weight", ctx=ast.Load())))
    return compile_method(tree, original, oa_kf_update=kf_update)


def build_tracker_update(original):
    """Attach OAO at stage 1 and BAM at accepted stage-2 matches only."""
    tree = method_ast(original)
    high_calls = 0
    low_calls = 0
    for node in list(ast.walk(tree)):
        if not isinstance(node, ast.Call):
            continue
        called = ast.unparse(node.func)
        if called in {"associate_4_points_with_score", "associate_4_points"}:
            node.args.insert(0, node.func)
            node.func = ast.parse("self._associate_high", mode="eval").body
            high_calls += 1
        elif (called == "self.trackers[trk_ind].update" and node.args
              and ast.unparse(node.args[0]).startswith("dets_second[")):
            node.func = ast.parse("self._update_low", mode="eval").body
            node.args.insert(0, ast.Name(id="trk_ind", ctx=ast.Load()))
            node.args.append(ast.parse("trks[trk_ind, :4]", mode="eval").body)
            low_calls += 1
    if (high_calls, low_calls) != (2, 1):
        raise RuntimeError(
            f"Hybrid-SORT update changed: found {high_calls} high and {low_calls} low hooks; expected 2 and 1."
        )
    return compile_method(tree, original)


def make_oa_tracker(base, kf_class):
    """Construct a subclass using the installed Hybrid-SORT implementation."""
    original_update = build_tracker_update(base.Hybrid_Sort.update)
    bam_track_update = build_track_update(
        base.KalmanBoxTracker.update, build_kf_update(kf_class.update)
    )
    spatial_hmiou = base.ASSO_FUNCS["Height_Modulated_IoU"]

    class OASort(base.Hybrid_Sort):
        def __init__(self, *args, oa_config, **kwargs):
            super().__init__(*args, **kwargs)
            self.oa_config = oa_config
            self.oa_stats = Counter()
            if oa_config.use_bam and not self.use_byte:
                raise RuntimeError("BAM needs Hybrid-SORT's low-score matching (use_byte=True).")

        def update(self, detections, img_info, img_size):
            self.image_hw = img_info[:2]
            tracks = original_update(self, detections, img_info, img_size)
            if self.oa_config.use_bam:
                # Algorithm 5 uses matched tracks. A newly initialized track still
                # has Hybrid-SORT's negative-score last_observation sentinel.
                observed = [track for track in self.trackers if track.time_since_update == 0
                            and track.last_observation[4] >= 0]
                boxes = [track.last_observation[:4] for track in observed]
                coefficients = occlusion_coefficients(boxes, self.image_hw, self.oa_config)
                for track, coefficient in zip(observed, coefficients):
                    track.oa_last_observed = float(coefficient)
                self.oa_stats["observed_occluded_tracks"] += int(np.count_nonzero(coefficients))
            return tracks

        def _associate_high(self, original, detections, predictions, *args, **kwargs):
            config = self.oa_config
            if not config.use_oao or config.tau == 0 or not len(detections) or not len(predictions):
                return original(detections, predictions, *args, **kwargs)

            coefficients = occlusion_coefficients(predictions[:, :4], self.image_hw, config)
            raw = self.asso_func(detections, predictions)
            adjusted = occlusion_aware_similarity(raw, coefficients, config.tau)
            self.oa_stats["predicted_occluded_tracks"] += int(np.count_nonzero(coefficients))
            original_assignment = original.__globals__.get("linear_assignment")
            if original_assignment is None or "linear_assignment" not in original.__code__.co_names:
                raise RuntimeError("Hybrid-SORT stage 1 no longer calls linear_assignment.")

            def assign_with_oao(cost):
                if cost.shape != raw.shape:
                    raise RuntimeError("Unexpected shape of the stage-1 assignment matrix.")
                self.oa_stats["oao_solver_calls"] += 1
                # Cost is minimized. Swap only the spatial term, retaining the
                # track direction and confidence terms and the upstream gate.
                return original_assignment(cost + raw - adjusted)

            namespace = {**original.__globals__, "linear_assignment": assign_with_oao}
            with_oao = FunctionType(
                original.__code__, namespace, original.__name__,
                original.__defaults__, original.__closure__,
            )
            with_oao.__kwdefaults__ = original.__kwdefaults__
            return with_oao(detections, predictions, *args, **kwargs)

        def _update_low(self, track_index, detection, prediction):
            track = self.trackers[track_index]
            if not self.oa_config.use_bam:
                track.update(detection)
                return

            similarity = spatial_hmiou(detection[None, :4], prediction[None, :4])[0, 0]
            weight = bias_aware_weight(similarity, getattr(track, "oa_last_observed", 0.0))
            bam_track_update(track, detection, oa_weight=weight)
            self.oa_stats["bam_updates"] += 1
            if weight < 1:
                self.oa_stats["bam_damped_updates"] += 1

    return OASort
