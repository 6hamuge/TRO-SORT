"""재현 OA-SORT 위에 가림 흐름 훅을 얹은 추적기 (OAFlowSort).

재현 코드(oa_sort_impl.py)와 같은 방식(AST 수정)으로 Hybrid-SORT update 를 다시 컴파일한다.
재현 코드 대비 추가된 것은 두 군데뿐이다.

  1) 1차 매칭 linear_assignment 직전 비용:  cost + raw - adjusted (OAO, 재현과 동일한 식)
                                            -> + 가림 흐름 보정 (hook.adjust)
  2) 'for m in unmatched_trks: update(None)' 직전:
        unmatched_dets, unmatched_trks = self._flow_recover(...)   (hook.recover)

update() 반환 직후에 ID 사후 점검(hook.audit)과 이력 갱신(hook.observe)을 한다.
모든 모듈의 효과를 0으로 둔 'null' 설정에서 출력이 재현 OA-SORT 와 완전히 같아야 한다.
"""
from __future__ import annotations

import ast

import numpy as np

from oa_sort_impl import (build_high_association, compile_method, make_oa_tracker,  # noqa: E402
                          method_ast, occlusion_aware_similarity, occlusion_coefficients)

from ..hook import OcclusionFlowHook

RECOVER_STMT = "unmatched_dets, unmatched_trks = self._flow_recover(dets, unmatched_dets, unmatched_trks)"


def build_flow_update(original):
    """oa_sort_impl.build_tracker_update 와 같은 수정 + 재연결 훅 삽입."""
    tree = method_ast(original)
    method = tree.body[0]
    high_calls = 0
    low_calls = 0
    # --- (재현 코드와 동일) 1차 매칭 -> self._associate_high, BYTE 갱신 -> self._update_low
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
        raise RuntimeError("Hybrid-SORT update changed: found %d high / %d low hooks" % (high_calls, low_calls))

    # --- (추가) 미매칭 트랙을 missed 처리하기 직전에 재연결 훅
    spots = [i for i, st in enumerate(method.body)
             if isinstance(st, ast.For) and ast.unparse(st.iter) == "unmatched_trks"
             and "update(None)" in ast.unparse(st)]
    if len(spots) != 1:
        raise RuntimeError("Could not locate 'for m in unmatched_trks: ...update(None)' in Hybrid-SORT update.")
    method.body.insert(spots[0], ast.parse(RECOVER_STMT).body[0])
    return compile_method(tree, original)


def make_flow_tracker(base, kf_class, oa_tracker_class=None):
    OASort = oa_tracker_class or make_oa_tracker(base, kf_class)
    flow_update = build_flow_update(base.Hybrid_Sort.update)
    g = base.Hybrid_Sort.update.__globals__
    high_associations = {fn: build_high_association(fn)
                         for fn in (g["associate_4_points_with_score"], g["associate_4_points"])}

    class OAFlowSort(OASort):
        def __init__(self, *args, flow_config, **kwargs):
            super().__init__(*args, **kwargs)
            self.flow_config = flow_config
            self.flow_hook = (OcclusionFlowHook(flow_config.flow, flow_config.recovery, flow_config.audit)
                              if flow_config.any_enabled else None)
            self.flow_assoc_logs = []
            self.flow_events = []
            self._flow_recovered = set()

        # ------------------------------------------------------------ update
        def update(self, detections, img_info, img_size):
            self.image_hw = img_info[:2]
            self._flow_recovered = set()
            tracks = flow_update(self, detections, img_info, img_size)
            # ---- 아래 BAM 기록은 oa_sort_impl.OASort.update 와 동일 ----
            if self.oa_config.use_bam:
                observed = [track for track in self.trackers if track.time_since_update == 0
                            and track.last_observation[4] >= 0]
                boxes = [track.last_observation[:4] for track in observed]
                coefficients = occlusion_coefficients(boxes, self.image_hw, self.oa_config)
                for track, coefficient in zip(observed, coefficients):
                    track.oa_last_observed = float(coefficient)
                self.oa_stats["observed_occluded_tracks"] += int(np.count_nonzero(coefficients))
            # ---- 추가: ID 사후 점검 + 이력 갱신 ----
            if self.flow_hook is not None:
                tracks = self._flow_post(tracks)
            return tracks

        # --------------------------------------------------- 1) 비용 보정
        def _associate_high(self, original, detections, predictions, *args, **kwargs):
            hook = self.flow_hook
            if hook is None or not hook.cfg.enable or not len(detections) or not len(predictions):
                return super()._associate_high(original, detections, predictions, *args, **kwargs)
            config = self.oa_config
            raw = self.asso_func(detections, predictions)          # (det, trk) HMIoU
            use_oao = config.use_oao and config.tau != 0
            if use_oao:
                coefficients = occlusion_coefficients(predictions[:, :4], self.image_hw, config)
                adjusted = occlusion_aware_similarity(raw, coefficients, config.tau)
                self.oa_stats["predicted_occluded_tracks"] += int(np.count_nonzero(coefficients))
            frame = int(self.frame_count)
            track_ids = [int(t.id) for t in self.trackers]
            if len(track_ids) != len(predictions):
                raise RuntimeError("tracker list and prediction rows are misaligned")
            valid_t = (np.asarray(raw) >= self.iou_threshold).T     # 원래 최종 gate 와 동일
            box = {}

            def adjust_cost(cost):
                if cost.shape != raw.shape:
                    raise RuntimeError("Unexpected shape of the stage-1 assignment matrix.")
                if use_oao:
                    self.oa_stats["oao_solver_calls"] += 1
                    cost = cost + raw - adjusted                    # 재현 OAO 와 같은 식/순서
                out_t, logs = hook.adjust(frame, track_ids, predictions[:, :4], detections[:, :4],
                                          cost.T, valid_t)
                box["logs"] = logs
                return out_t.T

            result = high_associations[original](detections, predictions, *args, **kwargs,
                                                 oa_adjust_cost=adjust_cost)
            logs = box.get("logs", [])
            if logs:
                matches = np.asarray(result[0]).reshape(-1, 2)
                chosen = {(int(d), int(t)) for d, t in matches}
                for lg in logs:
                    lg["chosen"] = int((lg["det_idx"], lg["row"]) in chosen)
                    lg["track_id"] += 1          # MOT 출력 ID 기준으로 기록
                    lg["partner_id"] += 1
                self.flow_assoc_logs.extend(logs)
            return result

        # ----------------------------------------------------- 2) 재연결
        def _flow_recover(self, dets, unmatched_dets, unmatched_trks):
            hook = self.flow_hook
            if hook is None or not hook.rec.enable:
                return unmatched_dets, unmatched_trks
            ud = np.asarray(unmatched_dets, dtype=int).reshape(-1)
            ut = np.asarray(unmatched_trks, dtype=int).reshape(-1)
            if not len(ud) or not len(ut):
                return unmatched_dets, unmatched_trks
            matched_boxes = {int(t.id): np.asarray(t.last_observation[:4], dtype=float)
                             for t in self.trackers
                             if t.time_since_update == 0 and t.last_observation.sum() >= 0}
            cand = [int(i) for i in ut if self.trackers[i].last_observation.sum() >= 0]
            lost = [(int(self.trackers[i].id), np.asarray(self.trackers[i].last_observation[:4], dtype=float),
                     int(self.trackers[i].time_since_update)) for i in cand]
            res = hook.recover(int(self.frame_count), lost, dets[ud, :4], matched_boxes)
            if not res:
                return unmatched_dets, unmatched_trks
            rm_d, rm_t = [], []
            for a, b, m in res:
                ti, di = cand[a], int(ud[b])
                trk = self.trackers[ti]
                partner = hook.histories[int(trk.id)].partner_id
                self.flow_events.append(dict(frame=int(self.frame_count), event="recover",
                                             track_id=int(trk.id) + 1, other_id=int(partner) + 1,
                                             value=float(m), lost_frames=int(trk.time_since_update),
                                             start_frame=-1))
                trk.update(dets[di, :])
                self._flow_recovered.add(int(trk.id))
                rm_d.append(di)
                rm_t.append(ti)
            return (np.setdiff1d(ud, np.asarray(rm_d, dtype=int)),
                    np.setdiff1d(ut, np.asarray(rm_t, dtype=int)))

        # ------------------------------------------- 3) 사후 점검 + 4) 이력
        def _flow_observations(self):
            obs = []
            for t in self.trackers:
                lo = t.last_observation
                if t.time_since_update != 0 or lo.sum() < 0:
                    continue
                cfg = self.flow_hook.cfg
                trusted = (lo[4] > cfg.trust_min_score and t.hits >= cfg.trust_min_hits
                           and int(t.id) not in self._flow_recovered)
                obs.append((int(t.id), np.asarray(lo[:4], dtype=float), bool(trusted)))
            return obs

        def _flow_post(self, tracks):
            hook = self.flow_hook
            frame = int(self.frame_count)
            if hook.aud.enable:
                mb = {tid: b for tid, b, ok in self._flow_observations() if ok}
                for a, b, start, cur, swp in hook.audit(frame, mb):
                    ta = next((t for t in self.trackers if int(t.id) == a), None)
                    tb = next((t for t in self.trackers if int(t.id) == b), None)
                    if ta is None or tb is None:
                        continue
                    ta.id, tb.id = b, a
                    if len(tracks):
                        col = tracks[:, 4].copy()
                        tracks[col == a + 1, 4] = b + 1
                        tracks[col == b + 1, 4] = a + 1
                    self.flow_events.append(dict(frame=frame, event="audit_swap", track_id=a + 1,
                                                 other_id=b + 1, value=float(cur - swp),
                                                 lost_frames=-1, start_frame=int(start)))
            hook.observe(frame, self._flow_observations())
            hook.prune([int(t.id) for t in self.trackers])
            return tracks

    return OAFlowSort
