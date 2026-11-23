
import json
from pathlib import Path
import sys

import numpy as np
import trackeval

work = Path(sys.argv[1])
modes = ["hybrid", "oa_sort"]
manifest = json.loads((work / "run_manifest.json").read_text(encoding="utf-8"))
dataset_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
dataset_cfg.update(
    GT_FOLDER=str(work / "gt"),
    TRACKERS_FOLDER=str(work / "trackers"),
    OUTPUT_FOLDER=str(work / "trackeval"),
    TRACKERS_TO_EVAL=modes,
    TRACKER_SUB_FOLDER="data",
    CLASSES_TO_EVAL=["pedestrian"],
    BENCHMARK="DanceTrack",
    SPLIT_TO_EVAL="val",
    SKIP_SPLIT_FOL=True,
    SEQMAP_FILE=str(work / "seqmap.txt"),
    PRINT_CONFIG=False,
)
eval_cfg = trackeval.Evaluator.get_default_eval_config()
eval_cfg.update(
    USE_PARALLEL=False, PRINT_CONFIG=False, PRINT_RESULTS=False,
    OUTPUT_SUMMARY=True, OUTPUT_DETAILED=True, PLOT_CURVES=False,
    TIME_PROGRESS=False, RETURN_ON_ERROR=True,
)
dataset = trackeval.datasets.MotChallenge2DBox(dataset_cfg)
metrics = [
    trackeval.metrics.HOTA(),
    trackeval.metrics.CLEAR({"THRESHOLD": 0.5, "PRINT_CONFIG": False}),
    trackeval.metrics.Identity({"THRESHOLD": 0.5, "PRINT_CONFIG": False}),
]
results, messages = trackeval.Evaluator(eval_cfg).evaluate([dataset], metrics)


def metric_values(result):
    hota, clear, identity = (result[key] for key in ("HOTA", "CLEAR", "Identity"))
    return {
        "HOTA": 100 * float(np.mean(hota["HOTA"])),
        "AssA": 100 * float(np.mean(hota["AssA"])),
        "DetA": 100 * float(np.mean(hota["DetA"])),
        "IDF1": 100 * float(identity["IDF1"]),
        "MOTA": 100 * float(clear["MOTA"]),
        "IDSW": int(clear["IDSW"]),
        "FP": int(clear["CLR_FP"]),
        "FN": int(clear["CLR_FN"]),
        "Frag": int(clear["Frag"]),
    }


dataset_name = dataset.get_name()
report = {}
for mode in modes:
    mode_results = results[dataset_name][mode]
    if mode_results is None:
        raise RuntimeError(f"TrackEval {mode}: {messages[dataset_name][mode]}")
    report[mode] = {
        sequence: metric_values(mode_results[sequence]["pedestrian"])
        for sequence in ("COMBINED_SEQ", *manifest["sequence_frames"])
    }
(work / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
