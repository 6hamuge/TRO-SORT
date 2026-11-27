"""TrackEval(HOTA/CLEAR/Identity)로 여러 모드를 평가. baseline 노트북 evaluate_trackeval.py 의 모드 일반화판.

  python evaluate_modes.py WORK hybrid,oa_sort,oa_null,oa_flow
WORK 구조: WORK/gt/<seq>/{seqinfo.ini,gt/gt.txt}, WORK/trackers/<mode>/data/<seq>.txt, WORK/seqmap.txt
"""
import json
import sys
from pathlib import Path

import numpy as np
import trackeval

work = Path(sys.argv[1])
modes = [m for m in sys.argv[2].split(",") if m]
dataset_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
dataset_cfg.update(
    GT_FOLDER=str(work / "gt"), TRACKERS_FOLDER=str(work / "trackers"),
    OUTPUT_FOLDER=str(work / "trackeval"), TRACKERS_TO_EVAL=modes, TRACKER_SUB_FOLDER="data",
    CLASSES_TO_EVAL=["pedestrian"], BENCHMARK="DanceTrack", SPLIT_TO_EVAL="val",
    SKIP_SPLIT_FOL=True, SEQMAP_FILE=str(work / "seqmap.txt"), PRINT_CONFIG=False,
)
eval_cfg = trackeval.Evaluator.get_default_eval_config()
eval_cfg.update(USE_PARALLEL=False, PRINT_CONFIG=False, PRINT_RESULTS=False, OUTPUT_SUMMARY=True,
                OUTPUT_DETAILED=True, PLOT_CURVES=False, TIME_PROGRESS=False, RETURN_ON_ERROR=True)
dataset = trackeval.datasets.MotChallenge2DBox(dataset_cfg)
metrics = [trackeval.metrics.HOTA(),
           trackeval.metrics.CLEAR({"THRESHOLD": 0.5, "PRINT_CONFIG": False}),
           trackeval.metrics.Identity({"THRESHOLD": 0.5, "PRINT_CONFIG": False})]
results, messages = trackeval.Evaluator(eval_cfg).evaluate([dataset], metrics)


def metric_values(r):
    hota, clear, identity = r["HOTA"], r["CLEAR"], r["Identity"]
    return {"HOTA": 100 * float(np.mean(hota["HOTA"])), "AssA": 100 * float(np.mean(hota["AssA"])),
            "DetA": 100 * float(np.mean(hota["DetA"])), "IDF1": 100 * float(identity["IDF1"]),
            "MOTA": 100 * float(clear["MOTA"]), "IDSW": int(clear["IDSW"]), "FP": int(clear["CLR_FP"]),
            "FN": int(clear["CLR_FN"]), "Frag": int(clear["Frag"])}


name = dataset.get_name()
seqs = [s.strip() for s in (work / "seqmap.txt").read_text(encoding="utf-8").splitlines()[1:] if s.strip()]
report = {}
for mode in modes:
    res = results[name][mode]
    if res is None:
        raise RuntimeError("TrackEval %s: %s" % (mode, messages[name][mode]))
    report[mode] = {s: metric_values(res[s]["pedestrian"]) for s in ("COMBINED_SEQ", *seqs)}
(work / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps({m: report[m]["COMBINED_SEQ"] for m in modes}, indent=1))
