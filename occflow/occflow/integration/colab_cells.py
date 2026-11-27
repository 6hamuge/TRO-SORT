# %% [markdown]
# # occflow 실험 셀 (baseline 노트북 뒤에 붙여서 실행)
#
# 전제: baseline/OA-SORT_DanceTrack_Val_Colab.ipynb 의 1~7번 셀을 실행해
# `ROOT, REPO, PYTHON, CODE_DIR, BENCH_DIR, SEQUENCES, DATA, VAL_DIR, ZIP_PATH, checkpoint, VAL_CFG,
#  USE_FP16, OA_TAU, OA_DEPTH_PIXELS, OA_GAUSSIAN, BASE_ENV, run, uv, eval_python` 이 정의된 상태.
# occflow 폴더를 Colab에 올려 두고 OCCFLOW 경로를 맞춘다 (Drive 복사 또는 zip 업로드).

# %% A. 설정
from pathlib import Path
import json, shutil, subprocess, tempfile
from zipfile import ZipFile

OCCFLOW = Path("/content/occflow")                    # 이 저장소(occflow 폴더) 위치
FLOW_DIR = BENCH_DIR / "occflow"                      # 재현 결과 폴더 안에 따로 저장
DET_CACHE = FLOW_DIR / "det_cache"                    # 시퀀스별 YOLOX 검출 캐시 (1회 계산)
RUN_DIR = FLOW_DIR / "runs"
MODES = "hybrid,oa_sort,oa_null,oa_flow,oa_flow_rec,oa_flow_audit,oa_full"
FLOW_SET = []                                         # 예: ["flow.weight=0.5", "audit.confirm_frames=4"]
TAG = ""                                              # 하이퍼파라미터 탐색 시 폴더 접미사 (예: "_w05")
FOCUS = ["dancetrack0026", "dancetrack0079"]          # 먼저 볼 시퀀스. 전체는 SEQUENCES
RUN_SEQS = FOCUS
for d in (DET_CACHE, RUN_DIR):
    d.mkdir(parents=True, exist_ok=True)

# %% B. 검출 캐시 + 모드별 재생
def run_flow_sequence(name):
    cmd = [PYTHON, OCCFLOW / "occflow/integration/run_oa_flow.py", "--repo", REPO, "--code_dir", CODE_DIR,
           "--det_cache", DET_CACHE, "--output", RUN_DIR, "--sequence", name, "--modes", MODES,
           "--tau", OA_TAU, "--depth-pixels", OA_DEPTH_PIXELS]
    if TAG:
        cmd += ["--tag", TAG]
    if FLOW_SET:
        cmd += ["--set", *FLOW_SET]
    if not OA_GAUSSIAN:
        cmd.append("--no-gaussian")
    with tempfile.TemporaryDirectory(prefix=name + "-", dir=ROOT) as tmp:
        if not (DET_CACHE / name / "detections.npz").is_file():
            if VAL_DIR is None:                       # baseline run_sequence 와 같은 방식으로 프레임 추출
                frames_dir = Path(tmp) / "img1"
                frames_dir.mkdir()
                with ZipFile(ZIP_PATH) as archive:
                    for inner, image in zip(DATA[name]["entries"], DATA[name]["frames"]):
                        with archive.open(inner) as s, (frames_dir / image).open("wb") as o:
                            shutil.copyfileobj(s, o)
            else:
                frames_dir = VAL_DIR / name / "img1"
            cmd += ["--frames", frames_dir, "--weights", checkpoint, "--config", REPO / VAL_CFG]
            if USE_FP16:
                cmd.append("--fp16")
        log = RUN_DIR / ("%s.log" % name)
        with log.open("w", encoding="utf-8") as fh:
            proc = subprocess.run([str(c) for c in cmd], cwd=REPO, env={**BASE_ENV, "PYTHONUNBUFFERED": "1"},
                                  stdout=fh, stderr=subprocess.STDOUT)
        if proc.returncode:
            print("\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]))
            raise RuntimeError("%s 실패 (로그 %s)" % (name, log))
    print(name, json.loads((RUN_DIR / ("%s_summary.json" % name)).read_text())["modes"].keys())

for i, name in enumerate(RUN_SEQS, 1):
    print("[%d/%d] %s" % (i, len(RUN_SEQS), name), flush=True)
    run_flow_sequence(name)

# %% C. 동작 검증: (1) 재생 oa_sort == 재현 노트북 oa_sort, (2) oa_null == oa_sort
ck = OCCFLOW / "tools/check_equivalence.py"
run([PYTHON, ck, RUN_DIR / "oa_sort", BENCH_DIR / "trackers/oa_sort/data"], cwd=OCCFLOW)
run([PYTHON, ck, RUN_DIR / "oa_sort", RUN_DIR / "oa_null"], cwd=OCCFLOW)

# %% D. TrackEval (HOTA/AssA/IDF1/IDSW/Frag)
EVAL_DIR = FLOW_DIR / ("eval" + TAG)
modes = [m + (TAG if m.startswith("oa_") and m not in ("oa_sort", "oa_null") else "") for m in MODES.split(",")]
(EVAL_DIR / "gt").mkdir(parents=True, exist_ok=True)
for name in RUN_SEQS:
    shutil.copytree(BENCH_DIR / "gt" / name, EVAL_DIR / "gt" / name, dirs_exist_ok=True)
    for m in modes:
        (EVAL_DIR / "trackers" / m / "data").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(RUN_DIR / m / ("%s.txt" % name), EVAL_DIR / "trackers" / m / "data" / ("%s.txt" % name))
(EVAL_DIR / "seqmap.txt").write_text("name\n" + "\n".join(RUN_SEQS) + "\n", encoding="utf-8")
run([eval_python, OCCFLOW / "occflow/integration/evaluate_modes.py", EVAL_DIR, ",".join(modes)],
    env={**os.environ, "PYTHONPATH": "", "MPLBACKEND": "Agg"})

# %% E. 실패 유형 분석 (리뷰 의견: 0026/0079 에서 교환 vs 단절 구분, 프레임 범위, 전후 ID)
# 첫 번째가 비교 기준 -> oa_sort 를 맨 앞에 둔다 (hybrid 와 flow 모드가 모두 oa_sort 대비로 비교됨)
order = ["oa_sort"] + [m for m in modes if m not in ("oa_sort", "oa_null")]
res_args = ["%s=%s" % (m, RUN_DIR / m) for m in order]
run([PYTHON, OCCFLOW / "tools/analyze_failures.py", "--gt_root", BENCH_DIR / "gt", "--res", *res_args,
     "--det_root", DET_CACHE, "--seqs", *RUN_SEQS, "--out", FLOW_DIR / ("analysis" + TAG)], cwd=OCCFLOW)
