#!/usr/bin/env bash
# WSL2(Ubuntu) / Linux 에서 baseline 노트북과 같은 환경을 구성한다.
#   bash setup_wsl.sh            # 전체 (패키지, HybridSORT, venv, 재현 코드 추출, 데이터, 가중치)
#   STEP=data bash setup_wsl.sh  # 특정 단계만: apt | repo | venv | code | data | weights | trackeval
# 결과 위치: OCC_ROOT (기본 ~/occ)
set -euo pipefail

OCC_ROOT="${OCC_ROOT:-$HOME/occ}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OCCFLOW="$(dirname "$HERE")"                                   # occflow 저장소
NOTEBOOK="${NOTEBOOK:-$(dirname "$OCCFLOW")/baseline/OA-SORT_DanceTrack_Val_Colab.ipynb}"
REPO="$OCC_ROOT/HybridSORT"
VENV="$OCC_ROOT/venv"
PY="$VENV/bin/python"
CODE_DIR="$OCC_ROOT/code_dir"
DATA="$OCC_ROOT/data"
STEP="${STEP:-all}"
mkdir -p "$OCC_ROOT" "$DATA"
export PATH="$HOME/.local/bin:$PATH"

want() { [[ "$STEP" == all || "$STEP" == "$1" ]]; }

if want apt; then
  echo "== apt"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq git curl unzip build-essential ca-certificates >/dev/null
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if want repo; then
  echo "== HybridSORT"
  [[ -d "$REPO/.git" ]] || git clone -q https://github.com/ymzis69/HybridSORT.git "$REPO"
  git -C "$REPO" rev-parse HEAD
fi

if want venv; then
  echo "== venv (python 3.11, baseline 노트북과 같은 버전)"
  [[ -x "$PY" ]] || uv venv -q --python 3.11 "$VENV"
  uv pip install -q --python "$PY" --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0 torchvision==0.21.0
  uv pip install -q --python "$PY" \
    numpy==1.23.5 scipy==1.13.1 opencv-python-headless==4.10.0.84 lap==0.5.12 filterpy==1.4.5 \
    scikit-image==0.22.0 pandas==2.2.3 matplotlib==3.7.5 scikit-learn==1.5.2 cython==0.29.37 \
    setuptools==69.5.1 wheel loguru tqdm pillow thop ninja tabulate tensorboard h5py xmltodict yacs \
    easydict termcolor fvcore iopath pyyaml prettytable einops faiss-cpu==1.8.0.post1 gdown==6.1.0
  uv pip install -q --python "$PY" --no-build-isolation cython-bbox==0.1.5 pycocotools==2.0.7
  PYTHONPATH="$REPO" "$PY" -c "import torch, lap, cython_bbox, numpy; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'numpy', numpy.__version__)"
fi

if want code; then
  echo "== 노트북에서 재현 코드 추출 -> $CODE_DIR"
  mkdir -p "$CODE_DIR"
  python3 - "$NOTEBOOK" "$CODE_DIR" <<'EOF'
import json, re, sys
from pathlib import Path
nb = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
out = Path(sys.argv[2])
src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
for var, name in (("OA_SOURCE", "oa_sort_impl.py"), ("RUNNER_SOURCE", "oa_sort_val_runner.py"),
                  ("EVAL_SOURCE", "evaluate_trackeval.py")):
    m = re.search(var + r" = r'''(.*?)'''", src, re.S)
    if not m:
        raise SystemExit("notebook: %s not found" % var)
    (out / name).write_text(m.group(1), encoding="utf-8")
    print("  wrote", out / name)
EOF
fi

if want data; then
  echo "== DanceTrack val (약 4.2GB)"
  if [[ ! -d "$DATA/val" ]]; then
    # 연결이 끊기거나 30초 동안 50KB/s 미만이면 끊고 이어받기 (.part 유지)
    for i in $(seq 1 50); do
      curl --fail --location --silent --show-error --continue-at - --retry 5 --retry-all-errors \
        --retry-delay 5 --speed-limit 50000 --speed-time 30 -o "$DATA/val.zip.part" \
        https://huggingface.co/datasets/noahcao/dancetrack/resolve/main/val.zip && break
      echo "  retry $i ($(du -h "$DATA/val.zip.part" | cut -f1))"; sleep 5
    done
    unzip -tq "$DATA/val.zip.part" >/dev/null
    mv "$DATA/val.zip.part" "$DATA/val.zip"
    unzip -q "$DATA/val.zip" -d "$DATA/"
    rm -f "$DATA/val.zip"
  fi
  curl -fsSL -o "$DATA/val_seqmap.txt" \
    https://raw.githubusercontent.com/DanceTrack/DanceTrack/main/dancetrack/val_seqmap.txt
  ls "$DATA/val" | head -3; ls "$DATA/val" | wc -l
fi

if want weights; then
  echo "== YOLOX 가중치 (Hybrid-SORT README 의 ocsort_dance_model.pth.tar)"
  mkdir -p "$REPO/pretrained"
  CK="$REPO/pretrained/ocsort_dance_model.pth.tar"
  if [[ ! -s "$CK" ]]; then
    "$PY" - "$CK" <<'EOF'
import pathlib, sys, gdown
target = pathlib.Path(sys.argv[1]); tmp = target.with_suffix(".part")
for folder in ("https://drive.google.com/drive/folders/18IsZGeGiyKDshhYIzbpYXoNEcBhPY8lN",
               "https://drive.google.com/drive/folders/1LnhZVJlpufUnWuObZASIN1KwfhuvT_a8"):
    try:
        entries = gdown.download_folder(url=folder, skip_download=True, quiet=True)
        e = next(x for x in entries if pathlib.Path(x.path).name == target.name)
        gdown.download(id=e.id, output=str(tmp), quiet=False)
        if tmp.stat().st_size > 1_000_000:
            tmp.replace(target); print("  ok", target); break
    except Exception as exc:
        print("  failed", folder, exc)
else:
    raise SystemExit("가중치 자동 다운로드 실패: 파일을 직접 받아 %s 에 두세요." % target)
EOF
  fi
  sha256sum "$CK"
fi

if want trackeval; then
  echo "== TrackEval venv"
  [[ -x "$OCC_ROOT/trackeval_venv/bin/python" ]] || uv venv -q --python 3.11 "$OCC_ROOT/trackeval_venv"
  uv pip install -q --python "$OCC_ROOT/trackeval_venv/bin/python" trackeval==1.0.0
fi
echo "done ($STEP)"
