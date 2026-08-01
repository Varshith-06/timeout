#!/usr/bin/env bash
# Bootstrap all public datasets and pretrained weights for the NBA play recommender.
# Run from the repo root. Creates ./data and ./models.
#
# PREREQUISITES (do these manually first):
#   1. Kaggle API token -> https://www.kaggle.com/settings -> "Create New API Token"
#      Save as ~/.kaggle/kaggle.json, then: chmod 600 ~/.kaggle/kaggle.json
#   2. Python 3.11 venv activated.
#
# Nothing here requires an NDA. Nothing here is NBA broadcast footage —
# you source that yourself (see section 6 at the bottom).

set -euo pipefail
mkdir -p data/{raw,processed,priors,playbook,video} models
ROOT="$(pwd)"

# ---------------------------------------------------------------------------
# 1. NBA tracking data (SportVU 2015-16) — trains value models 11-14
# ---------------------------------------------------------------------------
# Two routes. Route A is the canonical source and is what I recommend.
# Route B is the HuggingFace mirror, which is friendlier but has a version trap.

# --- Route A: clone the raw .7z game logs directly (RECOMMENDED) ---
# ~600 games, one .7z per game, each unpacking to a single JSON.
echo "==> [1a] Cloning raw SportVU logs"
if [ ! -d data/raw/NBA-Player-Movements ]; then
  git clone --depth 1 https://github.com/linouk23/NBA-Player-Movements.git \
    data/raw/NBA-Player-Movements
fi

# Mirror with the half-court conversion + corrected shot timestamps.
# You want this one for `data/shots/fixed_shots.csv` and the flip scripts
# referenced in roadmap section 2.3.
echo "==> [1b] Cloning nba-movement-data (conversion scripts)"
if [ ! -d data/raw/nba-movement-data ]; then
  git clone --depth 1 https://github.com/sealneaward/nba-movement-data.git \
    data/raw/nba-movement-data
fi

pip install py7zr polars pyarrow

# Unpack a starter set of 5 games so you can start work immediately.
python - <<'PY'
import py7zr, pathlib, itertools
src = pathlib.Path("data/raw/NBA-Player-Movements/data/2016.NBA.Raw.SportVU.Game.Logs")
out = pathlib.Path("data/raw/sportvu_json"); out.mkdir(parents=True, exist_ok=True)
archives = sorted(src.glob("*.7z"))
if not archives:
    print("!! No .7z files found — check the repo layout, the path may have moved.")
else:
    for a in itertools.islice(archives, 5):
        print(f"   unpacking {a.name}")
        with py7zr.SevenZipFile(a, mode="r") as z:
            z.extractall(path=out)
    print(f"   {len(archives)} archives available total; unpacked 5.")
PY

# --- Route B: HuggingFace mirror (optional) ---
# TRAP: this repo ships a loading script. `datasets` >= 3.0 dropped support for
# script-based datasets entirely, so a fresh install will fail with a confusing
# error. You must pin below 3.0 AND pass trust_remote_code=True.
# Configs: tiny=5 games, small=25, medium=100, large=all 600+.
#
#   pip install "datasets<3.0.0" py7zr
#   python -c "
#   from datasets import load_dataset
#   ds = load_dataset('dcayton/nba_tracking_data_15_16', 'tiny', trust_remote_code=True)
#   print(ds)"
#
# Also note: the HF repo itself is only ~22 kB. The script pulls from the
# linouk23 GitHub repo at load time, so it is slow and breaks if that repo moves.
# And the HF schema is RESHAPED vs the raw files — see the note in the README
# this script ships alongside.

# ---------------------------------------------------------------------------
# 2. Current NBA stats — personnel priors, rosters, play-by-play
# ---------------------------------------------------------------------------
echo "==> [2] Installing nba_api"
pip install nba_api
# Endpoints you actually need:
#   shotchartdetail   -> per-player shot charts by zone (personnel priors)
#   playbyplayv2      -> event labels for possession segmentation
#   commonteamroster  -> jersey number -> player_id map for the OCR bridge
#   synergyplaytypes  -> play-type frequencies for the playbook layer
# Rate limits are real and undocumented. Sleep 0.6s between calls and set a
# browser-like User-Agent header or you will get silently throttled.

# ---------------------------------------------------------------------------
# 3. DeepSportRadar — basketball CV pretraining (re-ID, calibration)
# ---------------------------------------------------------------------------
echo "==> [3] Downloading DeepSportRadar basketball-instants"
pip install kaggle
if [ ! -d data/raw/basketball-instants-dataset ]; then
  kaggle datasets download deepsportradar/basketball-instants-dataset -p data/raw/
  unzip -qo data/raw/basketball-instants-dataset.zip -d data/raw/basketball-instants-dataset
fi
# COCO-format annotations for the instance segmentation split:
if [ ! -d data/raw/dsr-instance-seg ]; then
  git clone --depth 1 https://github.com/DeepSportradar/instance-segmentation-challenge.git \
    data/raw/dsr-instance-seg
fi
# Camera calibration task (closest public analogue to your homography problem):
if [ ! -d data/raw/dsr-calibration ]; then
  git clone --depth 1 https://github.com/DeepSportradar/camera-calibration-challenge.git \
    data/raw/dsr-calibration
fi

# ---------------------------------------------------------------------------
# 4. SoccerNet jersey numbers — pretraining for components 6 and 7
# ---------------------------------------------------------------------------
echo "==> [4] Downloading SoccerNet jersey tracklets"
pip install SoccerNet
python - <<'PY'
from SoccerNet.Downloader import SoccerNetDownloader as SNdl
dl = SNdl(LocalDirectory="data/raw/soccernet")
dl.downloadDataTask(task="jersey-2023", split=["train", "test", "challenge"])
PY
# 2853 train tracklets + 1211 challenge tracklets. Labels are a flat JSON dict:
# {player_id_string: jersey_number_int}, with -1 meaning "not visible".
# That -1 class is what you train the legibility classifier (component 6) on.

# Reference implementation — clone it, do not reimplement:
if [ ! -d src/vendor/jersey-number-pipeline ]; then
  mkdir -p src/vendor
  git clone --depth 1 https://github.com/mkoshkina/jersey-number-pipeline.git \
    src/vendor/jersey-number-pipeline
fi

# ---------------------------------------------------------------------------
# 5. Pretrained model weights
# ---------------------------------------------------------------------------
echo "==> [5] Fetching pretrained weights"

# Detection. RF-DETR is Apache 2.0 — use it if this may ever be licensed.
pip install rfdetr
# Ultralytics is AGPL-3.0. Fine for research, a real problem for a commercial tool.
pip install ultralytics
python - <<'PY'
from ultralytics import YOLO
YOLO("yolo11l.pt")        # detection init  (components 1, 2)
YOLO("yolo11l-pose.pt")   # keypoint init   (component 8, court keypoints)
PY

# Tracking + annotation utilities
pip install supervision

# Team classification (component 5) — SigLIP, zero-shot, no fine-tuning
pip install transformers umap-learn scikit-learn
python - <<'PY'
from transformers import AutoModel, AutoProcessor
AutoModel.from_pretrained("google/siglip-base-patch16-224")
AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
PY

# Body pose (component 9) — COCO-pretrained, zero-shot is fine
pip install mmpose mmcv mmengine   # ViTPose / RTMPose

# Clock OCR (component 10)
pip install paddlepaddle paddleocr

# Value models (components 11-14)
pip install lightgbm torch

echo
echo "Done. Fetched everything public."
echo "Not fetched, because it does not exist publicly:"
echo "  - NBA broadcast frames with detection labels  (you annotate ~2500)"
echo "  - NBA court keypoint annotations              (you annotate ~1000)"
echo "  - Any NBA tracking data after the 2015-16 season"
