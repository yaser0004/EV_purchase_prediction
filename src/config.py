"""Shared config for S6E9 EV purchase prediction (GBDT-only, reusable)."""
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
REPO_DIR = SRC_DIR.parent
DATA_RAW = REPO_DIR / "data" / "raw"
DATA_PROCESSED = REPO_DIR / "data" / "processed"
OUTPUTS = REPO_DIR / "outputs"
SUBMISSIONS = REPO_DIR / "submissions"

TARGET = "Will_Buy_EV"
ID_COL = "id"

N_SPLITS = 5
SEEDS = (42, 7, 2026)
N_JOBS = 4  # capped: 7GB RAM laptop, keep folds sequential

COMPETITION = "playground-series-s6e9"

for d in (DATA_RAW, DATA_PROCESSED, OUTPUTS, SUBMISSIONS):
    d.mkdir(parents=True, exist_ok=True)
