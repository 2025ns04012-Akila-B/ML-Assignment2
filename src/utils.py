"""Shared constants and helper utilities for the NSL-KDD IDS project."""

from __future__ import annotations

import io
import logging
import urllib.request
from pathlib import Path
from typing import Dict, Optional

import joblib
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"
TEST_DATA_PATH = PROJECT_ROOT / "test_data.csv"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
METRICS_PATH = MODEL_DIR / "training_metrics.csv"

TRAIN_FILE = "KDDTrain+.txt"
# Public mirror of the Kaggle "hassan06/nslkdd" files (no login required).
DATASET_URL = (
    "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
)

# --------------------------------------------------------------------------- #
# Dataset schema
# --------------------------------------------------------------------------- #
FEATURE_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]
LABEL_COLUMN = "label"
DIFFICULTY_COLUMN = "difficulty"
RAW_COLUMNS = FEATURE_COLUMNS + [LABEL_COLUMN, DIFFICULTY_COLUMN]
CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]
TARGET_COLUMN = "target"  # 0 = Normal, 1 = Attack
CLASS_NAMES = ["Normal", "Attack"]

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Display name -> file name inside model/
MODEL_FILES: Dict[str, str] = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "SVM": "svm.pkl",
}


def get_logger(name: str) -> logging.Logger:
    """Return a consistently configured logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)


LOGGER = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def download_dataset(force: bool = False) -> Path:
    """Download KDDTrain+.txt into ``data/`` if it is not already present."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = DATA_DIR / TRAIN_FILE
    if target.exists() and not force:
        return target
    LOGGER.info("Downloading NSL-KDD training file from %s", DATASET_URL)
    try:
        urllib.request.urlretrieve(DATASET_URL, target)  # noqa: S310
    except OSError as exc:
        raise RuntimeError(
            f"Could not download the dataset. Download '{TRAIN_FILE}' from "
            "https://www.kaggle.com/datasets/hassan06/nslkdd and place it in "
            f"'{DATA_DIR}'."
        ) from exc
    return target


def load_raw_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the raw NSL-KDD training file (43 columns, no header)."""
    path = Path(path) if path else download_dataset()
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return pd.read_csv(path, header=None, names=RAW_COLUMNS)


def read_uploaded_csv(file_obj) -> pd.DataFrame:
    """Read a user supplied CSV; fall back to the header-less NSL-KDD layout."""
    raw = file_obj.read() if hasattr(file_obj, "read") else file_obj
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = [str(c).strip() for c in df.columns]

    if set(FEATURE_COLUMNS).issubset(df.columns):
        return df

    # Original NSL-KDD txt files have no header row.
    headerless = pd.read_csv(io.BytesIO(raw), header=None)
    if headerless.shape[1] in (41, 42, 43):
        headerless.columns = RAW_COLUMNS[: headerless.shape[1]]
        return headerless
    return df


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_artifact(obj, path: Path) -> None:
    """Persist an object with joblib (compressed to keep files small)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path, compress=3)


def load_artifact(path: Path):
    """Load a joblib artifact, raising a clear error if it is missing."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Missing artifact '{path}'. Run `python -m src.train_models` first."
        )
    return joblib.load(path)
