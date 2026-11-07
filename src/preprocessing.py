"""Data cleaning, encoding, scaling and splitting for NSL-KDD."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils import (
    CATEGORICAL_COLUMNS,
    DIFFICULTY_COLUMN,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    get_logger,
)

LOGGER = get_logger(__name__)

_LABEL_ALIASES = ("label", "class", "target", "attack", "attack_type")


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def make_binary_target(labels: pd.Series) -> pd.Series:
    """Map labels to 0 (Normal) / 1 (Attack).

    Accepts raw NSL-KDD attack names, 'normal'/'attack' strings or 0/1 values.
    """
    numeric = pd.to_numeric(labels, errors="coerce")
    if numeric.notna().all():
        return (numeric != 0).astype(int)
    cleaned = labels.astype(str).str.strip().str.lower().str.rstrip(".")
    return (cleaned != "normal").astype(int)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the difficulty column, rows without label, duplicates and infs.

    Remaining missing feature values are imputed later using statistics
    learned from the training split only (see ``fit_preprocessor``).
    """
    df = df.copy()
    df = df.drop(columns=[DIFFICULTY_COLUMN], errors="ignore")
    df = df.replace([np.inf, -np.inf], np.nan)

    LOGGER.info("Missing values before cleaning: %d", int(df.isna().sum().sum()))
    before = len(df)
    df = df.dropna(subset=[LABEL_COLUMN])
    df = df.drop_duplicates().reset_index(drop=True)
    LOGGER.info("Removed %d duplicate/unlabeled rows", before - len(df))
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split on the binary target (raw features kept)."""
    target = make_binary_target(df[LABEL_COLUMN])
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Fit / transform
# --------------------------------------------------------------------------- #
def fit_preprocessor(train_df: pd.DataFrame) -> Dict:
    """Learn imputation values, label encoders and the scaler from train data."""
    features = _coerce_features(train_df[FEATURE_COLUMNS])

    fill_values = {}
    for col in FEATURE_COLUMNS:
        if col in CATEGORICAL_COLUMNS:
            fill_values[col] = features[col].mode(dropna=True).iloc[0]
        else:
            fill_values[col] = features[col].median()
    features = features.fillna(fill_values)

    encoders: Dict[str, LabelEncoder] = {}
    for col in CATEGORICAL_COLUMNS:
        encoder = LabelEncoder()
        features[col] = encoder.fit_transform(features[col].astype(str))
        encoders[col] = encoder

    scaler = StandardScaler().fit(features[FEATURE_COLUMNS])
    return {
        "feature_columns": list(FEATURE_COLUMNS),
        "fill_values": fill_values,
        "encoders": encoders,
        "scaler": scaler,
    }


def transform_features(df: pd.DataFrame, preprocessor: Dict) -> pd.DataFrame:
    """Apply imputation, label encoding and scaling using a fitted preprocessor.

    Categories unseen during training are encoded as -1.
    """
    columns = preprocessor["feature_columns"]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            "Uploaded data is missing required columns: " + ", ".join(missing)
        )

    features = _coerce_features(df[columns])
    features = features.fillna(preprocessor["fill_values"])

    for col, encoder in preprocessor["encoders"].items():
        mapping = {cls: idx for idx, cls in enumerate(encoder.classes_)}
        features[col] = features[col].astype(str).map(mapping).fillna(-1)

    scaled = preprocessor["scaler"].transform(features[columns].astype(float))
    return pd.DataFrame(scaled, columns=columns, index=df.index)


def _coerce_features(features: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric columns to numbers; keep categorical columns as text."""
    features = features.copy()
    for col in features.columns:
        if col in CATEGORICAL_COLUMNS:
            features[col] = features[col].astype("object")
        else:
            features[col] = pd.to_numeric(features[col], errors="coerce")
    return features


def extract_target(df: pd.DataFrame) -> Optional[pd.Series]:
    """Return the binary target if a label column exists, otherwise None."""
    lowered = {str(c).lower(): c for c in df.columns}
    for alias in _LABEL_ALIASES:
        if alias in lowered:
            column = df[lowered[alias]]
            if column.isna().any():
                return None
            return make_binary_target(column)
    return None


def prepare_train_test(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict, pd.DataFrame]:
    """Run the full pipeline: clean -> split -> fit/transform.

    Returns ``X_train, X_test, y_train, y_test, preprocessor, raw_test_df``.
    """
    cleaned = clean_dataset(df)
    train_df, test_df = split_data(cleaned)
    preprocessor = fit_preprocessor(train_df)
    X_train = transform_features(train_df, preprocessor)
    X_test = transform_features(test_df, preprocessor)
    y_train = make_binary_target(train_df[LABEL_COLUMN]).rename(TARGET_COLUMN)
    y_test = make_binary_target(test_df[LABEL_COLUMN]).rename(TARGET_COLUMN)
    return X_train, X_test, y_train, y_test, preprocessor, test_df
