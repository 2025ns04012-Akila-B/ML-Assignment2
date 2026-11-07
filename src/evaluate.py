"""Model evaluation helpers shared by training code and the Streamlit app."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils import CLASS_NAMES

METRIC_COLUMNS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def predict_with_scores(model, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Return hard predictions and a continuous attack score for ROC/AUC."""
    y_pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:, 1]
    else:  # SVC without probability estimates
        y_score = model.decision_function(X)
    return np.asarray(y_pred), np.asarray(y_score)


def compute_metrics(y_true, y_pred, y_score) -> Dict[str, float]:
    """Compute the six assignment metrics (Attack is the positive class)."""
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:  # only one class present in y_true
        auc = float("nan")
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def get_confusion_matrix(y_true, y_pred) -> np.ndarray:
    """2x2 confusion matrix with fixed label order [Normal, Attack]."""
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def get_classification_report(y_true, y_pred) -> pd.DataFrame:
    """Classification report (precision, recall, F1, support) as a DataFrame."""
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    frame = pd.DataFrame(report).T
    frame["support"] = frame["support"].astype(int)
    return frame


def get_roc_curve(y_true, y_score) -> Tuple[np.ndarray, np.ndarray]:
    """Return false-positive and true-positive rates."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return fpr, tpr


def evaluate_models(models: Dict[str, object], X, y) -> pd.DataFrame:
    """Evaluate every model and return the comparison table."""
    rows = []
    for name, model in models.items():
        y_pred, y_score = predict_with_scores(model, X)
        rows.append({"ML Model": name, **compute_metrics(y, y_pred, y_score)})
    return pd.DataFrame(rows, columns=["ML Model"] + METRIC_COLUMNS)
