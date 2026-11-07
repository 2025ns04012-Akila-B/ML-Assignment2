"""Train all five classifiers, evaluate them and persist the artifacts.

Run from the project root::

    python -m src.train_models
"""

from __future__ import annotations

import time
from typing import Dict

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.evaluate import evaluate_models
from src.preprocessing import prepare_train_test
from src.utils import (
    METRICS_PATH,
    MODEL_DIR,
    MODEL_FILES,
    PREPROCESSOR_PATH,
    RANDOM_STATE,
    TEST_DATA_PATH,
    get_logger,
    load_raw_dataset,
    save_artifact,
)

LOGGER = get_logger(__name__)


def build_models() -> Dict[str, object]:
    """Return untrained instances of the five required classifiers."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Naive Bayes": GaussianNB(),
        # probability=False keeps training fast; AUC uses decision_function.
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE),
    }


def train_models(models: Dict[str, object], X_train, y_train) -> Dict[str, object]:
    """Fit every model, logging the training time."""
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(X_train, y_train)
        LOGGER.info("Trained %-20s in %.1fs", name, time.perf_counter() - start)
    return models


def save_models(models: Dict[str, object], preprocessor: Dict) -> None:
    """Save each model plus the fitted preprocessor under ``model/``."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        save_artifact(model, MODEL_DIR / MODEL_FILES[name])
        LOGGER.info("Saved %s -> model/%s", name, MODEL_FILES[name])
    save_artifact(preprocessor, PREPROCESSOR_PATH)


def export_test_data(raw_test_df: pd.DataFrame) -> None:
    """Write the hold-out split (raw, unscaled features + label) to CSV."""
    raw_test_df.to_csv(TEST_DATA_PATH, index=False)
    LOGGER.info("Exported %d rows to %s", len(raw_test_df), TEST_DATA_PATH.name)


def main() -> pd.DataFrame:
    """Run the complete training pipeline and return the metrics table."""
    raw = load_raw_dataset()
    LOGGER.info("Raw dataset shape: %s", raw.shape)

    X_train, X_test, y_train, y_test, preprocessor, raw_test = (
        prepare_train_test(raw)
    )
    LOGGER.info("Train: %s | Test: %s", X_train.shape, X_test.shape)

    models = train_models(build_models(), X_train, y_train)
    save_models(models, preprocessor)
    export_test_data(raw_test)

    results = evaluate_models(models, X_test, y_test)
    results.to_csv(METRICS_PATH, index=False)
    print(results.round(4).to_string(index=False))
    return results


if __name__ == "__main__":
    main()
