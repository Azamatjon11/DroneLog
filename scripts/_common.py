"""Shared protocol for the revised baselines: models, seeds and repeated grouped CV."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

REPEATS = 10  # repetitions of grouped cross-validation; repetition r uses random_state=r


def classifiers(impute: bool = False) -> dict:
    """The four baseline classifiers with default hyperparameters (no tuning)."""

    pre = [SimpleImputer()] if impute else []
    return {
        "Random Forest": lambda r: make_pipeline(*pre, RandomForestClassifier(random_state=r, n_jobs=2)),
        "SVM": lambda r: make_pipeline(*pre, StandardScaler(), SVC()),
        "KNN": lambda r: make_pipeline(*pre, StandardScaler(), KNeighborsClassifier()),
        "Logistic Regression": lambda r: make_pipeline(*pre, StandardScaler(), LogisticRegression(max_iter=3000)),
    }


def fmt(mean: float, sd: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def per_flight(files, y_true, y_pred):
    """Per-flight accuracy of window predictions (one row per log file)."""

    import pandas as pd

    df = pd.DataFrame(dict(file=files, label=y_true, correct=(pd.Series(y_pred) == pd.Series(y_true)).values))
    return df.groupby("file", sort=False).agg(label=("label", "first"), windows=("correct", "size"),
                                              accuracy=("correct", "mean")).reset_index()
