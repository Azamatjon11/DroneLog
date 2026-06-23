#!/usr/bin/env python3
"""Task 2: payload-scenario classification from ULog-native IMU features."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from dronelog.io import axis_fields, find_flights, list_topics, load_topic, print_startup
from dronelog.metrics import classification_metrics, window_feature_rows


SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 2: payload-scenario classification.")
    parser.add_argument("--data", default=".", type=Path, help="Dataset root. Default: current directory")
    parser.add_argument("--out", default=Path("outputs"), type=Path, help="Output directory. Default: outputs/")
    parser.add_argument("--window", default=5.0, type=float, help="Non-overlapping window length in seconds")
    parser.add_argument("--folds", default=5, type=int, help="Requested StratifiedGroupKFold splits")
    parser.add_argument("--inspect", action="store_true", help="Print topics and matched fields for one flight, then exit")
    parser.add_argument("--no-cache", action="store_true", help="Disable parquet cache reads/writes")
    return parser.parse_args()


def models() -> dict[str, object]:
    return {
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale")),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "LogisticRegression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=5000, random_state=SEED)
        ),
    }


def inspect_sample(args: argparse.Namespace, flights) -> None:
    record = flights[0]
    print(f"Sample flight: {record.ulg_path}")
    print("Topics:")
    for topic in list_topics(record.ulg_path):
        print(f"  - {topic}")
    imu = load_topic(record.ulg_path, "sensor_combined", use_cache=not args.no_cache)
    accel_cols = axis_fields(imu, ["accel", "accelerometer"])
    gyro_cols = axis_fields(imu, ["gyro"])
    print(f"Matched accel fields: {accel_cols}")
    print(f"Matched gyro fields: {gyro_cols}")


def collect_dataset(args: argparse.Namespace, flights) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    rows: list[list[float]] = []
    labels: list[int] = []
    groups: list[str] = []
    skips: list[str] = []

    for index, record in enumerate(flights, start=1):
        print(f"[{index}/{len(flights)}] {record.flight_id}")
        try:
            imu = load_topic(record.ulg_path, "sensor_combined", use_cache=not args.no_cache)
            accel_cols = axis_fields(imu, ["accel", "accelerometer"])
            gyro_cols = axis_fields(imu, ["gyro"])
            windows = window_feature_rows(imu, accel_cols + gyro_cols, args.window)
        except Exception as exc:
            skips.append(f"{record.ulg_path}: {exc}")
            print(f"  skip: {exc}")
            continue

        for _start, _end, features in windows:
            rows.append(features)
            labels.append(int(record.scenario))
            groups.append(record.flight_id)
        print(f"  windows: {len(windows)}")

    return np.asarray(rows, dtype=float), np.asarray(labels), np.asarray(groups), skips


def n_splits_for(y: np.ndarray, groups: np.ndarray, requested: int) -> int:
    group_label = {}
    for label, group in zip(y, groups):
        group_label.setdefault(group, label)
    per_class_groups = Counter(group_label.values())
    return max(2, min(requested, len(set(groups)), min(per_class_groups.values())))


def evaluate(X: np.ndarray, y: np.ndarray, groups: np.ndarray, requested_folds: int) -> pd.DataFrame:
    folds = n_splits_for(y, groups, requested_folds)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)
    rows: list[dict[str, float | str | int]] = []

    for name, model in models().items():
        y_true: list[int] = []
        y_pred: list[int] = []
        for train_idx, test_idx in splitter.split(X, y, groups):
            estimator = clone(model)
            estimator.fit(X[train_idx], y[train_idx])
            y_true.extend(y[test_idx])
            y_pred.extend(estimator.predict(X[test_idx]))
        rows.append(
            {
                "model": name,
                **classification_metrics(y_true, y_pred),
                "windows": len(X),
                "groups": len(set(groups)),
                "folds": folds,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    warnings.filterwarnings("ignore")
    args = parse_args()
    np.random.seed(SEED)
    print_startup("Task 2 - Payload-scenario classification")
    flights = find_flights(args.data, "payload")
    if not flights:
        raise SystemExit(f"No payload .ulg files found below {args.data}")
    if args.inspect:
        inspect_sample(args, flights)
        return

    X, y, groups, skips = collect_dataset(args, flights)
    if len(X) == 0:
        raise SystemExit("No usable windows were extracted.")
    results = evaluate(X, y, groups, args.folds)
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / "task2_payload_results.csv"
    results.to_csv(output, index=False)

    display = results.copy()
    for column in ["accuracy", "macro_f1", "macro_precision", "macro_recall"]:
        display[column] = display[column].map(lambda value: f"{value:.4f}")
    print("\nFinal summary")
    print(display.to_string(index=False))
    print(f"\nSaved {output}")
    print("Expected sanity check: payload-scenario accuracy is intentionally hard, often around 0.3.")
    if skips:
        print("\nSkipped flights:")
        for item in skips:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
