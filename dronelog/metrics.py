"""Metric and feature helpers shared by benchmark scripts."""

from __future__ import annotations

from collections.abc import Iterable
import math

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


FEATURE_NAMES = ["mean", "std", "min", "max", "median", "rms", "skew", "kurtosis"]


def signal_features(values: np.ndarray) -> list[float]:
    """Eight robust statistical features for one signal window."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [0.0] * len(FEATURE_NAMES)
    raw = [
        np.mean(values),
        np.std(values),
        np.min(values),
        np.max(values),
        np.median(values),
        np.sqrt(np.mean(values * values)),
        skew(values),
        kurtosis(values),
    ]
    return np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0).astype(float).tolist()


def window_bounds(timestamps: np.ndarray, window_seconds: float) -> list[tuple[float, float]]:
    """Non-overlapping complete window bounds in seconds."""

    if timestamps.size == 0:
        return []
    start = float(timestamps[0])
    end = float(timestamps[-1])
    bounds: list[tuple[float, float]] = []
    cursor = start
    while cursor + window_seconds <= end:
        bounds.append((cursor, cursor + window_seconds))
        cursor += window_seconds
    return bounds


def window_feature_rows(
    df: pd.DataFrame,
    columns: Iterable[str],
    window_seconds: float,
) -> list[tuple[float, float, list[float]]]:
    """Build statistical feature rows for complete non-overlapping windows."""

    columns = list(columns)
    rows: list[tuple[float, float, list[float]]] = []
    timestamps = df["timestamp"].to_numpy(dtype=float)
    for start, end in window_bounds(timestamps, window_seconds):
        window = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]
        if len(window) < 2:
            continue
        features: list[float] = []
        for column in columns:
            features.extend(signal_features(window[column].to_numpy(dtype=float)))
        rows.append((start, end, features))
    return rows


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae_kg": mean_absolute_error(y_true, y_pred),
        "rmse_kg": math.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def ate_rmse(estimate: np.ndarray, truth: np.ndarray) -> float:
    errors = np.asarray(estimate) - np.asarray(truth)
    return float(np.sqrt(np.mean(np.sum(errors * errors, axis=1))))


def path_length(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def final_drift_percent(estimate: np.ndarray, truth: np.ndarray) -> float:
    length = path_length(truth)
    if length <= 1e-9:
        return 0.0
    drift = np.linalg.norm(np.asarray(estimate)[-1] - np.asarray(truth)[-1])
    return float(100.0 * drift / length)


def event_scores(true_events: list[float], detected_events: list[float], tolerance: float) -> dict[str, float]:
    """Precision/recall and mean latency for one event list with one-to-one matching."""

    used: set[int] = set()
    latencies: list[float] = []
    true_positive = 0
    for true_time in true_events:
        candidates = [
            (abs(det_time - true_time), idx, det_time)
            for idx, det_time in enumerate(detected_events)
            if idx not in used and abs(det_time - true_time) <= tolerance
        ]
        if not candidates:
            continue
        _, idx, det_time = min(candidates)
        used.add(idx)
        true_positive += 1
        latencies.append(det_time - true_time)
    false_positive = len(detected_events) - true_positive
    false_negative = len(true_events) - true_positive
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "mean_latency_s": float(np.mean(latencies)) if latencies else np.nan,
        "mean_abs_latency_s": float(np.mean(np.abs(latencies))) if latencies else np.nan,
        "true_positives": true_positive,
        "false_positives": false_positive,
        "false_negatives": false_negative,
    }
