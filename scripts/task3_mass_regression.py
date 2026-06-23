#!/usr/bin/env python3
"""Task 3: continuous payload-mass estimation.

INTUITION: heavier payload changes hover thrust and dynamic response. Those
changes are visible in motor commands, acceleration variance, and attitude
response, so windowed telemetry can be used as a regression signal.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MPLCONFIGDIR = (ROOT / ".cache" / "matplotlib").resolve()
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
(ROOT / ".cache" / "fontconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str((ROOT / ".cache").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dronelog.events import carried_mass_for_window, get_events, load_event_table
from dronelog.io import axis_fields, find_fields, find_flights, get_flight_t0, list_topics, load_topic, print_startup, to_flight_time
from dronelog.metrics import regression_metrics, signal_features, window_feature_rows


SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 3: continuous payload-mass regression.")
    parser.add_argument("--data", default=".", type=Path, help="Dataset root. Default: current directory")
    parser.add_argument("--out", default=Path("outputs"), type=Path, help="Output directory. Default: outputs/")
    parser.add_argument("--window", default=5.0, type=float, help="Non-overlapping window length in seconds")
    parser.add_argument("--folds", default=5, type=int, help="Grouped CV folds")
    parser.add_argument("--max-flights", type=int, help="Optional cap for quick smoke runs")
    parser.add_argument("--events-csv", type=Path, help="Optional event annotation override CSV")
    parser.add_argument("--inspect", action="store_true", help="Print topics and matched fields for one flight, then exit")
    parser.add_argument("--no-cache", action="store_true", help="Disable parquet cache reads/writes")
    return parser.parse_args()


def models() -> dict[str, object]:
    return {
        "RandomForestRegressor": RandomForestRegressor(n_estimators=250, random_state=SEED, n_jobs=-1),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    }


def load_motor_topic(path: Path, use_cache: bool) -> tuple[str, pd.DataFrame]:
    errors: list[str] = []
    for topic in ["actuator_motors", "actuator_outputs"]:
        try:
            return topic, load_topic(path, topic, use_cache=use_cache)
        except Exception as exc:
            errors.append(str(exc))
    raise RuntimeError("; ".join(errors))


def motor_columns(df: pd.DataFrame, topic: str) -> list[str]:
    pattern = r"control\[\d+\]" if topic == "actuator_motors" else r"output\[\d+\]"
    candidates = find_fields(df, pattern)
    usable: list[str] = []
    for column in candidates:
        values = df[column].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size and (np.nanstd(finite) > 1e-6 or np.nanmax(np.abs(finite)) > 0.1):
            usable.append(column)
    if not usable:
        raise RuntimeError(f"No usable motor command fields in {topic}; candidates were {candidates}")
    return usable


def motor_window_features(df: pd.DataFrame, cols: list[str], start: float, end: float) -> list[float]:
    window = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]
    features: list[float] = []
    for col in cols:
        values = window[col].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            features.extend([0.0, 0.0])
        else:
            features.extend([float(np.mean(values)), float(np.std(values))])
    return features


def inspect_sample(args: argparse.Namespace, flights) -> None:
    record = flights[0]
    print(f"Sample flight: {record.ulg_path}")
    print("Topics:")
    for topic in list_topics(record.ulg_path):
        print(f"  - {topic}")
    imu = load_topic(record.ulg_path, "sensor_combined", use_cache=not args.no_cache)
    motor_topic, motor = load_motor_topic(record.ulg_path, use_cache=not args.no_cache)
    print(f"Matched accel fields: {axis_fields(imu, ['accel', 'accelerometer'])}")
    print(f"Matched gyro fields: {axis_fields(imu, ['gyro'])}")
    print(f"Matched motor topic: {motor_topic}")
    print(f"Matched motor fields: {motor_columns(motor, motor_topic)}")


def print_timebase_diagnostic(args: argparse.Namespace) -> None:
    path = args.data / "payload_detection" / "x500v2" / "0.5kg" / "4_scenario.ulg"
    if not path.exists():
        return
    imu = load_topic(path, "sensor_combined", use_cache=not args.no_cache)
    local = load_topic(path, "vehicle_local_position", use_cache=not args.no_cache)
    t0 = get_flight_t0(path, use_cache=not args.no_cache)
    start = float(imu["timestamp"].iloc[0])
    end = float(imu["timestamp"].iloc[-1])
    print("\nSTEP0 AFTER central timebase diagnostic")
    print(f"  flight: {path}")
    print(f"  log start timestamp: {start:.3f}s")
    print(f"  detected takeoff t0: {t0:.3f}s")
    print(f"  total log duration: {end - start:.3f}s")
    print(f"  flight duration after takeoff: {end - t0:.3f}s")
    for label, seconds in [("pickup", 103.0), ("drop", 179.0)]:
        raw = t0 + seconds
        nearest = (local["timestamp"] - raw).abs().idxmin()
        print(
            f"  Table-4 {label} {seconds:.0f}s -> raw timestamp {raw:.3f}s, "
            f"t_flight={raw - t0:.3f}s, nearest local z={local.loc[nearest, 'z']:.3f}m"
        )


def collect_dataset(args: argparse.Namespace, flights) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    event_table = load_event_table(args.events_csv)
    X: list[list[float]] = []
    y: list[float] = []
    groups: list[str] = []
    platforms: list[str] = []
    skips: list[str] = []
    printed_motor_fields = False
    printed_label_counts = False

    for index, record in enumerate(flights, start=1):
        print(f"[{index}/{len(flights)}] {record.flight_id}")
        try:
            imu = load_topic(record.ulg_path, "sensor_combined", use_cache=not args.no_cache)
            accel_cols = axis_fields(imu, ["accel", "accelerometer"])
            gyro_cols = axis_fields(imu, ["gyro"])
            motor_topic, motor = load_motor_topic(record.ulg_path, use_cache=not args.no_cache)
            motor_cols = motor_columns(motor, motor_topic)
            event_info = get_events(record.platform, record.mass_kg, record.scenario, event_table)
            t0 = get_flight_t0(record.ulg_path, use_cache=not args.no_cache)
            windows = window_feature_rows(imu, accel_cols + gyro_cols, args.window)
        except Exception as exc:
            skips.append(f"{record.ulg_path}: {exc}")
            print(f"  skip: {exc}")
            continue

        if not printed_motor_fields:
            print(f"  TASK3 motor diagnostic: topic={motor_topic}, fields={motor_cols}")
            printed_motor_fields = True

        label_counts: dict[float, int] = {}
        for start, end, imu_features in windows:
            label = carried_mass_for_window(
                float(record.mass_kg),
                int(record.scenario),
                start - t0,
                end - t0,
                event_info,
            )
            X.append(imu_features + motor_window_features(motor, motor_cols, start, end))
            y.append(label)
            groups.append(record.flight_id)
            platforms.append(record.platform)
            label_counts[label] = label_counts.get(label, 0) + 1
        if (
            not printed_label_counts
            and record.platform == "x500v2"
            and abs(float(record.mass_kg) - 0.5) < 1e-9
            and int(record.scenario) == 4
        ):
            print(
                "  TASK3 AFTER label diagnostic x500v2/0.5kg scenario4 "
                f"takeoff-relative window labels={label_counts}"
            )
            printed_label_counts = True
        print(f"  windows: {len(windows)}")

    return np.asarray(X, dtype=float), np.asarray(y, dtype=float), np.asarray(groups), np.asarray(platforms), skips


def evaluate(X: np.ndarray, y: np.ndarray, groups: np.ndarray, requested_folds: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = max(2, min(requested_folds, len(set(groups))))
    splitter = GroupKFold(n_splits=folds)
    rows: list[dict[str, float | str | int]] = []
    pred_rows: list[dict[str, float | str]] = []

    for name, model in models().items():
        y_true_all: list[float] = []
        y_pred_all: list[float] = []
        for train_idx, test_idx in splitter.split(X, y, groups):
            estimator = clone(model)
            estimator.fit(X[train_idx], y[train_idx])
            pred = estimator.predict(X[test_idx])
            y_true_all.extend(y[test_idx])
            y_pred_all.extend(pred)
            for truth, guess in zip(y[test_idx], pred):
                pred_rows.append({"model": name, "true_mass_kg": float(truth), "pred_mass_kg": float(guess)})
        rows.append(
            {
                "model": name,
                **regression_metrics(y_true_all, y_pred_all),
                "windows": len(X),
                "groups": len(set(groups)),
                "folds": folds,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


def save_scatter(predictions: pd.DataFrame, output: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, group in predictions.groupby("model"):
        ax.scatter(group["true_mass_kg"], group["pred_mass_kg"], s=18, alpha=0.55, label=name)
    lower = min(predictions["true_mass_kg"].min(), predictions["pred_mass_kg"].min())
    upper = max(predictions["true_mass_kg"].max(), predictions["pred_mass_kg"].max())
    ax.plot([lower, upper], [lower, upper], color="black", linewidth=1, linestyle="--", label="y=x")
    ax.set_xlabel("True carried mass (kg)")
    ax.set_ylabel("Predicted carried mass (kg)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    warnings.filterwarnings("ignore")
    args = parse_args()
    np.random.seed(SEED)
    print_startup("Task 3 - Payload-mass regression")
    flights = find_flights(args.data, "payload")
    if args.max_flights:
        flights = flights[: args.max_flights]
    if not flights:
        raise SystemExit(f"No payload .ulg files found below {args.data}")
    if args.inspect:
        inspect_sample(args, flights)
        return

    print_timebase_diagnostic(args)
    X, y, groups, platforms, skips = collect_dataset(args, flights)
    if len(X) == 0:
        raise SystemExit("No usable windows were extracted.")
    args.out.mkdir(parents=True, exist_ok=True)

    result_sets: list[tuple[str, pd.DataFrame]] = []
    results, predictions = evaluate(X, y, groups, args.folds)
    results.insert(0, "platform", "combined")
    table_path = args.out / "task3_mass_results.csv"
    figure_path = args.out / "task3_mass_scatter.png"
    results.to_csv(table_path, index=False)
    save_scatter(predictions, figure_path, "Task 3 payload-mass regression - combined")
    result_sets.append(("combined", results))
    print(f"Saved {table_path}")
    print(f"Saved {figure_path}")

    for platform in ["x500v2", "x650"]:
        mask = platforms == platform
        if np.count_nonzero(mask) < 2 or len(set(groups[mask])) < 2:
            print(f"Skipping per-platform regression for {platform}: not enough groups")
            continue
        platform_results, platform_predictions = evaluate(X[mask], y[mask], groups[mask], args.folds)
        platform_results.insert(0, "platform", platform)
        safe = platform.replace("/", "_")
        platform_table = args.out / f"task3_mass_results_{safe}.csv"
        platform_figure = args.out / f"task3_mass_scatter_{safe}.png"
        platform_results.to_csv(platform_table, index=False)
        save_scatter(platform_predictions, platform_figure, f"Task 3 payload-mass regression - {platform}")
        result_sets.append((platform, platform_results))
        print(f"Saved {platform_table}")
        print(f"Saved {platform_figure}")

    all_results = pd.concat([item[1] for item in result_sets], ignore_index=True)
    display = all_results.copy()
    for column in ["mae_kg", "rmse_kg", "r2"]:
        display[column] = display[column].map(lambda value: f"{value:.4f}")
    print("\nFinal summary")
    print(display.to_string(index=False))
    if skips:
        print("\nSkipped flights:")
        for item in skips:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
