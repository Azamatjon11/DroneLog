#!/usr/bin/env python3
"""Task 5: online payload pickup/drop event detection.

INTUITION: payload pickup or drop changes mass suddenly, producing a transient
in acceleration/gyro magnitude and often a step in motor output. A simple
rolling statistic plus thresholded derivative is a transparent baseline.
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
from scipy.signal import find_peaks

from dronelog.events import get_events, load_event_table
from dronelog.io import (
    axis_fields,
    find_fields,
    find_flights,
    get_flight_t0,
    get_flight_time_bounds,
    list_topics,
    load_topic,
    print_startup,
)
from dronelog.metrics import event_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 5: payload pickup/drop event detection.")
    parser.add_argument("--data", default=".", type=Path, help="Dataset root. Default: current directory")
    parser.add_argument("--out", default=Path("outputs"), type=Path, help="Output directory. Default: outputs/")
    parser.add_argument("--events-csv", type=Path, help="Optional event annotation override CSV")
    parser.add_argument("--tolerance", default=3.0, type=float, help="True/detected event matching tolerance in seconds")
    parser.add_argument("--rolling", default=1.0, type=float, help="Rolling statistic window in seconds")
    parser.add_argument("--threshold-z", default=3.0, type=float, help="Derivative z-score threshold")
    parser.add_argument("--cruise-buffer", default=5.0, type=float, help="Seconds to trim after takeoff and before landing")
    parser.add_argument("--step-delay-correction", default=5.0, type=float, help="Seconds added to thrust-step edge detections")
    parser.add_argument("--max-flights", type=int, help="Optional cap for quick smoke runs")
    parser.add_argument("--inspect", action="store_true", help="Print topics and matched fields for one flight, then exit")
    parser.add_argument("--no-cache", action="store_true", help="Disable parquet cache reads/writes")
    return parser.parse_args()


def load_motor_topic(path: Path, use_cache: bool) -> tuple[str, pd.DataFrame | None]:
    for topic in ["actuator_outputs", "actuator_motors"]:
        try:
            return topic, load_topic(path, topic, use_cache=use_cache)
        except Exception:
            continue
    return "", None


def motor_columns(df: pd.DataFrame | None, topic: str) -> list[str]:
    if df is None:
        return []
    pattern = r"control\[\d+\]" if topic == "actuator_motors" else r"output\[\d+\]"
    candidates = find_fields(df, pattern)
    usable: list[str] = []
    for column in candidates:
        values = df[column].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size and (np.nanstd(finite) > 1e-6 or np.nanmax(np.abs(finite)) > 0.1):
            usable.append(column)
    return usable


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    std = np.nanstd(values)
    if not np.isfinite(std) or std <= 1e-12:
        return np.zeros_like(values)
    return np.nan_to_num((values - np.nanmean(values)) / std)


def detection_statistic(
    imu: pd.DataFrame,
    accel_cols: list[str],
    gyro_cols: list[str],
    motor: pd.DataFrame | None,
    motor_cols: list[str],
    rolling_seconds: float,
    t0: float,
    cruise_start: float,
    cruise_end: float,
) -> pd.DataFrame:
    if motor is None or not motor_cols:
        raise RuntimeError("Task 5 requires actuator_outputs/actuator_motors fields for thrust-step detection")

    motor_work = motor[(motor["timestamp"] >= cruise_start) & (motor["timestamp"] <= cruise_end)].copy()
    if len(motor_work) < 10:
        raise RuntimeError("Not enough motor samples inside cruise window")

    t = motor_work["timestamp"].to_numpy(dtype=float)
    rel_t = t - t0
    thrust = motor_work[motor_cols].mean(axis=1).to_numpy(dtype=float)
    dt = np.nanmedian(np.diff(t)) if len(t) > 1 else 0.05
    window = max(3, int(round(rolling_seconds / max(dt, 1e-3))))

    thrust_smooth = pd.Series(thrust).rolling(window, center=True, min_periods=1).median().to_numpy()
    stat = zscore(thrust_smooth)

    imu_work = imu[(imu["timestamp"] >= cruise_start) & (imu["timestamp"] <= cruise_end)].copy()
    if len(imu_work) > 10:
        accel_mag = np.linalg.norm(imu_work[accel_cols].to_numpy(dtype=float), axis=1)
        accel_dt = np.nanmedian(np.diff(imu_work["timestamp"].to_numpy(dtype=float)))
        accel_window = max(3, int(round(rolling_seconds / max(accel_dt, 1e-3))))
        accel_rms = pd.Series(accel_mag).rolling(accel_window, center=True, min_periods=1).std().to_numpy()
        accel_interp = np.interp(t, imu_work["timestamp"].to_numpy(dtype=float), zscore(accel_rms))
        stat += 0.2 * accel_interp

    stat = pd.Series(stat).rolling(window, center=True, min_periods=1).mean().to_numpy()
    derivative = np.abs(np.gradient(stat, rel_t, edge_order=1))
    return pd.DataFrame({"t_flight": rel_t, "statistic": stat, "derivative": derivative})


def threshold_detector(df: pd.DataFrame, tolerance: float, threshold_z: float, max_events: int) -> list[float]:
    score = zscore(df["derivative"].to_numpy(dtype=float))
    distance = max(1, int(round(max(8.0, tolerance) / np.nanmedian(np.diff(df["t_flight"])))))
    peaks, properties = find_peaks(score, height=threshold_z, distance=distance)
    if len(peaks) == 0:
        peaks, properties = find_peaks(score, distance=distance)
    if len(peaks) == 0:
        return []
    order = np.argsort(score[peaks])[::-1]
    selected = peaks[order[:max_events]]
    return sorted(df["t_flight"].to_numpy(dtype=float)[selected].astype(float).tolist())


def ruptures_detector(df: pd.DataFrame, tolerance: float, max_events: int, max_points: int = 1500) -> list[float]:
    try:
        import ruptures as rpt
    except Exception:
        return []
    stride = max(1, int(np.ceil(len(df) / max_points)))
    reduced = df.iloc[::stride].reset_index(drop=True)
    signal = reduced[["statistic"]].to_numpy(dtype=float)
    if len(signal) < 20:
        return []
    result = rpt.Pelt(model="l2", min_size=5).fit(signal).predict(pen=3.0 * np.log(len(signal)))
    times = reduced["t_flight"].to_numpy(dtype=float)
    detected = [float(times[min(idx, len(times) - 1)]) for idx in result[:-1]]
    filtered: list[float] = []
    for event_time in detected:
        if not filtered or event_time - filtered[-1] >= tolerance:
            filtered.append(event_time)
    if len(filtered) <= max_events:
        return filtered
    score = np.interp(filtered, df["t_flight"].to_numpy(dtype=float), zscore(df["derivative"].to_numpy(dtype=float)))
    order = np.argsort(score)[::-1]
    return sorted([filtered[idx] for idx in order[:max_events]])


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
    if motor is not None:
        print(f"Matched motor topic: {motor_topic}")
        print(f"Matched motor fields: {motor_columns(motor, motor_topic)}")


def true_event_times(events: dict[str, float | None]) -> list[float]:
    return [float(value) for value in [events.get("pickup"), events.get("drop")] if value is not None]


def apply_delay_correction(detected: list[float], delay_s: float, stat_df: pd.DataFrame) -> list[float]:
    if not detected:
        return []
    lower = float(stat_df["t_flight"].min())
    upper = float(stat_df["t_flight"].max())
    return [min(max(time + delay_s, lower), upper) for time in detected]


def save_figure(stat_df: pd.DataFrame, true_events: list[float], detected: list[float], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(stat_df["t_flight"], stat_df["statistic"], label="Cruise thrust statistic", linewidth=1.2)
    for idx, event_time in enumerate(true_events):
        ax.axvline(event_time, color="tab:green", linestyle="--", linewidth=1.5, label="True event" if idx == 0 else None)
    for idx, event_time in enumerate(detected):
        ax.axvline(event_time, color="tab:red", linestyle=":", linewidth=1.2, label="Detected event" if idx == 0 else None)
    ax.set_xlabel("Seconds after takeoff (t_flight)")
    ax.set_ylabel("Statistic")
    ax.set_title("Task 5 event detection")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def maybe_save_diagnostic_plot(record, stat_df: pd.DataFrame, true_events: list[float], detected: list[float], output: Path) -> None:
    if record.platform == "x500v2" and abs(float(record.mass_kg) - 0.5) < 1e-9 and int(record.scenario) == 4:
        diagnostic = output.parent / "task5_diagnostic_x500v2_0.5kg_s4.png"
        save_figure(stat_df, true_events, detected, diagnostic)
        print(f"  TASK5 diagnostic plot saved {diagnostic}")
        for event_time in true_events:
            near = stat_df.iloc[(stat_df["t_flight"] - event_time).abs().argsort()[:5]]
            print(
                f"  TASK5 diagnostic near true event {event_time:.1f}s: "
                f"stat range {near['statistic'].min():.3f}..{near['statistic'].max():.3f}, "
                f"derivative max {near['derivative'].max():.3f}"
            )


def main() -> None:
    warnings.filterwarnings("ignore")
    args = parse_args()
    print_startup("Task 5 - Payload event detection")
    event_table = load_event_table(args.events_csv)
    flights = [
        record
        for record in find_flights(args.data, "payload")
        if record.scenario in {3, 4, 5}
    ]
    if args.max_flights:
        flights = flights[: args.max_flights]
    if not flights:
        raise SystemExit(f"No scenario 3/4/5 payload .ulg files found below {args.data}")
    if args.inspect:
        inspect_sample(args, flights)
        return
    args.out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str | int]] = []
    skips: list[str] = []
    figure_payload: tuple[pd.DataFrame, list[float], list[float]] | None = None
    for index, record in enumerate(flights, start=1):
        print(f"[{index}/{len(flights)}] {record.flight_id}")
        try:
            imu = load_topic(record.ulg_path, "sensor_combined", use_cache=not args.no_cache)
            accel_cols = axis_fields(imu, ["accel", "accelerometer"])
            gyro_cols = axis_fields(imu, ["gyro"])
            motor_topic, motor = load_motor_topic(record.ulg_path, use_cache=not args.no_cache)
            motor_cols = motor_columns(motor, motor_topic)
            events = get_events(record.platform, record.mass_kg, record.scenario, event_table)
            true_events = true_event_times(events)
            t0 = get_flight_t0(record.ulg_path, use_cache=not args.no_cache)
            cruise_start, cruise_end = get_flight_time_bounds(
                record.ulg_path,
                use_cache=not args.no_cache,
                trim_start_s=args.cruise_buffer,
                trim_end_s=args.cruise_buffer,
            )
            stat_df = detection_statistic(
                imu,
                accel_cols,
                gyro_cols,
                motor,
                motor_cols,
                args.rolling,
                t0,
                cruise_start,
                cruise_end,
            )
            max_events = max(1, len(true_events))
            threshold_events = apply_delay_correction(
                threshold_detector(stat_df, args.tolerance, args.threshold_z, max_events),
                args.step_delay_correction,
                stat_df,
            )
            ruptures_events = apply_delay_correction(
                ruptures_detector(stat_df, args.tolerance, max_events),
                args.step_delay_correction,
                stat_df,
            )
        except Exception as exc:
            skips.append(f"{record.ulg_path}: {exc}")
            print(f"  skip: {exc}")
            continue

        for method, detected in [
            ("thresholded_derivative", threshold_events),
            ("ruptures_pelt" if ruptures_events else "ruptures_pelt_unavailable", ruptures_events),
        ]:
            scores = event_scores(true_events, detected, args.tolerance)
            rows.append(
                {
                    "flight_id": record.flight_id,
                    "platform": record.platform,
                    "mass_kg": record.mass_kg,
                    "scenario": record.scenario,
                    "method": method,
                    "true_events": len(true_events),
                    "detected_events": len(detected),
                    **scores,
                }
            )
        if figure_payload is None and true_events:
            figure_payload = (stat_df, true_events, threshold_events)
        maybe_save_diagnostic_plot(record, stat_df, true_events, threshold_events, args.out / "task5_events.png")
        print(f"  true={len(true_events)} detected={len(threshold_events)}")

    if not rows:
        raise SystemExit("No flights could be scored.")
    results = pd.DataFrame(rows)
    table_path = args.out / "task5_event_results.csv"
    figure_path = args.out / "task5_events.png"
    results.to_csv(table_path, index=False)
    if figure_payload is not None:
        save_figure(*figure_payload, output=figure_path)

    summary = (
        results.groupby("method")[["precision", "recall", "mean_abs_latency_s"]]
        .mean(numeric_only=True)
        .reset_index()
    )
    print("\nMean summary by method")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved {table_path}")
    if figure_payload is not None:
        print(f"Saved {figure_path}")
    if skips:
        print("\nSkipped flights:")
        for item in skips:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
