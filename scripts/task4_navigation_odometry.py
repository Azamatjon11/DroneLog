#!/usr/bin/env python3
"""Task 4: IMU-only inertial dead-reckoning vs. EKF/GPS ground truth.

INTUITION: the IMU measures rotation rate plus specific force, not position.
This script integrates gyro to orientation, rotates body-frame acceleration into
PX4 NED, removes gravity, and integrates acceleration to velocity and position.
No GPS is used in the IMU-only estimate. EKF/GPS position is used only for
scoring, and large IMU-only drift is expected rather than a bug.
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

from dronelog.geometry import (
    common_time_grid,
    gps_to_local_ned,
    local_position_ned,
    resample_positions,
    strapdown_dead_reckon,
    umeyama_align,
    world_linear_acceleration,
)
from dronelog.io import axis_fields, find_flights, get_flight_t0, list_topics, load_topic, print_startup
from dronelog.metrics import ate_rmse, final_drift_percent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 4: inertial dead-reckoning navigation benchmark.")
    parser.add_argument("--data", default=".", type=Path, help="Dataset root. Default: current directory")
    parser.add_argument("--out", default=Path("outputs"), type=Path, help="Output directory. Default: outputs/")
    parser.add_argument("--subset", choices=["frame", "payload", "all"], default="all", help="Flight subset to score")
    parser.add_argument("--max-flights", type=int, help="Optional cap for quick runs")
    parser.add_argument("--inspect", action="store_true", help="Print topics and matched fields for one flight, then exit")
    parser.add_argument("--no-cache", action="store_true", help="Disable parquet cache reads/writes")
    return parser.parse_args()


def load_optional_topic(path: Path, topic: str, use_cache: bool) -> pd.DataFrame | None:
    try:
        return load_topic(path, topic, use_cache=use_cache)
    except Exception:
        return None


def load_gps(path: Path, use_cache: bool) -> tuple[str, pd.DataFrame | None]:
    for topic in ["vehicle_gps_position", "sensor_gps"]:
        df = load_optional_topic(path, topic, use_cache)
        if df is not None:
            return topic, gps_to_local_ned(df)
    return "", None


def inspect_sample(args: argparse.Namespace, flights) -> None:
    record = flights[0]
    print(f"Sample flight: {record.ulg_path}")
    print("Topics:")
    for topic in list_topics(record.ulg_path):
        print(f"  - {topic}")
    imu = load_topic(record.ulg_path, "sensor_combined", use_cache=not args.no_cache)
    attitude = load_topic(record.ulg_path, "vehicle_attitude", use_cache=not args.no_cache)
    print(f"Matched accel fields: {axis_fields(imu, ['accel', 'accelerometer'])}")
    print(f"Matched gyro fields: {axis_fields(imu, ['gyro'])}")
    quat_cols = ["q[0]", "q[1]", "q[2]", "q[3]"]
    missing = [col for col in quat_cols if col not in attitude.columns]
    if missing:
        raise RuntimeError(f"Missing attitude quaternion fields {missing}; available {list(attitude.columns)}")
    print(f"Matched quaternion fields: {quat_cols}")
    print("Ground truth: raw vehicle_gps_position projected to local NED. EKF local_position is scored as a method.")


def score_track(
    flight_id: str,
    method: str,
    estimate: pd.DataFrame,
    truth: pd.DataFrame,
    truth_source: str,
) -> tuple[dict[str, float | str], pd.DataFrame]:
    times = common_time_grid(estimate, truth)
    if len(times) < 2:
        raise RuntimeError("No overlapping timestamps for scoring")
    est = resample_positions(estimate, times)
    gt = resample_positions(truth, times)
    aligned = umeyama_align(est, gt, with_scale=False)
    row = {
        "flight_id": flight_id,
        "method": method,
        "truth_source": truth_source,
        "ate_rmse_m": ate_rmse(aligned, gt),
        "final_drift_percent": final_drift_percent(aligned, gt),
        "samples": len(times),
    }
    aligned_df = pd.DataFrame({"timestamp": times, "x": aligned[:, 0], "y": aligned[:, 1], "z": aligned[:, 2]})
    return row, aligned_df


def process_flight(record, use_cache: bool) -> tuple[list[dict[str, float | str]], dict[str, pd.DataFrame]]:
    imu = load_topic(record.ulg_path, "sensor_combined", use_cache=use_cache)
    attitude = load_topic(record.ulg_path, "vehicle_attitude", use_cache=use_cache)
    accel_cols = axis_fields(imu, ["accel", "accelerometer"])
    gyro_cols = axis_fields(imu, ["gyro"])
    quat_cols = ["q[0]", "q[1]", "q[2]", "q[3]"]
    missing_quat = [col for col in quat_cols if col not in attitude.columns]
    if missing_quat:
        raise RuntimeError(f"Missing attitude quaternion fields {missing_quat}; available {list(attitude.columns)}")

    t0 = get_flight_t0(record.ulg_path, use_cache=use_cache)
    bias_end = min(t0 - 0.1, float(imu["timestamp"].iloc[0]) + 5.0)
    bias_window = (float(imu["timestamp"].iloc[0]), bias_end) if bias_end > float(imu["timestamp"].iloc[0]) else None
    imu_only = strapdown_dead_reckon(imu, attitude, accel_cols, gyro_cols, quat_cols, bias_window=bias_window)
    local_df = load_optional_topic(record.ulg_path, "vehicle_local_position", use_cache)
    local = local_position_ned(local_df) if local_df is not None else None
    gps_topic, gps = load_gps(record.ulg_path, use_cache)

    if gps is None or gps.empty:
        raise RuntimeError("Missing usable vehicle_gps_position/sensor_gps truth after GPS quality filtering")
    truth = gps
    truth_source = gps_topic

    rows: list[dict[str, float | str]] = []
    tracks: dict[str, pd.DataFrame] = {"truth": truth, "gps": gps}
    imu_row, imu_aligned = score_track(record.flight_id, "IMU-only strapdown", imu_only, truth, truth_source)
    rows.append(imu_row)
    tracks["imu_aligned"] = imu_aligned

    if local is not None and not local.empty:
        ekf_row, ekf_aligned = score_track(record.flight_id, "PX4 EKF local_position", local, truth, truth_source)
        rows.append(ekf_row)
        tracks["ekf_aligned"] = ekf_aligned

    return rows, tracks


def print_accel_diagnostic(record, use_cache: bool) -> None:
    if Path(record.ulg_path).name != "d250_1.ulg":
        return
    imu = load_topic(record.ulg_path, "sensor_combined", use_cache=use_cache)
    attitude = load_topic(record.ulg_path, "vehicle_attitude", use_cache=use_cache)
    accel_cols = axis_fields(imu, ["accel", "accelerometer"])
    gyro_cols = axis_fields(imu, ["gyro"])
    quat_cols = ["q[0]", "q[1]", "q[2]", "q[3]"]
    _times, linear_accel, sign = world_linear_acceleration(imu, attitude, accel_cols, gyro_cols, quat_cols)
    mag = np.linalg.norm(linear_accel, axis=1)
    mean_vector_norm = float(np.linalg.norm(np.nanmean(linear_accel, axis=0)))
    print("  TASK4 AFTER gravity diagnostic d250_1:")
    print(f"    gravity sign used: {sign}")
    print(f"    mean |a_world_linear|: {np.nanmean(mag):.3f} m/s^2")
    print(f"    std  |a_world_linear|: {np.nanstd(mag):.3f} m/s^2")
    print(f"    |mean a_world_linear vector|: {mean_vector_norm:.3f} m/s^2")


def save_figure(tracks: dict[str, pd.DataFrame], output: Path) -> None:
    truth = tracks["truth"]
    imu = tracks["imu_aligned"]
    ekf = tracks.get("ekf_aligned")
    gps = tracks.get("gps")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(truth["y"], truth["x"], label="GPS truth", linewidth=2)
    axes[0].plot(imu["y"], imu["x"], label="IMU-only aligned", linewidth=1)
    if ekf is not None:
        axes[0].plot(ekf["y"], ekf["x"], label="PX4 EKF", linewidth=1)
    axes[0].set_xlabel("East (m)")
    axes[0].set_ylabel("North (m)")
    axes[0].set_title("Top-down trajectory")
    x_min, x_max = truth["y"].min(), truth["y"].max()
    y_min, y_max = truth["x"].min(), truth["x"].max()
    pad = max(5.0, 0.15 * max(float(x_max - x_min), float(y_max - y_min), 1.0))
    axes[0].set_xlim(float(x_min - pad), float(x_max + pad))
    axes[0].set_ylim(float(y_min - pad), float(y_max + pad))
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    times = common_time_grid(imu, truth)
    imu_xyz = resample_positions(imu, times)
    truth_xyz = resample_positions(truth, times)
    error = np.linalg.norm(imu_xyz - truth_xyz, axis=1)
    axes[1].plot(times - times[0], error, color="tab:red", label="IMU-only error")
    axes[1].set_xlabel("Seconds")
    axes[1].set_ylabel("Position error (m)")
    axes[1].set_title("Error over time")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    warnings.filterwarnings("ignore")
    args = parse_args()
    print_startup("Task 4 - Navigation odometry")
    flights = find_flights(args.data, args.subset)
    if args.max_flights:
        flights = flights[: args.max_flights]
    if not flights:
        raise SystemExit(f"No .ulg files found below {args.data}")
    if args.inspect:
        inspect_sample(args, flights)
        return

    rows: list[dict[str, float | str]] = []
    skips: list[str] = []
    figure_tracks: dict[str, pd.DataFrame] | None = None
    for index, record in enumerate(flights, start=1):
        print(f"[{index}/{len(flights)}] {record.flight_id}")
        try:
            print_accel_diagnostic(record, use_cache=not args.no_cache)
            flight_rows, tracks = process_flight(record, use_cache=not args.no_cache)
            rows.extend(flight_rows)
            if figure_tracks is None:
                figure_tracks = tracks
            print("  scored")
        except Exception as exc:
            skips.append(f"{record.ulg_path}: {exc}")
            print(f"  skip: {exc}")

    if not rows:
        raise SystemExit("No flights could be scored.")
    results = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    table_path = args.out / "task4_navigation_results.csv"
    figure_path = args.out / "task4_trajectory.png"
    results.to_csv(table_path, index=False)
    if figure_tracks is not None:
        save_figure(figure_tracks, figure_path)

    summary = (
        results.groupby("method")[["ate_rmse_m", "final_drift_percent"]]
        .mean()
        .reset_index()
        .sort_values("method")
    )
    print("\nMean summary by method")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nLarge IMU-only drift is expected; the benchmark quantifies that drift.")
    print(f"Saved {table_path}")
    if figure_tracks is not None:
        print(f"Saved {figure_path}")
    if skips:
        print("\nSkipped flights:")
        for item in skips:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
