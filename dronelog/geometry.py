"""Navigation geometry helpers for PX4 NED/FRD telemetry.

Conventions used by the benchmark:

* PX4 world frame is NED: +x North, +y East, +z Down.
* PX4 body frame is FRD: +x Forward, +y Right, +z Down.
* ``sensor_combined`` accelerometer data is body-frame FRD specific force in m/s^2.
* IMU-only dead reckoning rotates body specific force into NED, adds gravity
  ``[0, 0, 9.80665]`` to recover linear acceleration, and integrates that to
  velocity and position.
  This deliberately uses no GPS; GPS/EKF data is used only for scoring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


GRAVITY_NED = np.array([0.0, 0.0, 9.80665])
EARTH_RADIUS_M = 6_378_137.0


def quat_wxyz_to_rotation(q: np.ndarray) -> Rotation:
    """Convert PX4 ``[w, x, y, z]`` quaternion fields to a SciPy rotation."""

    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if norm <= 1e-12:
        return Rotation.identity()
    q = q / norm
    return Rotation.from_quat([q[1], q[2], q[3], q[0]])


def world_linear_acceleration(
    imu: pd.DataFrame,
    attitude: pd.DataFrame,
    accel_cols: list[str],
    gyro_cols: list[str],
    quat_cols: list[str],
    *,
    attitude_correction: bool = True,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Rotate body specific force to NED and add the correct gravity sign.

    PX4 quaternions are Hamilton quaternions stored as ``[w, x, y, z]``. SciPy
    expects ``[x, y, z, w]``; ``quat_wxyz_to_rotation`` performs that conversion.
    The resulting rotation maps body FRD vectors to world NED vectors. At rest,
    the rotated accelerometer specific force points upward, so adding
    ``g_ned=[0,0,+9.80665]`` yields near-zero linear acceleration.
    """

    timestamps = imu["timestamp"].to_numpy(dtype=float)
    accel_body = imu[accel_cols].to_numpy(dtype=float)
    gyro_body = imu[gyro_cols].to_numpy(dtype=float)
    finite = np.isfinite(timestamps) & np.all(np.isfinite(accel_body), axis=1) & np.all(np.isfinite(gyro_body), axis=1)
    timestamps = timestamps[finite]
    accel_body = accel_body[finite]
    gyro_body = gyro_body[finite]
    if len(timestamps) < 2:
        return timestamps, np.empty((0, 3)), "+g_ned"

    first_att = attitude.iloc[0][quat_cols].to_numpy(dtype=float).copy() if not attitude.empty else np.array([1, 0, 0, 0])
    rotation = quat_wxyz_to_rotation(first_att)
    accel_ned = np.zeros_like(accel_body)

    attitude_by_imu: np.ndarray | None = None
    if attitude_correction and not attitude.empty:
        # Gyro-only attitude drift quickly dominates double integration. The
        # logged PX4 attitude topic is still an attitude signal, not a position
        # truth signal; using it here keeps this benchmark focused on inertial
        # position drift instead of runaway orientation drift.
        imu_times = pd.DataFrame({"timestamp": timestamps})
        attitude_work = attitude[["timestamp"] + quat_cols].sort_values("timestamp").copy()
        matched = pd.merge_asof(
            imu_times,
            attitude_work,
            on="timestamp",
            direction="nearest",
            tolerance=0.05,
        )
        if matched[quat_cols].notna().all(axis=1).any():
            attitude_by_imu = matched[quat_cols].ffill().bfill().to_numpy(dtype=float)

    accel_ned[0] = rotation.apply(accel_body[0].copy()) + GRAVITY_NED

    for i in range(1, len(timestamps)):
        dt = max(float(timestamps[i] - timestamps[i - 1]), 0.0)
        # Quaternion exponential map: body-frame rotation vector = gyro * dt.
        rotation = rotation * Rotation.from_rotvec((gyro_body[i - 1] * dt).copy())
        if attitude_by_imu is not None:
            rotation = quat_wxyz_to_rotation(attitude_by_imu[i].copy())
        accel_ned[i] = rotation.apply(accel_body[i].copy()) + GRAVITY_NED

    sign = "+g_ned with attitude correction" if attitude_by_imu is not None else "+g_ned"
    return timestamps, accel_ned, sign


def strapdown_dead_reckon(
    imu: pd.DataFrame,
    attitude: pd.DataFrame,
    accel_cols: list[str],
    gyro_cols: list[str],
    quat_cols: list[str],
    *,
    bias_window: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Naive strapdown integration from IMU only.

    The initial orientation comes from the first PX4 attitude quaternion. After
    that, orientation is propagated using only gyroscope increments. A constant
    pre-takeoff accelerometer bias estimate is removed before integration to
    avoid turning small sensor offsets into kilometer-scale drift.
    """

    if imu.empty:
        return pd.DataFrame(columns=["timestamp", "x", "y", "z"])

    timestamps, accel_ned, _sign = world_linear_acceleration(imu, attitude, accel_cols, gyro_cols, quat_cols)
    if len(timestamps) < 2:
        return pd.DataFrame(columns=["timestamp", "x", "y", "z"])

    if bias_window is not None:
        start, end = bias_window
        mask = (timestamps >= start) & (timestamps <= end)
        if np.count_nonzero(mask) >= 5:
            accel_ned = accel_ned - np.nanmean(accel_ned[mask], axis=0)

    velocity = np.zeros_like(accel_ned)
    position = np.zeros_like(accel_ned)
    for i in range(1, len(timestamps)):
        dt = max(float(timestamps[i] - timestamps[i - 1]), 0.0)
        velocity[i] = velocity[i - 1] + 0.5 * (accel_ned[i - 1] + accel_ned[i]) * dt
        position[i] = position[i - 1] + 0.5 * (velocity[i - 1] + velocity[i]) * dt

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "x": position[:, 0],
            "y": position[:, 1],
            "z": position[:, 2],
        }
    )


def local_position_ned(df: pd.DataFrame) -> pd.DataFrame:
    """Extract NED position from PX4 local-position topic fields."""

    return df[["timestamp", "x", "y", "z"]].dropna().sort_values("timestamp").reset_index(drop=True)


def _scaled_lat_lon(values: np.ndarray) -> np.ndarray:
    values = values.astype(float)
    return values * 1e-7 if np.nanmax(np.abs(values)) > 1000 else values


def _scaled_alt(values: np.ndarray) -> np.ndarray:
    values = values.astype(float)
    return values * 1e-3 if np.nanmax(np.abs(values)) > 1000 else values


def gps_to_local_ned(df: pd.DataFrame) -> pd.DataFrame:
    """Project raw GPS latitude/longitude/altitude to local NED about first fix.

    Equirectangular projection around the first valid fix:
    north = (lat-lat0) * R, east = (lon-lon0) * R * cos(lat0), down = -(alt-alt0).
    """

    lat_col = "latitude_deg" if "latitude_deg" in df.columns else "lat"
    lon_col = "longitude_deg" if "longitude_deg" in df.columns else "lon"
    alt_candidates = ["altitude_msl_m", "alt", "altitude_ellipsoid_m", "alt_ellipsoid"]
    alt_col = next((col for col in alt_candidates if col in df.columns), None)
    if alt_col is None:
        raise ValueError(f"No GPS altitude field found. Available columns: {list(df.columns)}")

    optional_cols = [col for col in ["fix_type", "eph", "epv"] if col in df.columns]
    work = df[["timestamp", lat_col, lon_col, alt_col] + optional_cols].dropna().copy()
    lat_deg = _scaled_lat_lon(work[lat_col].to_numpy())
    lon_deg = _scaled_lat_lon(work[lon_col].to_numpy())
    alt = _scaled_alt(work[alt_col].to_numpy())
    valid = np.isfinite(lat_deg) & np.isfinite(lon_deg) & np.isfinite(alt)
    valid &= (np.abs(lat_deg) > 1e-9) & (np.abs(lon_deg) > 1e-9)
    if "fix_type" in work.columns:
        valid &= work["fix_type"].to_numpy(dtype=float) >= 3
    if "eph" in work.columns:
        valid &= work["eph"].to_numpy(dtype=float) < 100.0
    if "epv" in work.columns:
        valid &= work["epv"].to_numpy(dtype=float) < 100.0
    work = work.loc[valid].reset_index(drop=True)
    lat_deg = lat_deg[valid]
    lon_deg = lon_deg[valid]
    alt = alt[valid]
    if len(work) == 0:
        return pd.DataFrame(columns=["timestamp", "x", "y", "z"])
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)

    lat0 = lat[0]
    lon0 = lon[0]
    alt0 = alt[0]
    north = (lat - lat0) * EARTH_RADIUS_M
    east = (lon - lon0) * EARTH_RADIUS_M * np.cos(lat0)
    down = -(alt - alt0)
    return pd.DataFrame(
        {"timestamp": work["timestamp"].to_numpy(dtype=float), "x": north, "y": east, "z": down}
    )


def resample_positions(source: pd.DataFrame, target_times: np.ndarray) -> np.ndarray:
    """Interpolate a position DataFrame to target timestamps."""

    if source.empty:
        return np.empty((0, 3))
    src_t = source["timestamp"].to_numpy(dtype=float)
    result = np.column_stack(
        [np.interp(target_times, src_t, source[axis].to_numpy(dtype=float)) for axis in ["x", "y", "z"]]
    )
    return result


def common_time_grid(*frames: pd.DataFrame, max_points: int = 3000) -> np.ndarray:
    """Create a common time grid over the overlap of several position tracks."""

    starts = [float(frame["timestamp"].iloc[0]) for frame in frames if not frame.empty]
    ends = [float(frame["timestamp"].iloc[-1]) for frame in frames if not frame.empty]
    if not starts or not ends:
        return np.array([])
    start = max(starts)
    end = min(ends)
    if end <= start:
        return np.array([])
    lengths = [len(frame[(frame["timestamp"] >= start) & (frame["timestamp"] <= end)]) for frame in frames if not frame.empty]
    count = max(2, min(max_points, min(lengths) if lengths else max_points))
    return np.linspace(start, end, count)


def umeyama_align(source: np.ndarray, target: np.ndarray, *, with_scale: bool = False) -> np.ndarray:
    """Align ``source`` points to ``target`` with a rigid transform, optionally scale."""

    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if len(source) == 0 or len(source) != len(target):
        return source
    src_mean = source.mean(axis=0)
    tgt_mean = target.mean(axis=0)
    src_centered = source - src_mean
    tgt_centered = target - tgt_mean
    covariance = src_centered.T @ tgt_centered / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    scale = 1.0
    if with_scale:
        variance = np.mean(np.sum(src_centered * src_centered, axis=1))
        if variance > 1e-12:
            scale = float(np.sum(singular_values) / variance)
    return scale * (src_centered @ rotation) + tgt_mean
