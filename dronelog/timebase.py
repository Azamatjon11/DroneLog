"""Flight-relative timebase helpers.

Paper Table 4 event annotations are seconds into flight, not seconds from the
first ULog sample. These helpers centralize the conversion so tasks do not mix
log-record time with takeoff-relative event time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _load(path: str | Path, topic: str, use_cache: bool = True) -> pd.DataFrame:
    from .io import load_topic

    return load_topic(path, topic, use_cache=use_cache)


def _airborne_from_land_detected(ulg_path: str | Path, use_cache: bool = True) -> tuple[float, float] | None:
    try:
        landed = _load(ulg_path, "vehicle_land_detected", use_cache=use_cache)
    except Exception:
        return None
    if "landed" not in landed.columns or landed.empty:
        return None
    airborne = landed[landed["landed"].astype(float) == 0.0]
    if airborne.empty:
        return None
    return float(airborne["timestamp"].iloc[0]), float(airborne["timestamp"].iloc[-1])


def _airborne_from_local_position(ulg_path: str | Path, use_cache: bool = True) -> tuple[float, float] | None:
    try:
        local = _load(ulg_path, "vehicle_local_position", use_cache=use_cache)
    except Exception:
        return None
    if "z" not in local.columns or local.empty:
        return None

    z = local["z"].to_numpy(dtype=float)
    t = local["timestamp"].to_numpy(dtype=float)

    # PX4 NED convention is z<0 when moving up from the origin. Some logs in
    # this dataset use a shifted local origin, so after trying the documented
    # threshold, fall back to "moved more than 0.5 m from initial ground z".
    direct = np.flatnonzero(z < -0.5)
    if direct.size:
        return float(t[direct[0]]), float(t[direct[-1]])

    initial_window = t <= (t[0] + 2.0)
    ground_z = float(np.nanmedian(z[initial_window])) if np.any(initial_window) else float(z[0])
    relative = np.flatnonzero(np.abs(z - ground_z) > 0.5)
    if relative.size:
        return float(t[relative[0]]), float(t[relative[-1]])
    return None


def get_flight_time_bounds(
    ulg_path: str | Path,
    *,
    use_cache: bool = True,
    trim_start_s: float = 0.0,
    trim_end_s: float = 0.0,
) -> tuple[float, float]:
    """Return airborne start/end timestamps in the log's seconds timebase."""

    bounds = _airborne_from_land_detected(ulg_path, use_cache=use_cache)
    if bounds is None:
        bounds = _airborne_from_local_position(ulg_path, use_cache=use_cache)
    if bounds is None:
        sensor = _load(ulg_path, "sensor_combined", use_cache=use_cache)
        bounds = float(sensor["timestamp"].iloc[0]), float(sensor["timestamp"].iloc[-1])

    start = bounds[0] + trim_start_s
    end = bounds[1] - trim_end_s
    if end <= start:
        return bounds
    return start, end


def get_flight_t0(ulg_path: str | Path, *, use_cache: bool = True) -> float:
    """Detect takeoff time in seconds, in the same base as ``load_topic`` timestamps."""

    return get_flight_time_bounds(ulg_path, use_cache=use_cache)[0]


def to_flight_time(df: pd.DataFrame, t0: float) -> pd.DataFrame:
    """Return a copy of ``df`` with ``t_flight = timestamp - t0``."""

    result = df.copy()
    result["t_flight"] = result["timestamp"] - float(t0)
    return result
