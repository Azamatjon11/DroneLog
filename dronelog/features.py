"""Window features and dataset tables used by the revised (v1.1) baselines.

Time is measured in seconds from the start of the log (PX4 logs from arming),
which is the same time base as the pickup/drop times in metadata/events.csv.
Each unique log is parsed once and the signals the baselines need are cached
as .npz files keyed by the log's MD5 checksum.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from pyulog import ULog
from scipy.stats import kurtosis, skew

CACHE_DIR = Path(".cache") / "signals"
IMU_COLUMNS = ["ax", "ay", "az", "gx", "gy", "gz"]


# ----------------------------------------------------------------------------
# metadata
# ----------------------------------------------------------------------------
def load_metadata(data_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (flights, events) from ``<data_root>/metadata``.

    ``flights.csv`` has one row per log file; ``events.csv`` one row per
    pickup/drop event with ``time_s`` in seconds from the start of the log.
    """

    meta = Path(data_root) / "metadata"
    flights = pd.read_csv(meta / "flights.csv", dtype={"scenario": "Int64", "trial": "Int64"})
    events = pd.read_csv(meta / "events.csv")
    flights["path"] = [str(Path(data_root) / f) for f in flights["file"]]
    return flights, events


def fw_code(version: str) -> int:
    major, minor, patch = (int(p) for p in str(version).lstrip("v").split(".")[:3])
    return major * 10000 + minor * 100 + patch


def meta_features(row) -> list[float]:
    """Flight-level quantities unrelated to the dynamics (metadata-only control)."""

    return [row.imu_rate_hz, fw_code(row.px4_version), row.battery_v_start, row.duration_s]


# ----------------------------------------------------------------------------
# signal cache
# ----------------------------------------------------------------------------
def _extract(ulg_path: str | Path) -> dict[str, np.ndarray]:
    ulg = ULog(str(ulg_path), ["sensor_combined", "actuator_motors", "actuator_outputs", "battery_status"])
    t0 = ulg.start_timestamp
    sc = ulg.get_dataset("sensor_combined").data
    imu_t = (sc["timestamp"].astype(np.int64) - t0) / 1e6
    imu = np.stack(
        [sc[f"accelerometer_m_s2[{i}]"] for i in range(3)] + [sc[f"gyro_rad[{i}]"] for i in range(3)], 1
    ).astype(np.float32)
    names = {d.name for d in ulg.data_list}
    if "actuator_motors" in names:
        act = ulg.get_dataset("actuator_motors").data
        mot_t = (act["timestamp"].astype(np.int64) - t0) / 1e6
        mot = np.mean([act[f"control[{i}]"] for i in range(4)], 0)
    else:  # older PX4: PWM outputs, 1000-2000 us normalised to 0-1
        act = ulg.get_dataset("actuator_outputs").data
        mot_t = (act["timestamp"].astype(np.int64) - t0) / 1e6
        mot = (np.mean([act[f"output[{i}]"] for i in range(4)], 0) - 1000) / 1000
    bat = ulg.get_dataset("battery_status").data
    bat_t = (bat["timestamp"].astype(np.int64) - t0) / 1e6
    return dict(
        imu_t=imu_t, imu=imu, mot_t=mot_t, mot=mot.astype(np.float32),
        bat_t=bat_t, bat_v=bat["voltage_v"].astype(np.float32), dur=(ulg.last_timestamp - t0) / 1e6,
    )


_MEMORY: dict[str, dict[str, np.ndarray]] = {}


def load_signals(md5: str, ulg_path: str | Path, use_cache: bool = True) -> dict[str, np.ndarray]:
    """IMU, mean motor command and battery voltage of one log (cached by MD5)."""

    if md5 in _MEMORY:
        return _MEMORY[md5]
    cache = CACHE_DIR / f"{md5}.npz"
    if use_cache and cache.exists():
        signals = dict(np.load(cache))
    else:
        signals = _extract(ulg_path)
        if use_cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, **signals)
    _MEMORY[md5] = signals
    return signals


# ----------------------------------------------------------------------------
# window features
# ----------------------------------------------------------------------------
def stats(x: np.ndarray) -> np.ndarray:
    """Eight statistics per column: mean, std, min, max, median, RMS, skewness, kurtosis."""

    return np.concatenate([
        x.mean(0), x.std(0), x.min(0), x.max(0), np.median(x, 0), np.sqrt((x ** 2).mean(0)),
        np.nan_to_num(skew(x, 0)), np.nan_to_num(kurtosis(x, 0)),
    ])


def resample(t: np.ndarray, x: np.ndarray, hz: float) -> tuple[np.ndarray, np.ndarray]:
    grid = np.arange(t[0], t[-1], 1 / hz)
    return grid, np.stack([np.interp(grid, t, x[:, j]) for j in range(x.shape[1])], 1)


def windows(signals: dict, w: float, motor: bool = False, hz: float | None = None) -> list[tuple[float, np.ndarray]]:
    """Non-overlapping windows of ``w`` seconds over the whole log.

    Returns (window start, feature vector). IMU features are 6 channels x 8
    statistics; with ``motor=True`` the mean/std/min/max motor command and the
    mean battery voltage are appended (53 features).
    """

    t, x = signals["imu_t"], signals["imu"]
    if hz:
        t, x = resample(t, x, hz)
    end = float(signals["dur"])
    out = []
    start = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        while start + w <= end:
            mask = (t >= start) & (t < start + w)
            if mask.sum() >= 10:
                f = stats(x[mask])
                if motor:
                    mm = (signals["mot_t"] >= start) & (signals["mot_t"] < start + w)
                    mv = signals["mot"][mm] if mm.sum() else np.array([np.nan])
                    bm = (signals["bat_t"] >= start) & (signals["bat_t"] < start + w)
                    bv = signals["bat_v"][bm] if bm.sum() else np.array([np.nan])
                    f = np.concatenate([f, [np.nanmean(mv), np.nanstd(mv), np.nanmin(mv), np.nanmax(mv), np.nanmean(bv)]])
                out.append((start, f))
            start += w
    return out


# ----------------------------------------------------------------------------
# payload labels
# ----------------------------------------------------------------------------
def events_of(events: pd.DataFrame, file: str) -> dict[str, float]:
    sel = events[events["file"] == file]
    return {r.event: r.time_s for r in sel.itertuples()}


def state_at(scenario: int, ev: dict[str, float], t: float) -> int:
    """1 if the payload is attached at time ``t`` (s from log start), else 0."""

    s = int(scenario)
    if s == 1:
        return 0
    if s == 2:
        return 1
    if s == 3:
        return int(t >= ev["pickup"])
    if s == 4:
        return int(ev["pickup"] <= t < ev["drop"])
    if s == 5:
        return int(t < ev["drop"])
    raise ValueError(f"unknown scenario {scenario}")


def payload_table(flights: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    p = flights[flights["subset"] == "payload"].reset_index(drop=True).copy()
    p["mass"] = p["payload_mass_kg"].astype(float)
    return p


def payload_windows(payload: pd.DataFrame, events: pd.DataFrame, w: float, motor: bool = True,
                    margin: float = 1.0, use_cache: bool = True) -> pd.DataFrame:
    """One row per window of every payload log, labelled with the payload state.

    Windows that contain an event or lie within ``margin`` s of one are dropped.
    """

    rows = []
    for _, r in payload.iterrows():
        ev = events_of(events, r.file)
        sig = load_signals(r.md5, r.path, use_cache)
        for s, f in windows(sig, w, motor=motor):
            if any(s - margin < x < s + w + margin for x in ev.values()):
                continue
            st = state_at(r.scenario, ev, s + w / 2)
            rows.append(dict(file=r.file, md5=r.md5, platform=r.platform, mass=r.mass, scenario=int(r.scenario),
                             trial=int(r.trial), t=s, state=st, carried=r.mass * st, f=f, meta=meta_features(r)))
    return pd.DataFrame(rows)
