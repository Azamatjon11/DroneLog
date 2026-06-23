"""ULog-native I/O, field discovery, cache handling, and flight discovery."""

from __future__ import annotations

from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import hashlib
import os
import platform
import re
import sys
import warnings

import numpy as np
import pandas as pd
from pyulog import ULog

from . import __version__
from .labels import FlightRecord, parse_frame_record, parse_payload_record


CACHE_DIR = Path(".cache")
os.environ.setdefault("ARROW_USER_SIMD_LEVEL", "NONE")


class MissingTopicError(RuntimeError):
    """Raised when a requested ULog topic is absent."""


class FieldMatchError(RuntimeError):
    """Raised when regex-based field discovery finds no columns."""


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _cache_dir_for(ulg_path: Path) -> Path:
    digest = hashlib.sha1(str(ulg_path.resolve()).encode("utf-8")).hexdigest()
    return CACHE_DIR / digest


def _cache_path(ulg_path: Path, topic_name: str, instance: int) -> Path:
    return _cache_dir_for(ulg_path) / f"{_safe_name(topic_name)}__instance{instance}.parquet"


def _is_fresh(cache_path: Path, source_path: Path) -> bool:
    return cache_path.exists() and cache_path.stat().st_mtime >= source_path.stat().st_mtime


def _topic_table(data) -> pd.DataFrame:
    frame = pd.DataFrame({name: np.asarray(values) for name, values in data.data.items()})
    if "timestamp" not in frame.columns:
        raise ValueError(f"Topic {data.name!r} has no timestamp field")
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce") / 1_000_000.0
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return frame


def _available_topics(ulg: ULog) -> list[str]:
    return [f"{data.name}[instance={data.multi_id}]" for data in ulg.data_list]


def list_topics(ulg_path: str | Path) -> list[str]:
    """Parse a ULog and return topic names exactly as recorded by the log."""

    ulg = ULog(str(ulg_path))
    names: list[str] = []
    seen: set[str] = set()
    for data in ulg.data_list:
        if data.name not in seen:
            names.append(data.name)
            seen.add(data.name)
    return names


def load_topic(
    ulg_path: str | Path,
    topic_name: str,
    instance: int = 0,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load one ULog topic as a DataFrame with timestamps converted to seconds.

    Field names are never guessed or normalized; columns are exactly the names
    exposed by ``pyulog`` plus the seconds-based ``timestamp`` column.
    """

    source = Path(ulg_path)
    cache_path = _cache_path(source, topic_name, instance)
    if use_cache and _is_fresh(cache_path, source):
        return pd.read_parquet(cache_path)

    ulg = ULog(str(source))
    matches = [data for data in ulg.data_list if data.name == topic_name and data.multi_id == instance]
    if not matches:
        present = ", ".join(_available_topics(ulg))
        raise MissingTopicError(
            f"Missing topic {topic_name!r} instance {instance} in {source}. "
            f"Available topics: {present}"
        )

    requested = _topic_table(matches[0])
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        for data in ulg.data_list:
            topic_path = _cache_path(source, data.name, data.multi_id)
            try:
                _topic_table(data).to_parquet(topic_path, index=False)
            except ImportError as exc:
                warnings.warn(
                    "Parquet cache disabled because pyarrow/fastparquet is unavailable. "
                    "Install requirements.txt to enable .cache parquet files.",
                    RuntimeWarning,
                )
                break
            except Exception as exc:  # pragma: no cover - cache failure must not block parsing.
                warnings.warn(f"Could not write cache file {topic_path}: {exc}", RuntimeWarning)
    return requested


def find_fields(df: pd.DataFrame, patterns: str | Iterable[str]) -> list[str]:
    """Return columns whose names match one or more regexes, case-insensitively."""

    if isinstance(patterns, str):
        patterns = [patterns]
    matches: list[str] = []
    for pattern in patterns:
        regex = re.compile(pattern, flags=re.IGNORECASE)
        for column in df.columns:
            if column == "timestamp":
                continue
            if regex.search(column) and column not in matches:
                matches.append(column)
    if not matches:
        raise FieldMatchError(
            f"No fields matched {list(patterns)!r}. Available columns: {list(df.columns)}"
        )
    return matches


def first_field(df: pd.DataFrame, patterns: str | Iterable[str]) -> str:
    """Return the first matching field for a field role."""

    return find_fields(df, patterns)[0]


def axis_fields(df: pd.DataFrame, prefix_patterns: Iterable[str]) -> list[str]:
    """Locate three axis fields, accepting both ``name[0]`` and x/y/z spellings."""

    fields: list[str] = []
    axis_patterns = [
        (r"\[0\]$", r"(^|_)x($|_)"),
        (r"\[1\]$", r"(^|_)y($|_)"),
        (r"\[2\]$", r"(^|_)z($|_)"),
    ]
    for index_pattern, named_pattern in axis_patterns:
        patterns = [
            rf"{prefix}.*{index_pattern}"
            for prefix in prefix_patterns
        ] + [
            rf"{prefix}.*{named_pattern}"
            for prefix in prefix_patterns
        ]
        fields.append(first_field(df, patterns))
    return fields


def _ulog_files(base: Path) -> list[Path]:
    return sorted([*base.rglob("*.ulg"), *base.rglob("*.ulog")])


def _frame_base(root: Path) -> Path:
    return root / "frame_size" if (root / "frame_size").exists() else root


def _payload_base(root: Path) -> Path:
    return root / "payload_detection" if (root / "payload_detection").exists() else root


def find_flights(root: str | Path, subset: str) -> list[FlightRecord]:
    """Recurse a dataset root and return records for ``frame``, ``payload``, or ``all``."""

    root_path = Path(root)
    subset = subset.lower()
    if subset not in {"frame", "payload", "all"}:
        raise ValueError("subset must be one of {'frame', 'payload', 'all'}")

    records: list[FlightRecord] = []
    if subset in {"frame", "all"}:
        base = _frame_base(root_path)
        for path in _ulog_files(base):
            if re.fullmatch(r"\d+mm", path.parent.name, flags=re.IGNORECASE):
                records.append(parse_frame_record(path, root_path))
    if subset in {"payload", "all"}:
        base = _payload_base(root_path)
        for path in _ulog_files(base):
            if re.fullmatch(r"\d+_scenario", path.stem, flags=re.IGNORECASE):
                records.append(parse_payload_record(path, root_path))
    return records


def clear_cache(cache_dir: str | Path = CACHE_DIR) -> int:
    """Delete cached parquet files and return the number of files removed."""

    cache = Path(cache_dir)
    if not cache.exists():
        return 0
    count = sum(1 for path in cache.rglob("*") if path.is_file())
    import shutil

    shutil.rmtree(cache)
    return count


def print_startup(script_name: str) -> None:
    """Print reproducibility-relevant package versions at script startup."""

    mpl_config = (CACHE_DIR / "matplotlib").resolve()
    mpl_config.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "fontconfig").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR.resolve()))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))

    import matplotlib
    import scipy
    import sklearn

    try:
        pyulog_version = version("pyulog")
    except PackageNotFoundError:
        pyulog_version = "unknown"

    print(f"\n{script_name}")
    print(f"Python {sys.version.split()[0]} on {platform.platform()}")
    print(
        "Versions: "
        f"dronelog {__version__}, pyulog {pyulog_version}, "
        f"pandas {pd.__version__}, numpy {np.__version__}, scipy {scipy.__version__}, "
        f"scikit-learn {sklearn.__version__}, matplotlib {matplotlib.__version__}"
    )
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


from .timebase import get_flight_t0, get_flight_time_bounds, to_flight_time  # noqa: E402
