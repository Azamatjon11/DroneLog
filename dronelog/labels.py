"""Path-label parsing for the DroneLog dataset layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class FlightRecord:
    """One ULog file plus labels parsed from its dataset path."""

    ulg_path: Path
    flight_id: str
    frame_mm: int | None = None
    platform: str | None = None
    mass_kg: float | None = None
    scenario: int | None = None


def relative_stem(path: Path, root: Path) -> str:
    """Return the full path stem relative to root, suitable as a grouped-CV ID."""

    try:
        return path.resolve().relative_to(root.resolve()).with_suffix("").as_posix()
    except ValueError:
        return path.with_suffix("").as_posix()


def parse_frame_record(path: Path, root: Path) -> FlightRecord:
    """Parse a frame-size record such as ``frame_size/250mm/d250_1.ulg``."""

    match = re.fullmatch(r"(\d+)mm", path.parent.name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse frame size from parent folder: {path.parent}")
    return FlightRecord(
        ulg_path=path,
        flight_id=relative_stem(path, root),
        frame_mm=int(match.group(1)),
    )


def parse_payload_record(path: Path, root: Path) -> FlightRecord:
    """Parse a payload record such as ``x500v2/0.75kg/3_scenario.ulg``."""

    mass_match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)kg", path.parent.name, flags=re.IGNORECASE)
    scenario_match = re.fullmatch(r"(\d+)_scenario", path.stem, flags=re.IGNORECASE)
    if not mass_match:
        raise ValueError(f"Cannot parse payload mass from parent folder: {path.parent}")
    if not scenario_match:
        raise ValueError(f"Cannot parse payload scenario from filename: {path.name}")
    return FlightRecord(
        ulg_path=path,
        flight_id=relative_stem(path, root),
        platform=path.parent.parent.name.lower(),
        mass_kg=float(mass_match.group(1)),
        scenario=int(scenario_match.group(1)),
    )
