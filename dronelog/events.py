"""Payload pickup/drop annotations (DroneLog v1.1: ``metadata/events.csv``)."""

from __future__ import annotations

from pathlib import Path
import csv


def load_event_table(events_csv: str | Path) -> dict[tuple[str, float, int, int], dict[str, float | None]]:
    """Read ``metadata/events.csv`` into {(platform, mass_kg, scenario, trial): {"pickup": s, "drop": s}}.

    Times are seconds after arming (log start). Scenarios 1 and 2 have no events.
    """

    table: dict[tuple[str, float, int, int], dict[str, float | None]] = {}
    with Path(events_csv).open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["platform"].strip().lower(), float(row["payload_mass_kg"]), int(row["scenario"]), int(row["trial"]))
            table.setdefault(key, {"pickup": None, "drop": None})[row["event"]] = float(row["time_s"])
    return table


def get_events(
    platform: str,
    mass_kg: float,
    scenario: int,
    trial: int,
    table: dict[tuple[str, float, int, int], dict[str, float | None]],
) -> dict[str, float | None]:
    """Return pickup/drop seconds for one payload flight."""

    return table.get((platform.lower(), float(mass_kg), int(scenario), int(trial)), {"pickup": None, "drop": None})


def carried_mass_for_window(
    base_mass_kg: float,
    scenario: int,
    start_s: float,
    end_s: float,
    events: dict[str, float | None],
) -> float:
    """Label a window by the payload mass actually carried at its midpoint."""

    midpoint = 0.5 * (start_s + end_s)
    pickup = events.get("pickup")
    drop = events.get("drop")
    if scenario == 1:
        return 0.0
    if scenario == 2:
        return float(base_mass_kg)
    if scenario == 3:
        return float(base_mass_kg) if pickup is not None and midpoint >= pickup else 0.0
    if scenario == 4:
        after_pickup = pickup is not None and midpoint >= pickup
        before_drop = drop is None or midpoint < drop
        return float(base_mass_kg) if after_pickup and before_drop else 0.0
    if scenario == 5:
        return float(base_mass_kg) if drop is None or midpoint < drop else 0.0
    return float(base_mass_kg)
