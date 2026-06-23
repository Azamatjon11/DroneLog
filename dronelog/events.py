"""Payload pickup/drop annotations from the DroneLog paper Table 4."""

from __future__ import annotations

from pathlib import Path
import csv


# Paper Table 4, seconds into flight. Scenarios 1 and 2 have no transitions.
PAPER_TABLE_4: dict[tuple[str, float, int], dict[str, float | None]] = {
    ("x500v2", 0.5, 3): {"pickup": 84.0, "drop": None},
    ("x500v2", 0.5, 4): {"pickup": 103.0, "drop": 179.0},
    ("x500v2", 0.5, 5): {"pickup": None, "drop": 88.0},
    ("x500v2", 0.75, 3): {"pickup": 124.0, "drop": None},
    ("x500v2", 0.75, 4): {"pickup": 115.0, "drop": 208.0},
    ("x500v2", 0.75, 5): {"pickup": None, "drop": 103.0},
    ("x500v2", 1.0, 3): {"pickup": 138.0, "drop": None},
    ("x500v2", 1.0, 4): {"pickup": 100.0, "drop": 166.0},
    ("x500v2", 1.0, 5): {"pickup": None, "drop": 82.0},
    ("x650", 0.5, 3): {"pickup": 101.0, "drop": None},
    ("x650", 0.5, 4): {"pickup": 128.0, "drop": 231.0},
    ("x650", 0.5, 5): {"pickup": None, "drop": 189.0},
    ("x650", 1.0, 3): {"pickup": 98.0, "drop": None},
    ("x650", 1.0, 4): {"pickup": 109.0, "drop": 185.0},
    ("x650", 1.0, 5): {"pickup": None, "drop": 125.0},
    ("x650", 1.5, 3): {"pickup": 149.0, "drop": None},
    ("x650", 1.5, 4): {"pickup": 94.0, "drop": 168.0},
    ("x650", 1.5, 5): {"pickup": None, "drop": 335.0},
    ("x650", 2.0, 3): {"pickup": 120.0, "drop": None},
    ("x650", 2.0, 4): {"pickup": 168.0, "drop": 323.0},
    ("x650", 2.0, 5): {"pickup": None, "drop": 335.0},
}


def _parse_seconds(value: str | None) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def load_event_table(events_csv: str | Path | None = None) -> dict[tuple[str, float, int], dict[str, float | None]]:
    """Return Paper Table 4 annotations, optionally overridden by a CSV file.

    CSV override columns: ``platform,mass_kg,scenario,pickup,drop``.
    Blank pickup/drop values are treated as no event.
    """

    table = dict(PAPER_TABLE_4)
    if events_csv is None:
        return table

    with Path(events_csv).open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (row["platform"].strip().lower(), float(row["mass_kg"]), int(row["scenario"]))
            table[key] = {"pickup": _parse_seconds(row.get("pickup")), "drop": _parse_seconds(row.get("drop"))}
    return table


def get_events(
    platform: str,
    mass_kg: float,
    scenario: int,
    table: dict[tuple[str, float, int], dict[str, float | None]] | None = None,
) -> dict[str, float | None]:
    """Return pickup/drop seconds for one payload condition."""

    table = table or PAPER_TABLE_4
    return table.get((platform.lower(), float(mass_kg), int(scenario)), {"pickup": None, "drop": None})


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
