#!/usr/bin/env python3
"""Regenerate every result table of the revised DroneLog paper.

    python run_all.py --data <dataset_root> [--out outputs]

<dataset_root> is the unpacked DroneLog v1.1 release (it contains log/ and
metadata/). Runs Tasks 1, 2, 3 and 5 in order; every script uses fixed random
seeds (cross-validation repetition r uses random_state=r).
Task 4 (navigation) is kept from v1.0 but is not part of the revised paper;
run it separately if needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time

TASKS = [
    ("task1_frame_classification.py", "Table 7  - platform identification"),
    ("task2_payload_classification.py", "Table 8  - payload state; flight-level scenario"),
    ("task3_mass_regression.py", "Tables 9-10 - payload mass"),
    ("task5_event_detection.py", "Table 11 - pickup/drop detection"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, type=Path, help="Dataset root (contains log/ and metadata/)")
    parser.add_argument("--out", default=Path("outputs"), type=Path, help="Output directory. Default: outputs/")
    parser.add_argument("--no-cache", action="store_true", help="Re-parse the ULog files instead of using .cache")
    args = parser.parse_args()
    if not (args.data / "metadata" / "flights.csv").exists():
        raise SystemExit(f"{args.data}/metadata/flights.csv not found - point --data at the DroneLog v1.1 root.")

    scripts = Path(__file__).resolve().parent / "scripts"
    extra = ["--no-cache"] if args.no_cache else []
    for script, what in TASKS:
        print(f"\n=== {what} ({script}) ===", flush=True)
        start = time.time()
        subprocess.run([sys.executable, str(scripts / script), "--data", str(args.data), "--out", str(args.out), *extra],
                       check=True)
        print(f"--- done in {time.time() - start:.0f} s", flush=True)
    print(f"\nAll results written to {args.out}/")


if __name__ == "__main__":
    main()
