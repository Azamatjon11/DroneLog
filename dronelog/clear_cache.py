"""Command-line helper to remove DroneLog parquet caches."""

from __future__ import annotations

import argparse

from .io import CACHE_DIR, clear_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Clear DroneLog .cache parquet files.")
    parser.add_argument("--cache-dir", default=CACHE_DIR, help="Cache directory to remove. Default: .cache")
    args = parser.parse_args()
    removed = clear_cache(args.cache_dir)
    print(f"Removed {removed} cached files from {args.cache_dir}")


if __name__ == "__main__":
    main()
