#!/usr/bin/env python3
"""Task 1: platform identification on the frame-size subset.

Five classes (250/450/500/650/680 mm), one vehicle each, so the task identifies
the platform rather than the frame geometry. IMU statistics of non-overlapping
windows (2, 5, 10 s), four classifiers, stratified grouped 3-fold CV by flight,
repeated 10 times. Two controls: a metadata-only model (effective IMU rate and
firmware version) and IMU data resampled to a common 200 Hz.
Reproduces Table 7 of the revised paper.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold

from _common import REPEATS, classifiers, fmt, per_flight
from dronelog.features import load_metadata, load_signals, meta_features, windows
from dronelog.io import print_startup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 1: platform identification (frame-size subset).")
    parser.add_argument("--data", default=".", type=Path, help="Dataset root (contains log/ and metadata/)")
    parser.add_argument("--out", default=Path("outputs"), type=Path, help="Output directory. Default: outputs/")
    parser.add_argument("--no-cache", action="store_true", help="Re-parse the ULog files instead of using .cache")
    return parser.parse_args()


def build(frames: pd.DataFrame, w: float, hz: float | None = None, meta: bool = False, use_cache: bool = True):
    X, y, g, f = [], [], [], []
    for _, r in frames.iterrows():
        sig = load_signals(r.md5, r.path, use_cache)
        for _s, feat in windows(sig, w, hz=hz):
            X.append(meta_features(r)[:2] if meta else feat)
            y.append(r.platform)
            g.append(r.md5)
            f.append(r.file)
    return np.array(X), np.array(y), np.array(g), np.array(f)


def cv(X, y, g, make, keep_pred: bool = False):
    scores, first = [], None
    for rep in range(REPEATS):
        pred = np.empty(len(y), dtype=object)
        for tr, te in StratifiedGroupKFold(3, shuffle=True, random_state=rep).split(X, y, g):
            model = make(rep)
            model.fit(X[tr], y[tr])
            pred[te] = model.predict(X[te])
        scores.append([
            accuracy_score(y, pred), f1_score(y, pred, average="macro"),
            precision_score(y, pred, average="macro", zero_division=0), recall_score(y, pred, average="macro"),
        ])
        if keep_pred and rep == 0:
            first = pred
    return np.array(scores), first


def row(window, model, s, y, g):
    return dict(window=window, model=model, acc=s[:, 0].mean(), acc_sd=s[:, 0].std(), f1=s[:, 1].mean(),
                f1_sd=s[:, 1].std(), prec=s[:, 2].mean(), rec=s[:, 3].mean(), windows=len(y), groups=len(set(g)))


def main() -> None:
    warnings.filterwarnings("ignore")
    args = parse_args()
    print_startup("Task 1 - Platform identification")
    flights, _ = load_metadata(args.data)
    frames = flights[flights["subset"] == "frame_size"].reset_index(drop=True)
    use_cache = not args.no_cache
    models = classifiers()

    out, per_flight_df = [], None
    for w in [2, 5, 10]:
        X, y, g, f = build(frames, w, use_cache=use_cache)
        for name, make in models.items():
            s, pred = cv(X, y, g, make, keep_pred=(w == 5 and name == "Random Forest"))
            out.append(row(w, name, s, y, g))
            print(f"{w:>2} s  {name:<20} acc {fmt(s[:, 0].mean(), s[:, 0].std())}", flush=True)
            if pred is not None:
                per_flight_df = per_flight(f, y, pred)
    X, y, g, _ = build(frames, 5, meta=True, use_cache=use_cache)
    s, _ = cv(X, y, g, models["Random Forest"])
    out.append(row(5, "Metadata only (RF)", s, y, g))
    X, y, g, _ = build(frames, 5, hz=200, use_cache=use_cache)
    s, _ = cv(X, y, g, models["Random Forest"])
    out.append(row(5, "Random Forest, resampled to 200 Hz", s, y, g))

    args.out.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(out)
    results.to_csv(args.out / "task1_frame_results.csv", index=False)
    per_flight_df.to_csv(args.out / "task1_per_flight.csv", index=False)
    print("\n" + results.round(4).to_string(index=False))
    print(f"\nSaved {args.out / 'task1_frame_results.csv'} and per-flight accuracy (RF, 5 s, repetition 0).")


if __name__ == "__main__":
    main()
