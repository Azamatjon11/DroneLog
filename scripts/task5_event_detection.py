#!/usr/bin/env python3
"""Task 5: payload pickup/drop event detection.

Signal: mean motor command in the cruise phase (15 s after take-off to 15 s
before landing, resampled to 10 Hz, 1 s median filter). Detectors:
  * thresholded change-point: s(t) = mean(x[t, t+3 s)) - mean(x[t-3 s, t));
    every run of |s| >= threshold gives one detection at its peak;
  * Pelt (ruptures, l2 cost, min_size 3 s, jump 0.5 s) with penalty beta.
The sign of the step gives the event type (up = pickup, down = drop).
The threshold and penalty are chosen on the training flights only (best F1
at tau = 2 s), within grouped 5-fold CV repeated 10 times. Detections are
matched one-to-one to annotated events of the same type in order of
increasing time difference; a detection is correct within tolerance tau.
Scored on the pickup/drop flights of trials 2-3. Reproduces Table 11.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import os
import warnings

import numpy as np
import pandas as pd
import ruptures as rpt
from sklearn.model_selection import GroupKFold

from _common import REPEATS
from dronelog.features import events_of, load_metadata, load_signals, payload_table
from dronelog.io import print_startup

TYPES = ["pickup", "drop"]
TAUS = [1, 2, 5]
THRESHOLDS = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12]
PENALTIES = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0]
EXAMPLE = "log/payload/x500v2/0.75kg/4_scenario_t2.ulg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 5: pickup/drop event detection.")
    parser.add_argument("--data", default=".", type=Path, help="Dataset root (contains log/ and metadata/)")
    parser.add_argument("--out", default=Path("outputs"), type=Path, help="Output directory. Default: outputs/")
    parser.add_argument("--no-cache", action="store_true", help="Re-parse the ULog files instead of using .cache")
    return parser.parse_args()


def cruise_signal(sig):
    t, m = sig["mot_t"], sig["mot"]
    air = t[m > 0.2]
    grid = np.arange(air[0] + 15, air[-1] - 15, 0.1)
    x = np.interp(grid, t, m)
    k = 11  # 1 s median filter at 10 Hz
    xs = np.array([np.median(x[max(0, i - k // 2):i + k // 2 + 1]) for i in range(len(x))])
    return grid, xs


def step_statistic(x, half=3.0):
    n = int(half / 0.1)
    s = np.zeros(len(x))
    for i in range(n, len(x) - n):
        s[i] = x[i:i + n].mean() - x[i - n:i].mean()
    return s


class Detectors:
    def __init__(self, signals):
        self.sig = signals
        self.step = {h: step_statistic(signals[h][1]) for h in signals}
        self._pelt = {}

    def threshold(self, h, th):
        t, s, out, i = self.sig[h][0], self.step[h], [], 0
        while i < len(s):
            if abs(s[i]) >= th:
                j = i
                while j < len(s) and abs(s[j]) >= th and np.sign(s[j]) == np.sign(s[i]):
                    j += 1
                k = i + np.argmax(np.abs(s[i:j]))
                out.append((t[k], "pickup" if s[k] > 0 else "drop"))
                i = j
            else:
                i += 1
        return out

    def pelt(self, h, pen):
        t, x = self.sig[h]
        if (h, pen) not in self._pelt:
            self._pelt[(h, pen)] = rpt.Pelt(model="l2", min_size=30, jump=5).fit(x.reshape(-1, 1)).predict(pen=pen)[:-1]
        return [(t[b], "pickup" if x[b:b + 30].mean() > x[max(0, b - 30):b].mean() else "drop")
                for b in self._pelt[(h, pen)]]


def match(dets, truth, tau, typ):
    D = [d for d, ty in dets if ty == typ]
    T = [x for x, ty in truth if ty == typ]
    pairs = sorted((abs(d - x), i, j) for i, d in enumerate(D) for j, x in enumerate(T) if abs(d - x) <= tau)
    used_d, used_t, err = set(), set(), []
    for e, i, j in pairs:
        if i in used_d or j in used_t:
            continue
        used_d.add(i)
        used_t.add(j)
        err.append(e)
    return len(err), len(D), len(T), err


def evaluate(rows, detect, param, tau, truth):
    """Per event type: (true positives, detections, annotations, timing errors) summed over flights."""

    res = {}
    for typ in TYPES:
        tp = nd = nt = 0
        errs = []
        for r in rows:
            a, b, c, e = match(detect(r.md5, param), truth[r.file], tau, typ)
            tp, nd, nt, errs = tp + a, nd + b, nt + c, errs + e
        res[typ] = (tp, nd, nt, errs)
    return res


def f1_of(res):
    tp = sum(v[0] for v in res.values())
    nd = sum(v[1] for v in res.values())
    nt = sum(v[2] for v in res.values())
    p = tp / nd if nd else 0
    r = tp / nt if nt else 0
    return 2 * p * r / (p + r) if p + r else 0


def main() -> None:
    warnings.filterwarnings("ignore")
    args = parse_args()
    print_startup("Task 5 - Pickup/drop event detection")
    flights, events = load_metadata(args.data)
    payload = payload_table(flights, events)
    use_cache = not args.no_cache
    F = payload[(payload.scenario.astype(int) >= 3) & (payload.trial.astype(int) > 1)].reset_index(drop=True)
    signals = {r.md5: cruise_signal(load_signals(r.md5, r.path, use_cache)) for r in F.itertuples()}
    truth = {r.file: [(e.time_s, e.event) for e in events[events.file == r.file].itertuples()] for r in F.itertuples()}
    det = Detectors(signals)
    grids = {"Thresholded change-point": (det.threshold, THRESHOLDS), "Pelt change-point": (det.pelt, PENALTIES)}
    rows = list(F.itertuples())
    groups = np.array([r.md5 for r in rows])
    cruise_min = {r.md5: (signals[r.md5][0][-1] - signals[r.md5][0][0]) / 60 for r in rows}
    out, detections = [], []

    for name, (fn, grid) in grids.items():
        acc = {(typ, tau): [] for typ in TYPES for tau in TAUS}
        fa = {typ: [] for typ in TYPES}
        lat = {typ: [] for typ in TYPES}
        chosen = []
        for rep in range(REPEATS):
            tot = {(typ, tau): [0, 0, 0] for typ in TYPES for tau in TAUS}
            fp = {typ: 0 for typ in TYPES}
            errs = {typ: [] for typ in TYPES}
            minutes = 0
            for tr, te in GroupKFold(5, shuffle=True, random_state=rep).split(rows, groups=groups):
                train = [rows[i] for i in tr]
                test = [rows[i] for i in te]
                best = max(grid, key=lambda p: f1_of(evaluate(train, fn, p, 2, truth)))
                chosen.append(best)
                for tau in TAUS:
                    res = evaluate(test, fn, best, tau, truth)
                    for typ, (a, b, c, e) in res.items():
                        t = tot[(typ, tau)]
                        t[0] += a
                        t[1] += b
                        t[2] += c
                        if tau == 2:
                            fp[typ] += b - a
                            errs[typ] += e
                if rep == 0:
                    for r in test:
                        for time_s, typ in fn(r.md5, best):
                            detections.append(dict(detector=name, file=r.file, event=typ, time_s=round(float(time_s), 1),
                                                   parameter=best))
                minutes += sum(cruise_min[r.md5] for r in test)
            for (typ, tau), (a, b, c) in tot.items():
                p = a / b if b else 0
                r = a / c
                acc[(typ, tau)].append((p, r, 2 * p * r / (p + r) if p + r else 0))
            for typ in TYPES:
                fa[typ].append(fp[typ] / minutes)
                lat[typ].append(np.median(errs[typ]) if errs[typ] else np.nan)
        for typ in TYPES:
            row = dict(detector=name, event=typ, parameter_mode=pd.Series(chosen).mode()[0])
            for tau in TAUS:
                a = np.array(acc[(typ, tau)])
                row.update({f"precision_tau{tau}": a[:, 0].mean(), f"recall_tau{tau}": a[:, 1].mean(),
                            f"f1_tau{tau}": a[:, 2].mean(), f"f1_tau{tau}_sd": a[:, 2].std()})
            row["false_alarms_per_min"] = np.mean(fa[typ])
            row["median_abs_latency_s"] = np.nanmean(lat[typ])
            out.append(row)

    args.out.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(out)
    results.to_csv(args.out / "task5_event_results.csv", index=False)
    pd.DataFrame(detections).to_csv(args.out / "task5_detections_per_flight.csv", index=False)
    show = results[["detector", "event", "precision_tau2", "recall_tau2", "f1_tau2", "false_alarms_per_min",
                    "median_abs_latency_s"]]
    print("\n" + show.round(3).to_string(index=False))
    print(f"\nFlights {len(rows)}, events {sum(len(v) for v in truth.values())}, "
          f"cruise minutes {sum(cruise_min.values()):.1f}")
    ex = payload[payload.file == EXAMPLE]
    if len(ex):
        save_example(ex.iloc[0], events, args.out, use_cache)


def save_example(r, events, out: Path, use_cache: bool) -> None:
    """Figure: cruise signal, step statistic and detections on one flight."""

    os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    blue, orange, aqua, ink, ink2 = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
                         "grid.linewidth": 0.5, "legend.frameon": False, "pdf.fonttype": 42})
    sig = load_signals(r.md5, r.path, use_cache)
    ev = events_of(events, r.file)
    g, xs = cruise_signal(sig)
    det = Detectors({r.md5: (g, xs)})
    th, pen = 0.05, 0.5
    s = det.step[r.md5]
    fig, ax = plt.subplots(2, 1, figsize=(7.0, 3.6), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]})
    ax[0].plot(sig["mot_t"], sig["mot"], color="#b8b7b1", lw=0.8)
    ax[0].plot(g, xs, color=blue, lw=1.2)
    ax[1].plot(g, s, color=blue, lw=1.0)
    for y in (th, -th):
        ax[1].axhline(y, color=ink2, lw=0.8, ls="--")
    for a in ax:
        for name, col in [("pickup", aqua), ("drop", orange)]:
            a.axvspan(ev[name] - 1, ev[name] + 1, color=col, alpha=0.2, lw=0)
            a.axvline(ev[name], color=col, lw=1.0)
    top = ax[0].get_ylim()[1]
    for t, _ in det.pelt(r.md5, pen):
        ax[0].plot(t, 0.97 * top, marker="v", ms=8, color=ink, mec="white", mew=1.5)
    for t, _ in det.threshold(r.md5, th):
        ax[1].plot(t, 0, marker="o", ms=8, color=ink, mec="white", mew=1.5)
    ax[0].set_ylabel("Motor command (–)")
    ax[1].set_ylabel("Step statistic (–)")
    ax[1].set_xlabel("Time since arming (s)")
    for name in ("pickup", "drop"):
        ax[0].text(ev[name], 1.02, name, transform=ax[0].get_xaxis_transform(), ha="center", va="bottom", color=ink2)
    legend = [Line2D([], [], color="#b8b7b1", label="mean motor command"),
              Line2D([], [], color=blue, label="cruise-phase signal"),
              Line2D([], [], marker="v", ls="", color=ink, ms=7, label="Pelt change point"),
              Line2D([], [], marker="o", ls="", color=ink, ms=7, label="threshold detection"),
              Line2D([], [], color=ink2, ls="--", label=f"threshold ±{th}")]
    fig.legend(handles=legend, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    ax[1].set_xlim(0, sig["dur"])
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(out / "task5_events.png", dpi=200)
    fig.savefig(out / "task5_events.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
