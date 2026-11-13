#!/usr/bin/env python3
"""Task 2: payload state and flight-level scenario recognition.

Payload state (per window): is a payload attached? Labels follow from the
scenario and the event times in metadata/events.csv; windows within 1 s of an
event are excluded. Features: IMU statistics plus mean motor command and
battery voltage. Stratified grouped 5-fold CV by flight (identical files share
a group), repeated 10 times; metadata-only, IMU-only and cross-session
(train trial 1, test trials 2-3) controls. Reproduces Table 8.

Flight-level scenario (per flight): which of the five scenarios was flown?
Each flight is summarised by the mean and std of its 5 s window features and
the mean motor command in the first, middle and last third.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedGroupKFold

from _common import REPEATS, classifiers, fmt, per_flight
from dronelog.features import load_metadata, meta_features, payload_table, payload_windows
from dronelog.io import print_startup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 2: payload state and flight-level scenario recognition.")
    parser.add_argument("--data", default=".", type=Path, help="Dataset root (contains log/ and metadata/)")
    parser.add_argument("--out", default=Path("outputs"), type=Path, help="Output directory. Default: outputs/")
    parser.add_argument("--no-cache", action="store_true", help="Re-parse the ULog files instead of using .cache")
    return parser.parse_args()


def scores(y, p):
    return [accuracy_score(y, p), f1_score(y, p, average="macro"), precision_score(y, p, average="macro", zero_division=0),
            recall_score(y, p, average="macro"), balanced_accuracy_score(y, p)]


def cv(X, y, g, make, keep_pred=False):
    res, first = [], None
    for rep in range(REPEATS):
        p = np.zeros(len(y))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(X, y, g):
            m = make(rep)
            m.fit(X[tr], y[tr])
            p[te] = m.predict(X[te])
        res.append(scores(y, p))
        if keep_pred and rep == 0:
            first = p.copy()
    return np.array(res), first


def state_row(window, model, r, y, g, **extra):
    return dict(window=window, model=model, acc=r[:, 0].mean(), acc_sd=r[:, 0].std(), f1=r[:, 1].mean(),
                f1_sd=r[:, 1].std(), prec=r[:, 2].mean(), rec=r[:, 3].mean(), bacc=r[:, 4].mean(),
                windows=len(y), groups=len(set(g)), **extra)


def payload_state(payload, events, use_cache):
    models = classifiers(impute=True)
    out, flights_df = [], None
    for w in [2, 5, 10]:
        D = payload_windows(payload, events, w, use_cache=use_cache)
        X, y, g = np.stack(D.f.values), D.state.values, D.md5.values
        for name, make in models.items():
            r, pred = cv(X, y, g, make, keep_pred=(w == 5 and name == "Random Forest"))
            out.append(state_row(w, name, r, y, g, loaded_frac=y.mean()))
            print(f"state {w:>2} s  {name:<20} acc {fmt(r[:, 0].mean(), r[:, 0].std())}", flush=True)
            if pred is not None:
                flights_df = per_flight(D.file.values, y, pred)
        if w == 5:
            Xm = np.array(D.meta.tolist())
            r, _ = cv(Xm, y, g, models["Random Forest"])
            out.append(state_row(5, "Metadata only (RF)", r, y, g))
            r, _ = cv(X[:, :48], y, g, models["Random Forest"])
            out.append(state_row(5, "RF, IMU only", r, y, g))
            tr = D.trial.values == 1
            te = ~tr
            for name, make in models.items():
                m = make(0)
                m.fit(X[tr], y[tr])
                s = scores(y[te], m.predict(X[te]))
                out.append(dict(window=5, model=f"{name}, train t1 -> test t2/t3", acc=s[0], f1=s[1], prec=s[2],
                                rec=s[3], bacc=s[4], windows=int(te.sum()), groups=len(set(g[te]))))
            for pl in ["x500v2", "x650"]:
                k = D.platform.values == pl
                r, _ = cv(X[k], y[k], g[k], models["Random Forest"])
                out.append(state_row(5, f"Random Forest, {pl} only", r, y[k], g[k]))
    return pd.DataFrame(out), flights_df


def flight_vector(X):
    n = len(X)
    q = max(1, n // 3)
    mot = X[:, 48]  # mean motor command per window
    return np.concatenate([X.mean(0), X.std(0), [
        np.nanmean(mot[:q]), np.nanmean(mot[q:2 * q]), np.nanmean(mot[2 * q:]),
        np.nanmean(mot[2 * q:]) - np.nanmean(mot[:q]), np.nanmax(np.abs(np.diff(mot)))]])


def flight_scenario(payload, events, use_cache):
    D = payload_windows(payload, events, 5, margin=0, use_cache=use_cache)
    rows = []
    for f, df in D.groupby("file", sort=False):
        df = df.sort_values("t")
        r = payload[payload.file == f].iloc[0]
        rows.append(dict(file=f, md5=r.md5, scenario=int(r.scenario), trial=int(r.trial),
                         x=flight_vector(np.stack(df.f.values)), meta=meta_features(r)))
    G = pd.DataFrame(rows)
    X, y, g, XM = np.stack(G.x.values), G.scenario.values, G.md5.values, np.array(G.meta.tolist())
    models = classifiers(impute=True)
    out, cm, preds = [], None, None

    def run(Xin, name, make, keep=False):
        nonlocal cm, preds
        res = []
        for rep in range(REPEATS):
            p = np.zeros(len(y), int)
            for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(Xin, y, g):
                m = make(rep)
                m.fit(Xin[tr], y[tr])
                p[te] = m.predict(Xin[te])
            res.append([accuracy_score(y, p), f1_score(y, p, average="macro")])
            if keep and rep == 0:
                cm = confusion_matrix(y, p, labels=[1, 2, 3, 4, 5])
                preds = p.copy()
        res = np.array(res)
        out.append(dict(model=name, acc=res[:, 0].mean(), acc_sd=res[:, 0].std(), f1=res[:, 1].mean(),
                        f1_sd=res[:, 1].std(), flights=len(y)))
        print(f"scenario  {name:<20} acc {fmt(res[:, 0].mean(), res[:, 0].std())}", flush=True)

    for name, make in models.items():
        run(X, name, make, keep=(name == "Random Forest"))
    run(XM, "Metadata only (RF)", models["Random Forest"])
    tr = G.trial.values == 1
    m = models["Random Forest"](0)
    m.fit(X[tr], y[tr])
    p = m.predict(X[~tr])
    out.append(dict(model="Random Forest, train t1 -> test t2/t3", acc=accuracy_score(y[~tr], p),
                    f1=f1_score(y[~tr], p, average="macro"), flights=int((~tr).sum())))
    cm_df = pd.DataFrame(cm, index=[f"true_S{i}" for i in range(1, 6)], columns=[f"pred_S{i}" for i in range(1, 6)])
    flights_df = pd.DataFrame(dict(file=G.file, true_scenario=y, predicted_scenario=preds))
    return pd.DataFrame(out), cm_df, flights_df


def main() -> None:
    warnings.filterwarnings("ignore")
    args = parse_args()
    print_startup("Task 2 - Payload state and flight-level scenario")
    flights, events = load_metadata(args.data)
    payload = payload_table(flights, events)
    use_cache = not args.no_cache
    args.out.mkdir(parents=True, exist_ok=True)

    state, state_flights = payload_state(payload, events, use_cache)
    state.to_csv(args.out / "task2_payload_results.csv", index=False)
    state_flights.to_csv(args.out / "task2_payload_per_flight.csv", index=False)

    scen, cm, scen_flights = flight_scenario(payload, events, use_cache)
    scen.to_csv(args.out / "task2_scenario_results.csv", index=False)
    cm.to_csv(args.out / "task2_scenario_confusion.csv")
    scen_flights.to_csv(args.out / "task2_scenario_per_flight.csv", index=False)

    print("\nPayload state\n" + state.round(4).to_string(index=False))
    print("\nFlight-level scenario\n" + scen.round(4).to_string(index=False))
    print("\nScenario confusion (Random Forest, repetition 0)\n" + cm.to_string())


if __name__ == "__main__":
    main()
