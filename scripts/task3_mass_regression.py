#!/usr/bin/env python3
"""Task 3: carried payload mass estimation, per platform.

Each 5 s window is labelled with the mass actually carried at that time (0 kg
before pickup and after drop); windows within 1 s of an event are excluded.
Features: IMU statistics plus mean motor command and battery voltage.
Random Forest and Ridge with grouped 5-fold CV by flight, repeated 10 times,
against a mean predictor and a metadata-only control. Also reports the error
per carried mass, held-out (unseen) masses and the cross-session split.
Reproduces Tables 9 and 10 and the mass-scatter figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from _common import REPEATS, fmt
from dronelog.features import load_metadata, payload_table, payload_windows
from dronelog.io import print_startup

PLATFORMS = ["x500v2", "x650"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 3: payload mass estimation.")
    parser.add_argument("--data", default=".", type=Path, help="Dataset root (contains log/ and metadata/)")
    parser.add_argument("--out", default=Path("outputs"), type=Path, help="Output directory. Default: outputs/")
    parser.add_argument("--no-cache", action="store_true", help="Re-parse the ULog files instead of using .cache")
    return parser.parse_args()


def regressors() -> dict:
    return {
        "Random Forest": lambda r: make_pipeline(SimpleImputer(), RandomForestRegressor(random_state=r, n_jobs=2)),
        "Ridge": lambda r: make_pipeline(SimpleImputer(), StandardScaler(), Ridge()),
    }


def grouped_cv(X, y, g, make):
    """Repeated grouped 5-fold CV; returns per-repetition [MAE, RMSE, R2], per-mass MAE and rep-0 predictions."""

    res, permass, first = [], [], None
    for rep in range(REPEATS):
        p = np.zeros(len(y))
        for tr, te in GroupKFold(5, shuffle=True, random_state=rep).split(X, y, g):
            m = make(rep)
            m.fit(X[tr], y[tr])
            p[te] = m.predict(X[te])
        res.append([mean_absolute_error(y, p), np.sqrt(mean_squared_error(y, p)), r2_score(y, p)])
        permass.append({c: mean_absolute_error(y[y == c], p[y == c]) for c in sorted(set(y))})
        if rep == 0:
            first = p.copy()
    return np.array(res), pd.DataFrame(permass).mean(), first


def scatter(ax, y, p, title):
    import matplotlib.pyplot as plt  # noqa: F401
    blue, ink, ink2 = "#2a78d6", "#0b0b0b", "#52514e"
    jit = np.random.default_rng(0).uniform(-0.06, 0.06, len(y))
    ax.scatter(y + jit, p, s=5, color=blue, alpha=0.25, lw=0)
    for c in sorted(set(y)):
        q = np.percentile(p[y == c], [25, 50, 75])
        ax.plot([c, c], [q[0], q[2]], color=ink, lw=2)
        ax.plot(c, q[1], "o", color="white", mec=ink, mew=1.5, ms=6)
    lim = max(y.max(), p.max()) + 0.15
    ax.plot([-0.1, lim], [-0.1, lim], color=ink2, ls="--", lw=0.8)
    ax.set_xlim(-0.15, lim)
    ax.set_ylim(-0.15, lim)
    ax.set_aspect("equal")
    ax.set_xticks(sorted(set(y)))
    ax.set_xlabel("True carried mass (kg)")
    ax.set_ylabel("Predicted mass (kg)")
    ax.set_title(title, loc="left", fontsize=8)


def save_figures(preds: dict, out: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
                         "grid.linewidth": 0.5, "legend.frameon": False, "pdf.fonttype": 42})
    legend = [Line2D([], [], marker="o", ls="", color="#2a78d6", alpha=0.5, ms=4, label="5 s window (held-out)"),
              Line2D([], [], marker="o", ls="", color="white", mec="#0b0b0b", ms=6, label="median"),
              Line2D([], [], color="#0b0b0b", lw=2, label="interquartile range"),
              Line2D([], [], color="#52514e", ls="--", label="y = x")]
    titles = {"x500v2": "(a) X500 V2", "x650": "(b) X650"}
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, pl in zip(axs, PLATFORMS):
        scatter(ax, *preds[pl], titles[pl])
    fig.legend(handles=legend, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "task3_mass_scatter.png", dpi=200)
    fig.savefig(out / "task3_mass_scatter.pdf")
    plt.close(fig)
    for pl in PLATFORMS:
        fig, ax = plt.subplots(figsize=(3.5, 3.4))
        scatter(ax, *preds[pl], titles[pl][4:])
        fig.tight_layout()
        fig.savefig(out / f"task3_mass_scatter_{pl}.png", dpi=200)
        plt.close(fig)


def main() -> None:
    warnings.filterwarnings("ignore")
    args = parse_args()
    print_startup("Task 3 - Payload mass estimation")
    flights, events = load_metadata(args.data)
    payload = payload_table(flights, events)
    W = payload_windows(payload, events, 5, use_cache=not args.no_cache)
    out, unseen, figure_preds = [], [], {}

    for pl in PLATFORMS:
        d = W[W.platform == pl].reset_index(drop=True)
        X, y, g = np.stack(d.f.values), d.carried.values, d.md5.values
        masses = sorted(d.mass.unique())
        for name, make in regressors().items():
            res, pm, first = grouped_cv(X, y, g, make)
            out.append(dict(platform=pl, model=name, mae=res[:, 0].mean(), mae_sd=res[:, 0].std(),
                            rmse=res[:, 1].mean(), rmse_sd=res[:, 1].std(), r2=res[:, 2].mean(),
                            r2_sd=res[:, 2].std(), windows=len(y), groups=len(set(g)),
                            **{f"mae_{k:g}kg": v for k, v in pm.items()}))
            print(f"{pl:<7} {name:<14} MAE {fmt(res[:, 0].mean(), res[:, 0].std())}  "
                  f"R2 {fmt(res[:, 2].mean(), res[:, 2].std())}", flush=True)
            if name == "Random Forest":
                figure_preds[pl] = (y, first)
        mean_pred = np.full_like(y, y.mean())
        out.append(dict(platform=pl, model="Mean predictor", mae=mean_absolute_error(y, mean_pred),
                        rmse=np.sqrt(mean_squared_error(y, mean_pred)), r2=0.0, windows=len(y), groups=len(set(g))))
        res, _, _ = grouped_cv(np.array(d.meta.tolist()), y, g, regressors()["Random Forest"])
        out.append(dict(platform=pl, model="Metadata only (RF)", mae=res[:, 0].mean(), mae_sd=res[:, 0].std(),
                        rmse=res[:, 1].mean(), r2=res[:, 2].mean(), windows=len(y), groups=len(set(g))))
        tr = d.trial.values == 1
        m = regressors()["Random Forest"](0)
        m.fit(X[tr], y[tr])
        p = m.predict(X[~tr])
        out.append(dict(platform=pl, model="Random Forest, train t1 -> test t2/t3", mae=mean_absolute_error(y[~tr], p),
                        rmse=np.sqrt(mean_squared_error(y[~tr], p)), r2=r2_score(y[~tr], p),
                        windows=int((~tr).sum())))
        # unseen interior mass: every flight of that mass condition removed from training
        for um in masses[1:-1]:
            te = d.mass.values == um
            m = regressors()["Random Forest"](0)
            m.fit(X[~te], y[~te])
            p = m.predict(X[te])
            loaded = y[te] > 0
            unseen.append(dict(platform=pl, held_out_mass_kg=um, mae_loaded=mean_absolute_error(y[te][loaded], p[loaded]),
                               mean_prediction_loaded=float(p[loaded].mean()),
                               mae_unloaded=mean_absolute_error(y[te][~loaded], p[~loaded]), windows=int(te.sum())))

    args.out.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(out)
    results.to_csv(args.out / "task3_mass_results.csv", index=False)
    for pl in PLATFORMS:
        results[results.platform == pl].to_csv(args.out / f"task3_mass_results_{pl}.csv", index=False)
    pd.DataFrame(unseen).to_csv(args.out / "task3_mass_unseen.csv", index=False)
    save_figures(figure_preds, args.out)
    print("\n" + results.round(3).to_string(index=False))
    print("\nUnseen masses\n" + pd.DataFrame(unseen).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
