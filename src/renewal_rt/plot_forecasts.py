"""Diagnostic plot: full-season renewal/Rt sample trajectories vs. observed history.

Usage (from repo root):
    PYTHONPATH=$(pwd) src/renewal_rt/.venv/bin/python src/renewal_rt/plot_forecasts.py \
        --forecast model-output/UMass-renewal_rt/2025-11-22-UMass-renewal_rt.csv
"""
from pathlib import Path

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Tol muted palette (color-blind safe, neutral)
C_OBS = "#332288"      # indigo - observations
C_MED = "#117733"      # green - forecast median
C_FAN = "#44AA99"      # teal - intervals
C_SAMP = "#999933"     # olive - individual sample paths

STATES = ["texas", "massachusetts", "georgia", "minnesota", "maryland", "oregon"]


@click.command()
@click.option("--forecast", type=str, required=True, help="Path to a forecast CSV (relative to repo root).")
def main(forecast: str):
    hub_root = Path(__file__).parent.parent.parent
    fc = pd.read_csv(hub_root / forecast)
    fc["target_end_date"] = pd.to_datetime(fc["target_end_date"])
    ref_date = pd.to_datetime(fc["reference_date"].iloc[0])

    obs = pd.read_csv(hub_root / "target-data" / "latest-data.csv")
    obs = obs[obs["target"] == "Flu ED visits pct"].copy()
    obs["target_end_date"] = pd.to_datetime(obs["target_end_date"])

    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    for ax, loc in zip(axes.ravel(), STATES):
        f = fc[fc["location"] == loc]
        # fan from samples
        gq = f.groupby("target_end_date")["value"]
        med = gq.median()
        lo50, hi50 = gq.quantile(0.25), gq.quantile(0.75)
        lo90, hi90 = gq.quantile(0.05), gq.quantile(0.95)
        d = med.index

        ax.fill_between(d, lo90, hi90, color=C_FAN, alpha=0.25, lw=0, label="90% PI")
        ax.fill_between(d, lo50, hi50, color=C_FAN, alpha=0.45, lw=0, label="50% PI")
        # a few sample trajectories to show coherence
        for s in [1, 25, 50, 75, 100]:
            traj = f[f["output_type_id"] == s].sort_values("target_end_date")
            ax.plot(traj["target_end_date"], traj["value"], color=C_SAMP, lw=0.6, alpha=0.6)
        ax.plot(d, med, color=C_MED, lw=2, label="median")

        # observed: history + realized truth after ref_date (hindcast check)
        o = obs[(obs["location"] == loc)].sort_values("target_end_date")
        o = o[o["target_end_date"] >= ref_date - pd.Timedelta(weeks=20)]
        ax.plot(o["target_end_date"], o["observation"], color=C_OBS, lw=1.4, marker="o",
                ms=2.5, label="observed")
        ax.axvline(ref_date, color="0.5", ls="--", lw=1)
        ax.set_title(loc, loc="left", fontsize=11)
        ax.set_ylabel("% ED visits (flu)")
        ax.margins(x=0.01)

    axes.ravel()[0].legend(fontsize=8, framealpha=0.9, loc="upper right")
    fig.suptitle(f"Renewal/Rt full-season sample trajectories — reference date {ref_date.date()}",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = Path(__file__).parent / "plots"
    out.mkdir(exist_ok=True)
    path = out / f"trajectories-{ref_date.date()}.png"
    fig.savefig(path, dpi=200)
    print(f"Saved {path}")

    # explosion / turnover diagnostics
    print("\nfull-sample value summary:\n", fc["value"].describe().round(2).to_string())
    print("\nper-location max sample value:")
    print(fc.groupby("location")["value"].max().round(1).sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
