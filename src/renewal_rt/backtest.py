"""Retrospective backtest: run the renewal/Rt model for ALL metrocast locations at biweekly
reference dates across a season, saving a forecast CSV and an all-location trajectory plot per date.

Uses the (revised) latest target data filtered to each reference date (not true real-time vintages).

Usage (from repo root):
    PYTHONPATH=$(pwd) src/renewal_rt/.venv/bin/python src/renewal_rt/backtest.py \
        --first_ref 2025-11-08 --last_ref 2026-05-16 --season_end 2026-06-27
"""
import datetime
import math
from pathlib import Path

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.renewal_rt import RenewalModelConfig, RenewalRtModel
from idmodels.config import PowerTransform, RunConfig, SourceType

C_OBS, C_MED, C_FAN = "#332288", "#117733", "#44AA99"


def _saturdays(first: datetime.date, last: datetime.date, step_days: int):
    d, out = first, []
    while d <= last:
        out.append(d)
        d += datetime.timedelta(days=step_days)
    return out


def _plot_all_locations(preds, obs, ref_date, loc_order, out_path):
    ncol = 7
    nrow = math.ceil(len(loc_order) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.6, nrow * 2.5), squeeze=False)
    ref_ts = pd.Timestamp(ref_date)
    for ax in axes.ravel():
        ax.axis("off")
    for i, loc in enumerate(loc_order):
        ax = axes[i // ncol][i % ncol]
        ax.axis("on")
        f = preds[preds.location == loc]
        if not f.empty:
            gq = f.groupby("target_end_date")["value"]
            d = pd.to_datetime(gq.median().index)
            ax.fill_between(d, gq.quantile(.05), gq.quantile(.95), color=C_FAN, alpha=0.25, lw=0)
            ax.fill_between(d, gq.quantile(.25), gq.quantile(.75), color=C_FAN, alpha=0.45, lw=0)
            ax.plot(d, gq.median().values, color=C_MED, lw=1.4)
        o = obs[obs.location == loc].sort_values("target_end_date")
        o = o[(o.target_end_date >= ref_ts - pd.Timedelta(weeks=16))]
        ax.plot(o.target_end_date, o.observation, color=C_OBS, lw=1.0, marker="o", ms=1.5)
        ax.axvline(ref_ts, color="0.6", ls="--", lw=0.8)
        ax.set_title(loc, fontsize=10, loc="left")
        ax.tick_params(labelsize=7)
        for lab in ax.get_xticklabels():
            lab.set_rotation(45)
    fig.suptitle(f"Renewal/Rt all-location forecasts — reference date {ref_date}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@click.command()
@click.option("--first_ref", default="2025-11-08", show_default=True, help="First reference Saturday.")
@click.option("--last_ref", default="2026-05-16", show_default=True, help="Last reference Saturday.")
@click.option("--season_end", default="2026-06-27", show_default=True,
              help="Season-end target date; forecasts run out to here at every reference date.")
@click.option("--cadence_weeks", default=2, show_default=True)
@click.option("--n_samples", default=100, show_default=True)
def main(first_ref, last_ref, season_end, cadence_weeks, n_samples):
    hub_root = Path(__file__).parent.parent.parent
    all_locs = pd.read_csv(hub_root / "auxiliary-data/locations.csv", dtype=str)
    loc_order = all_locs.sort_values(["state", "location"])["location"].tolist()
    locs = all_locs["location"].tolist()

    obs = pd.read_csv(hub_root / "target-data/latest-data.csv")
    obs = obs[obs.target.isin(["Flu ED visits pct", "ILI ED visits pct"])].copy()
    obs["target_end_date"] = pd.to_datetime(obs["target_end_date"])

    ref_dates = _saturdays(datetime.date.fromisoformat(first_ref),
                           datetime.date.fromisoformat(last_ref), cadence_weeks * 7)
    end = datetime.date.fromisoformat(season_end)
    plot_dir = Path(__file__).parent / "plots" / "backtest"
    plot_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Backtest: {len(ref_dates)} reference dates, {len(locs)} locations, "
               f"forecasting through {end}.")
    for ref in ref_dates:
        max_h = max(1, (end - ref).days // 7)
        cfg = RenewalModelConfig(
            model_name="renewal_rt", sources=[SourceType.NSSP], fit_locations_separately=True,
            power_transform=PowerTransform.NONE, hub_root=hub_root, use_local=True,
            include_nyc=True, output_type="sample", n_samples=n_samples)
        rc = RunConfig(disease=None, ref_date=ref, output_root=hub_root / "model-output",
                       artifact_store_root=None, max_horizon=max_h, states=locs, hsas=[],
                       q_levels=[], q_labels=[])
        preds = RenewalRtModel(cfg).run(rc)
        preds["target_end_date"] = pd.to_datetime(preds["target_end_date"])
        out_path = plot_dir / f"traj-{ref}.png"
        _plot_all_locations(preds, obs, ref, loc_order, out_path)
        click.echo(f"  {ref}: {preds.location.nunique()} locations, horizon {max_h} -> {out_path.name}")


if __name__ == "__main__":
    main()
