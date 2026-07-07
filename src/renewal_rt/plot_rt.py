"""Diagnostic plot: estimated Rt by season-week — historical seasons vs. the current season
estimate and the model's Rt forecast.

Shows, per location, on a season-week x-axis: each past season's estimated Rt (thin grey), the
pooled Rt climatology, the current season's estimated Rt up to the reference date, and the forecast
Rt trajectory (median + 90% band).

Usage (from repo root):
    PYTHONPATH=$(pwd) src/renewal_rt/.venv/bin/python src/renewal_rt/plot_rt.py --today_date 2025-11-19
"""
import datetime
import tempfile
from pathlib import Path

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from dateutil import relativedelta

from src.renewal_rt import RenewalModelConfig, RenewalRtModel
from idmodels.config import PowerTransform, RunConfig, SourceType

C_HIST = "#BBBBBB"   # past seasons
C_CLIM = "#332288"   # pooled climatology
C_CUR = "#CC6677"    # current season estimate
C_FC = "#117733"     # forecast median
C_FAN = "#44AA99"    # forecast band

STATES = ["texas", "massachusetts", "georgia", "minnesota", "maryland", "oregon"]
SW_MIN, SW_MAX = 8, 46


@click.command()
@click.option("--today_date", type=str, default="2025-11-19", show_default=True)
@click.option("--max_horizon", type=int, default=26, show_default=True)
def main(today_date: str, max_horizon: int):
    hub_root = Path(__file__).parent.parent.parent
    today = datetime.date.fromisoformat(today_date)
    ref_date = today + relativedelta.relativedelta(weekday=5)
    states = pd.read_csv(hub_root / "auxiliary-data/locations.csv", dtype=str)
    states = states.loc[states.original_location_code == "All", "location"].tolist()

    cfg = RenewalModelConfig(
        model_name="renewal_rt_diag", sources=[SourceType.NSSP], fit_locations_separately=True,
        power_transform=PowerTransform.NONE, hub_root=hub_root, use_local=True)
    with tempfile.TemporaryDirectory() as tmp:
        rc = RunConfig(disease=None, ref_date=ref_date, output_root=Path(tmp),
                       artifact_store_root=None, max_horizon=max_horizon, states=states, hsas=[],
                       q_levels=[], q_labels=[])
        model = RenewalRtModel(cfg)
        model.run(rc)

    hist = model.rt_history_
    clim = model.rt_clim_.sort_values("season_week")
    fc = model.rt_forecast_
    # current season = the most recent season present in the data (contains the reference date)
    cur_season = hist.assign(d=pd.to_datetime(hist["wk_end_date"])) \
                     .sort_values("d")["season"].iloc[-1]

    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True, sharey=True)
    for ax, loc in zip(axes.ravel(), STATES):
        h = hist[(hist.location == loc) & hist.season_week.between(SW_MIN, SW_MAX)]
        # past seasons (thin grey)
        first = True
        for season, g in h[h.season != cur_season].groupby("season"):
            g = g.sort_values("season_week")
            ax.plot(g.season_week, g.R, color=C_HIST, lw=1,
                    label="past seasons" if first else None)
            first = False
        # pooled climatology
        c = clim[clim.season_week.between(SW_MIN, SW_MAX)]
        ax.plot(c.season_week, c.R_clim, color=C_CLIM, lw=2.2, ls="--", label="pooled climatology")
        # current season estimate
        cs = h[h.season == cur_season].sort_values("season_week")
        ax.plot(cs.season_week, cs.R, color=C_CUR, lw=2, marker="o", ms=3,
                label=f"current ({cur_season})")
        # forecast Rt (median + 90% band)
        f = fc[(fc.location == loc) & fc.season_week.between(SW_MIN, SW_MAX)].sort_values("season_week")
        ax.fill_between(f.season_week, f.R_lo, f.R_hi, color=C_FAN, alpha=0.3, lw=0, label="forecast 90%")
        ax.plot(f.season_week, f.R_med, color=C_FC, lw=2, label="forecast median")

        ax.axhline(1.0, color="0.5", lw=0.8, ls=":")
        ax.set_title(loc, loc="left", fontsize=11)
        ax.set_xlim(SW_MIN, SW_MAX)
        ax.set_ylim(0.4, 2.2)
        ax.set_ylabel("estimated Rt")
    for ax in axes[-1]:
        ax.set_xlabel("season week (MMWR week 30 = 1)")
    axes.ravel()[0].legend(fontsize=8, framealpha=0.9, ncol=1, loc="upper right")
    fig.suptitle(f"Estimated Rt by season week — history, current season, and forecast "
                 f"(reference {ref_date})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = Path(__file__).parent / "plots" / f"rt-by-season-week-{ref_date}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
