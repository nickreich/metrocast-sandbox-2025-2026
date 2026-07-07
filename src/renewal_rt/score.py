"""Retrospective scoring: WIS + interval coverage for the renewal/Rt model vs. hub baselines.

At each biweekly reference date the renewal model is run with REAL-TIME VERSIONED data (the hub's
time-series vintages), emitting quantiles for horizons 0-3. Baseline forecasts are read from the
flu-metrocast hub's model-output. All are scored against oracle-output.csv (final truth) using the
Weighted Interval Score and central-interval coverage.

Usage (from repo root):
    PYTHONPATH=$(pwd) src/renewal_rt/.venv/bin/python src/renewal_rt/score.py
"""
import datetime
from pathlib import Path

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.renewal_rt import RenewalModelConfig, RenewalRtModel
from idmodels.config import PowerTransform, RunConfig, SourceType

SANDBOX = Path(__file__).parent.parent.parent
FLU_HUB = Path("/Users/nick/Documents/research-versioned/flu-metrocast")


def discover_baselines() -> list[str]:
    """All teams present in the local flu-metrocast hub checkout (model-output/*/)."""
    mo = FLU_HUB / "model-output"
    return sorted(d.name for d in mo.iterdir() if d.is_dir())

Q_LEVELS = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]
# central prediction intervals (nominal coverage -> (lower q, upper q))
PIS = {0.50: (0.25, 0.75), 0.80: (0.10, 0.90), 0.90: (0.05, 0.95), 0.95: (0.025, 0.975)}


def biweekly_saturdays(first="2025-11-22", last="2026-05-09"):
    d, out = datetime.date.fromisoformat(first), []
    last = datetime.date.fromisoformat(last)
    while d <= last:
        out.append(d)
        d += datetime.timedelta(days=14)
    return out


def wis_row(qvals: dict, y: float) -> float:
    """Weighted interval score for one forecast (dict quantile->value) and observation y."""
    m = qvals[0.5]
    total = 0.5 * abs(y - m)
    k = 0
    for alpha, (lo_q, hi_q) in [(0.05, (0.025, 0.975)), (0.10, (0.05, 0.95)),
                                (0.20, (0.10, 0.90)), (0.50, (0.25, 0.75))]:
        lo, hi = qvals[lo_q], qvals[hi_q]
        is_a = (hi - lo) + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)
        total += (alpha / 2) * is_a
        k += 1
    return total / (k + 0.5)


def run_renewal_quantiles(ref_dates, n_samples):
    frames = []
    all_locs = pd.read_csv(SANDBOX / "auxiliary-data/locations.csv", dtype=str)["location"].tolist()
    for ref in ref_dates:
        cfg = RenewalModelConfig(
            model_name="renewal_rt_score", sources=[SourceType.NSSP], fit_locations_separately=True,
            power_transform=PowerTransform.NONE, hub_root=SANDBOX, use_local=True, versioned=True,
            include_nyc=True, output_type="quantile", n_samples=n_samples)
        rc = RunConfig(disease=None, ref_date=ref, output_root=SANDBOX / "model-output",
                       artifact_store_root=None, max_horizon=3, states=all_locs, hsas=[],
                       q_levels=Q_LEVELS, q_labels=[str(q) for q in Q_LEVELS])
        preds = RenewalRtModel(cfg).run(rc)
        preds["model"] = "renewal_rt"
        frames.append(preds)
    df = pd.concat(frames, ignore_index=True)
    return df


def load_baselines(ref_dates, teams):
    frames = []
    for team in teams:
        for ref in ref_dates:
            f = FLU_HUB / "model-output" / team / f"{ref}-{team}.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f)
            d = d[d["output_type"] == "quantile"].copy()
            d["model"] = team
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def assemble_scores(fc: pd.DataFrame, oracle: pd.DataFrame) -> pd.DataFrame:
    fc = fc.copy()
    fc["q"] = fc["output_type_id"].astype(float)
    fc["target_end_date"] = pd.to_datetime(fc["target_end_date"]).dt.strftime("%Y-%m-%d")
    fc["reference_date"] = pd.to_datetime(fc["reference_date"]).dt.strftime("%Y-%m-%d")
    keys = ["model", "reference_date", "location", "target", "horizon", "target_end_date"]
    wide = fc.pivot_table(index=keys, columns="q", values="value").reset_index()
    wide = wide.merge(oracle, on=["location", "target", "target_end_date"], how="inner")
    qcols = Q_LEVELS
    rows = []
    for _, r in wide.iterrows():
        qv = {q: r[q] for q in qcols}
        y = r["oracle_value"]
        rec = {k: r[k] for k in keys}
        rec["y"] = y
        rec["wis"] = wis_row(qv, y)
        for nominal, (lq, uq) in PIS.items():
            rec[f"cov{int(nominal*100)}"] = int(qv[lq] <= y <= qv[uq])
        rows.append(rec)
    return pd.DataFrame(rows)


@click.command()
@click.option("--n_samples", default=200, show_default=True)
def main(n_samples):
    ref_dates = biweekly_saturdays()
    oracle = pd.read_csv(SANDBOX / "target-data/oracle-output.csv")
    oracle["target_end_date"] = pd.to_datetime(oracle["target_end_date"]).dt.strftime("%Y-%m-%d")

    teams = discover_baselines()
    click.echo(f"Baselines found in hub: {', '.join(teams)}")
    click.echo(f"Running renewal model (versioned) at {len(ref_dates)} reference dates...")
    ren = run_renewal_quantiles(ref_dates, n_samples)
    base = load_baselines([d.isoformat() for d in ref_dates], teams)
    common_cols = ["model", "reference_date", "location", "target", "horizon",
                   "target_end_date", "output_type", "output_type_id", "value"]
    allfc = pd.concat([ren[common_cols], base[common_cols]], ignore_index=True)

    scores = assemble_scores(allfc, oracle)
    scores = scores[scores["horizon"].between(0, 3)]
    scores.to_csv(SANDBOX / "src/renewal_rt/plots/scores.csv", index=False)

    # ---- headline: pairwise-common-cell mean WIS vs each baseline ----
    cell = ["reference_date", "location", "target", "horizon", "target_end_date"]
    ren_s = scores[scores.model == "renewal_rt"].set_index(cell)
    click.echo("\n=== Mean WIS on cells common to renewal_rt AND each baseline (lower is better) ===")
    print(f"{'baseline':24s} {'n_cells':>8s} {'renewal_rt':>11s} {'baseline':>10s} {'rel(ren/base)':>14s}")
    other_models = sorted(m for m in scores.model.unique() if m != "renewal_rt")
    for team in other_models:
        b = scores[scores.model == team].set_index(cell)
        idx = ren_s.index.intersection(b.index)
        if len(idx) == 0:
            print(f"{team:24s} {'0':>8s}  (no overlapping cells)")
            continue
        rw, bw = ren_s.loc[idx, "wis"].mean(), b.loc[idx, "wis"].mean()
        print(f"{team:24s} {len(idx):>8d} {rw:>11.3f} {bw:>10.3f} {rw/bw:>14.2f}")

    # ---- WIS by horizon (all cells each model covers) ----
    click.echo("\n=== Mean WIS by horizon ===")
    piv = scores.pivot_table(index="model", columns="horizon", values="wis", aggfunc="mean").round(3)
    print(piv.to_string())

    # ---- coverage ----
    click.echo("\n=== Interval coverage (nominal -> empirical) ===")
    covcols = [f"cov{int(n*100)}" for n in PIS]
    cov = scores.groupby("model")[covcols].mean().round(3)
    print(cov.to_string())

    # ---- plot: WIS by horizon + coverage (renewal_rt emphasized against the pack) ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))
    nominal = [n for n in PIS]  # [0.5, 0.8, 0.9, 0.95]

    def _emphasis(model):
        if model == "renewal_rt":
            return dict(color="black", lw=3.0, marker="o", ms=6, zorder=10)
        return dict(lw=1.3, alpha=0.65, marker="o", ms=3)

    for model in piv.index:
        ax1.plot(piv.columns, piv.loc[model], label=model, **_emphasis(model))
    ax1.set_xlabel("horizon (weeks)"); ax1.set_ylabel("mean WIS")
    ax1.set_title("WIS by horizon (lower is better)")
    ax1.set_xticks(list(piv.columns)); ax1.legend(fontsize=8, ncol=2)

    for model in cov.index:
        ax2.plot(nominal, cov.loc[model].values, label=model, **_emphasis(model))
    ax2.plot(nominal, nominal, "k:", lw=1.2, label="nominal")
    ax2.set_xticks(nominal); ax2.set_xticklabels([f"{int(n*100)}%" for n in nominal])
    ax2.set_xlabel("nominal central interval"); ax2.set_ylabel("empirical coverage")
    ax2.set_title("Interval coverage (closer to dotted = better)"); ax2.legend(fontsize=8, ncol=2)
    fig.suptitle("Renewal/Rt vs. hub baselines — 2025/26 biweekly backtest (versioned data)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = SANDBOX / "src/renewal_rt/plots/score-comparison.png"
    fig.savefig(out, dpi=200)
    click.echo(f"\nSaved {out} and scores.csv")


if __name__ == "__main__":
    main()
