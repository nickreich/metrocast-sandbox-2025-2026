"""Generate flu-metrocast renewal/Rt sample-trajectory forecasts (state level, v1).

Usage (from the repo root):
    PYTHONPATH=$(pwd) src/renewal_rt/.venv/bin/python src/renewal_rt/main.py \
        --today_date 2026-01-28 --use_local_mchub

Writes model-output/UMass-renewal_rt/{ref_date}-UMass-renewal_rt.csv
"""

import datetime
from pathlib import Path

import click
import pandas as pd
from dateutil import relativedelta

# Importing the package installs the idmodels dependency stubs (see _stubs.py) before
# any idmodels import below.
from src.renewal_rt import RenewalModelConfig, RenewalRtModel

from idmodels.config import PowerTransform, RunConfig, SourceType


def _state_locations(hub_root: Path) -> list[str]:
    loc = pd.read_csv(hub_root / "auxiliary-data" / "locations.csv", dtype=str)
    return loc.loc[loc["original_location_code"] == "All", "location"].tolist()


@click.command()
@click.option("--today_date", type=str, required=True,
              help="Effective model run date (YYYY-MM-DD); reference date is the next Saturday.")
@click.option("--max_horizon", type=int, default=26, show_default=True,
              help="Number of weeks ahead (full-season trajectories).")
@click.option("--n_samples", type=int, default=100, show_default=True,
              help="Number of trajectory samples per location.")
@click.option("--output_type", type=click.Choice(["sample", "quantile"]), default="sample",
              show_default=True, help="Emit 100 trajectory samples, or hub-comparable quantiles.")
@click.option("--transform", type=click.Choice(["log", "logit"]), default="log",
              show_default=True, help="Working scale (default log).")
@click.option("--level_revert_kappa", type=float, default=0.15, show_default=True,
              help="Weekly OU reversion toward the seasonal-average level; lower = slower reversion "
                   "(half-life ~2.4 wk at 0.25, ~4.3 wk at 0.15, ~6.6 wk at 0.10).")
@click.option("--model_name", type=str, default="renewal_rt", show_default=True,
              help="Model name (drives output dir/filename UMass-<model_name>).")
@click.option("--use_local_mchub", is_flag=True,
              help="Use local target-data/latest-data.csv instead of downloading from GitHub.")
def main(today_date: str, max_horizon: int, n_samples: int, output_type: str,
         transform: str, level_revert_kappa: float, model_name: str, use_local_mchub: bool):
    """Run the renewal/Rt model for all state-level metrocast locations."""
    try:
        today = datetime.date.fromisoformat(today_date)
    except (TypeError, ValueError):
        today = datetime.date.today()
    ref_date = today + relativedelta.relativedelta(weekday=5)  # next Saturday

    hub_root = Path(__file__).parent.parent.parent
    states = _state_locations(hub_root)

    model_config = RenewalModelConfig(
        model_name=model_name,
        sources=[SourceType.NSSP],
        fit_locations_separately=True,
        power_transform=PowerTransform.NONE,
        hub_root=hub_root,
        use_local=use_local_mchub,
        target="Flu ED visits pct",
        transform=transform,
        level_revert_kappa=level_revert_kappa,
        output_type=output_type,
        n_samples=n_samples,
    )
    run_config = RunConfig(
        disease=None,
        ref_date=ref_date,
        output_root=hub_root / "model-output",
        artifact_store_root=None,
        max_horizon=max_horizon,
        states=states,
        hsas=[],
        q_levels=[],
        q_labels=[],
    )

    click.echo(f"Running renewal/Rt model: {model_name}")
    click.echo(f"  Reference date: {ref_date}")
    click.echo(f"  Locations: {len(states)} states")
    click.echo(f"  Max horizon: {max_horizon} weeks   Output: {output_type}"
               + (f" ({n_samples} samples)" if output_type == "sample" else "")
               + f"   Transform: {transform}")
    click.echo()

    model = RenewalRtModel(model_config)
    preds = model.run(run_config)
    click.echo(f"\nGenerated forecasts for {preds['location'].nunique()} locations, "
               f"{preds['horizon'].nunique()} horizons.")


if __name__ == "__main__":
    main()
