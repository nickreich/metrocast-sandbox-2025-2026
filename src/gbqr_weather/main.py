"""GBQR model using MCHub + Weather data.

This model uses:
- MCHub target data (primary source)
- Weather data features (temperature, humidity, precipitation)
  - Raw values
  - 1-week and 2-week lags
  - 3-week trailing rolling average
"""

import datetime
from pathlib import Path

import click
import numpy as np
import pandas as pd
from dateutil import relativedelta

from src.mchub_gbqr import ModelConfig, RunConfig, GBQRModel


@click.command()
@click.option(
    "--today_date",
    type=str,
    required=True,
    help="Date to use as effective model run date (YYYY-MM-DD)"
)
@click.option(
    "--short_run",
    is_flag=True,
    help="Perform a short run with reduced bagging (10 bags)"
)
@click.option(
    "--use_local_mchub",
    is_flag=True,
    help="Use local MCHub target data instead of downloading from GitHub"
)
@click.option(
    "--use_versioned_mchub",
    is_flag=True,
    help="Fetch MCHub data as of reference date using GitHub API versioning"
)
@click.option(
    "--analyze_importance",
    is_flag=True,
    default=True,
    help="Run variable importance analysis"
)
def main(today_date: str, short_run: bool, use_local_mchub: bool,
         use_versioned_mchub: bool, analyze_importance: bool):
    """Generate GBQR forecasts using MCHub + Weather data."""
    # Parse date and compute reference date (next Saturday from today_date)
    try:
        today_date = datetime.date.fromisoformat(today_date)
    except (TypeError, ValueError):
        today_date = datetime.date.today()
    reference_date = today_date + relativedelta.relativedelta(weekday=5)

    # Model configuration - MCHub + Weather features
    model_config = ModelConfig(
        model_name="gbqr_weather",

        # No supplementary incidence sources
        use_ilinet=False,
        use_flusurvnet=False,
        use_nhsn=False,
        use_nssp_extra=False,

        # Weather data configuration
        use_weather=True,
        weather_features=[
            "temp_avg_f", "temp_max_f", "temp_min_f",
            "humidity_avg_pct", "humidity_abs_gm3", "precip_total_mm"
        ],
        weather_lags=[1, 2],  # 1-week and 2-week lags
        weather_rolling_windows=[3],  # 3-week trailing average

        # Bagging parameters
        num_bags=100,
        bag_frac_samples=0.7,

        # Feature configuration
        incl_level_feats=True,
        power_transform="4rt",
        fit_locations_separately=False,

        # Drop seasons with data quality issues or anomalous patterns
        drop_seasons=["2020/21", "2021/22"]
    )

    # Run configuration
    hub_root = Path(__file__).parent.parent.parent
    run_config = RunConfig(
        ref_date=reference_date,
        hub_root=hub_root,
        output_root=hub_root / "model-output",
        max_horizon=4,
        q_levels=[0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975],
        q_labels=['0.025', '0.05', '0.1', '0.25', '0.5', '0.75', '0.9', '0.95', '0.975']
    )

    if short_run:
        model_config.num_bags = 10

    # Print configuration
    click.echo("Running GBQR model: gbqr_weather")
    click.echo(f"  Reference date: {reference_date}")
    click.echo(f"  Num bags: {model_config.num_bags}")
    click.echo(f"  Data sources: MCHub + Weather")
    click.echo(f"  Weather features: {model_config.weather_features}")
    click.echo(f"  Weather lags: {model_config.weather_lags}")
    click.echo(f"  Weather rolling windows: {model_config.weather_rolling_windows}")
    click.echo()

    # Run model
    model = GBQRModel(model_config)
    preds_df = model.run(run_config, use_local_mchub=use_local_mchub, use_versioned_mchub=use_versioned_mchub)

    click.echo(f"\nGenerated {len(preds_df)} predictions for {preds_df['location'].nunique()} locations")

    # Variable importance analysis
    if analyze_importance:
        click.echo("\n" + "=" * 60)
        click.echo("Variable Importance Analysis")
        click.echo("=" * 60)
        run_importance_analysis(model_config, run_config, use_local_mchub, use_versioned_mchub)


def run_importance_analysis(model_config: ModelConfig, run_config: RunConfig,
                            use_local_mchub: bool, use_versioned_mchub: bool):
    """Run variable importance analysis using LightGBM feature importance."""
    import lightgbm as lgb
    from src.mchub_gbqr.data_loader import (
        load_all_data, load_location_crosswalk, load_weather_with_lags
    )
    from src.mchub_gbqr.hsa_populations import load_mchub_populations
    from src.mchub_gbqr.transforms import apply_scale_center_transform
    from idmodels.preprocess import create_features_and_targets

    # Load data (same as model)
    df = load_all_data(
        model_config=model_config,
        run_config=run_config,
        use_local_mchub=use_local_mchub,
        use_versioned_mchub=use_versioned_mchub
    )
    df = df[df["wk_end_date"] <= pd.Timestamp(run_config.ref_date)]

    locations_df = load_location_crosswalk(run_config.hub_root)
    df["agg_level"] = df["geo_type"]

    mchub_populations = load_mchub_populations(run_config.hub_root)
    df["pop"] = df["location"].map(mchub_populations)
    df["log_pop"] = np.log(df["pop"])

    df = apply_scale_center_transform(
        df,
        power_transform=model_config.power_transform,
        group_cols=["source", "location"]
    )

    # Load weather features
    weather_feat_names = []
    if model_config.use_weather:
        try:
            weather_df = load_weather_with_lags(
                weather_file=model_config.weather_file,
                features=model_config.weather_features,
                hub_root=run_config.hub_root,
                lags=model_config.weather_lags,
                rolling_windows=model_config.weather_rolling_windows
            )
            weather_feat_cols = [c for c in weather_df.columns if c not in ["location", "wk_end_date"]]
            df = df.merge(
                weather_df[["location", "wk_end_date"] + weather_feat_cols],
                on=["location", "wk_end_date"],
                how="left"
            )
            for col in weather_feat_cols:
                col_mean = df[col].mean()
                col_std = df[col].std()
                if col_std > 0:
                    df[col] = (df[col] - col_mean) / col_std
            weather_feat_names = weather_feat_cols
        except FileNotFoundError:
            pass

    init_feats = ["inc_trans_cs", "season_week", "log_pop"] + weather_feat_names
    df, feat_names = create_features_and_targets(
        df=df,
        incl_level_feats=model_config.incl_level_feats,
        max_horizon=run_config.max_horizon,
        curr_feat_names=init_feats
    )

    df = df.query("season_week >= 5 and season_week <= 45")
    df_train = df.loc[~df["delta_target"].isna()]

    x_train = df_train[feat_names]
    y_train = df_train["delta_target"]

    # Train a single model for importance analysis
    model = lgb.LGBMRegressor(
        verbosity=-1,
        objective="quantile",
        alpha=0.5,
        random_state=42
    )
    model.fit(X=x_train, y=y_train)

    # Get feature importance
    importance_df = pd.DataFrame({
        "feature": feat_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    print("\nTop 20 Most Important Features:")
    print("-" * 50)
    for i, row in importance_df.head(20).iterrows():
        bar = "#" * int(row["importance"] / importance_df["importance"].max() * 30)
        print(f"  {row['feature']:<35} {row['importance']:>6.0f} {bar}")

    # Save importance to analyses folder (not model-output)
    analyses_dir = run_config.hub_root / "analyses" / "feature_importance"
    analyses_dir.mkdir(parents=True, exist_ok=True)
    importance_file = analyses_dir / f"{run_config.ref_date}-{model_config.model_name}-feature_importance.csv"
    importance_df.to_csv(importance_file, index=False)
    print(f"\nSaved feature importance to: {importance_file}")

    # Weather feature summary
    weather_features = [f for f in feat_names if any(
        w in f for w in ["temp", "humidity", "precip"]
    )]
    if weather_features:
        weather_importance = importance_df[importance_df["feature"].isin(weather_features)]
        total_weather = weather_importance["importance"].sum()
        total_all = importance_df["importance"].sum()
        print(f"\nWeather features account for {total_weather/total_all*100:.1f}% of total importance")
        print("\nWeather Feature Importance:")
        print("-" * 50)
        for i, row in weather_importance.iterrows():
            print(f"  {row['feature']:<35} {row['importance']:>6.0f}")


if __name__ == "__main__":
    main()
