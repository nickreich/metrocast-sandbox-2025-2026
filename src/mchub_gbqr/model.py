"""GBQR Model for MetroCast Hub forecasting."""

import calendar
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm

from idmodels.preprocess import create_features_and_targets

from .config import ModelConfig, RunConfig
from .data_loader import (
    load_all_data, load_location_crosswalk, get_mchub_locations,
    load_weather_with_lags
)
from .hsa_populations import load_mchub_populations
from .transforms import apply_scale_center_transform, get_transform_factors


class GBQRModel:
    """Gradient Boosted Quantile Regression model for MetroCast Hub."""

    def __init__(self, model_config: ModelConfig):
        """Initialize the model with configuration.

        Args:
            model_config: Model configuration parameters.
        """
        self.config = model_config

    def run(
        self,
        run_config: RunConfig,
        use_local_mchub: bool = False,
        use_versioned_mchub: bool = False
    ) -> pd.DataFrame:
        """Generate predictions and write to file.

        Args:
            run_config: Runtime configuration for this forecast run.
            use_local_mchub: If True, use local MCHub data instead of GitHub.
            use_versioned_mchub: If True, fetch MCHub data as of ref_date using GitHub API.

        Returns:
            DataFrame with predictions in MCHub format.
        """
        # Load all data sources
        df = load_all_data(
            model_config=self.config,
            run_config=run_config,
            use_local_mchub=use_local_mchub,
            use_versioned_mchub=use_versioned_mchub
        )

        # Filter to data available as of reference date
        df = df[df["wk_end_date"] <= pd.Timestamp(run_config.ref_date)]

        # Load location info for geo_type
        locations_df = load_location_crosswalk(run_config.hub_root)

        # Add agg_level from geo_type for compatibility with idmodels preprocessing
        df["agg_level"] = df["geo_type"]

        # Load population for log_pop feature
        mchub_populations = load_mchub_populations(run_config.hub_root)
        df["pop"] = df["location"].map(mchub_populations)
        df["log_pop"] = np.log(df["pop"])

        # Apply power transform and scale/center normalization
        df = apply_scale_center_transform(
            df,
            power_transform=self.config.power_transform,
            group_cols=["source", "location"]
        )

        # Load and merge weather features if enabled
        weather_feat_names = []
        if self.config.use_weather:
            df, weather_feat_names = self._merge_weather_features(df, run_config)

        # Create features and targets using idmodels preprocessing
        init_feats = ["inc_trans_cs", "season_week", "log_pop"] + weather_feat_names
        df, feat_names = create_features_and_targets(
            df=df,
            incl_level_feats=self.config.incl_level_feats,
            max_horizon=run_config.max_horizon,
            curr_feat_names=init_feats
        )

        # Keep only in-season data for training (season_week 5-45)
        df = df.query("season_week >= 5 and season_week <= 45")

        # Test set: most recent week for MCHub locations
        mchub_locs = get_mchub_locations(run_config.hub_root)
        df_test = df.loc[
            (df["wk_end_date"] == df["wk_end_date"].max()) &
            (df["location"].isin(mchub_locs))
        ].copy()

        # Train set: all data with non-missing target
        df_train = df.loc[~df["delta_target"].isna()]

        # Train model and predict
        if self.config.fit_locations_separately:
            locations = df_test["location"].unique()
            preds_dfs = [
                self._train_and_predict(
                    run_config, df_train, df_test, feat_names, location
                )
                for location in tqdm(locations, desc="Locations")
            ]
            preds_df = pd.concat(preds_dfs, axis=0)
        else:
            preds_df = self._train_and_predict(
                run_config, df_train, df_test, feat_names
            )

        # Format as MCHub output
        preds_df = self._format_as_mchub_output(preds_df, run_config, locations_df)

        # Apply quantile non-crossing fix
        preds_df = self._quantile_noncrossing(
            preds_df,
            gcols=["location", "reference_date", "horizon", "target_end_date", "target", "output_type"]
        )

        # Save predictions
        self._save_predictions(preds_df, run_config)

        return preds_df

    def _merge_weather_features(
        self,
        df: pd.DataFrame,
        run_config: RunConfig
    ) -> tuple[pd.DataFrame, list]:
        """Load and merge weather features into the main dataframe.

        Weather data provides temperature and humidity features that can improve
        flu forecasting. Features are merged by location and week_end_date.

        For supplementary data sources (ILINet, NHSN, etc.) that don't have
        direct HSA locations, weather features are filled with NaN and the model
        handles missing values.

        Args:
            df: Main dataframe with incidence data.
            run_config: Runtime configuration.

        Returns:
            Tuple of (dataframe with weather features, list of weather feature names).
        """
        try:
            # Load weather data with lags and rolling averages
            # - Raw features (current week)
            # - Lagged features (weather affects transmission with delay)
            # - Rolling averages (smoothed signal)
            weather_df = load_weather_with_lags(
                weather_file=self.config.weather_file,
                features=self.config.weather_features,
                hub_root=run_config.hub_root,
                lags=self.config.weather_lags,
                rolling_windows=self.config.weather_rolling_windows
            )
        except FileNotFoundError as e:
            print(f"Warning: Weather data not found, skipping weather features: {e}")
            return df, []

        # Get weather feature columns (everything except location and date)
        weather_feat_cols = [
            c for c in weather_df.columns
            if c not in ["location", "wk_end_date"]
        ]

        if not weather_feat_cols:
            print("Warning: No weather features found in data")
            return df, []

        # Merge weather features
        # Note: This will only match MCHub locations directly; supplementary
        # sources with prefixed locations (ilinet_, nhsn_, etc.) won't match
        # and will get NaN values, which LightGBM handles natively
        df = df.merge(
            weather_df[["location", "wk_end_date"] + weather_feat_cols],
            on=["location", "wk_end_date"],
            how="left"
        )

        # Standardize weather features (z-score normalization)
        for col in weather_feat_cols:
            col_mean = df[col].mean()
            col_std = df[col].std()
            if col_std > 0:
                df[col] = (df[col] - col_mean) / col_std

        print(f"Added {len(weather_feat_cols)} weather features: {weather_feat_cols}")

        return df, weather_feat_cols

    def _train_and_predict(
        self,
        run_config: RunConfig,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        feat_names: list,
        location: Optional[str] = None
    ) -> pd.DataFrame:
        """Train model and generate predictions.

        Args:
            run_config: Runtime configuration.
            df_train: Training data.
            df_test: Test data.
            feat_names: List of feature column names.
            location: Optional location to filter to (for per-location fitting).

        Returns:
            DataFrame with predictions.
        """
        if location is not None:
            df_test = df_test.query(f'location == "{location}"').copy()
            df_train = df_train.query(f'location == "{location}"').copy()

        # Get features and target
        x_test = df_test[feat_names]
        x_train = df_train[feat_names]
        y_train = df_train["delta_target"]

        # Get quantile predictions through bagging
        test_pred_qs_df = self._get_test_quantile_predictions(
            run_config, df_train, x_train, y_train, x_test
        )

        # Add predictions to test data
        df_test = df_test.reset_index(drop=True)
        df_test_with_preds = pd.concat([df_test, test_pred_qs_df], axis=1)

        # Melt quantile columns to rows
        cols_to_keep = [
            "source", "location", "wk_end_date", "pop",
            "inc_trans_cs", "horizon",
            "inc_trans_center_factor", "inc_trans_scale_factor"
        ]
        preds_df = df_test_with_preds[cols_to_keep + run_config.q_labels]

        # Filter to MCHub source (predictions only for MCHub locations)
        preds_df = preds_df.loc[preds_df["source"] == "mchub"]

        preds_df = pd.melt(
            preds_df,
            id_vars=cols_to_keep,
            var_name="quantile",
            value_name="delta_hat"
        )

        # Inverse transform to original scale
        preds_df["inc_trans_cs_target_hat"] = preds_df["inc_trans_cs"] + preds_df["delta_hat"]
        preds_df["inc_trans_target_hat"] = (
            (preds_df["inc_trans_cs_target_hat"] + preds_df["inc_trans_center_factor"]) *
            (preds_df["inc_trans_scale_factor"] + 0.01)
        )

        if self.config.power_transform == "4rt":
            inv_power = 4
        elif self.config.power_transform is None:
            inv_power = 1
        else:
            raise ValueError(f'Unsupported power_transform: {self.config.power_transform}')

        preds_df["value"] = np.maximum(preds_df["inc_trans_target_hat"], 0.0) ** inv_power - 0.01
        preds_df["value"] = np.maximum(preds_df["value"], 0.0)

        return preds_df

    def _get_test_quantile_predictions(
        self,
        run_config: RunConfig,
        df_train: pd.DataFrame,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame
    ) -> pd.DataFrame:
        """Train bagged quantile models and get predictions.

        Args:
            run_config: Runtime configuration.
            df_train: Training data (for season sampling).
            x_train: Training features.
            y_train: Training target.
            x_test: Test features.

        Returns:
            DataFrame with quantile predictions as columns.
        """
        # Seed based on reference date for reproducibility
        rng_seed = int(calendar.timegm(run_config.ref_date.timetuple()))
        rng = np.random.default_rng(seed=rng_seed)

        # Seeds for LightGBM fits
        lgb_seeds = rng.integers(
            1e8,
            size=(self.config.num_bags, len(run_config.q_levels))
        )

        # Storage for predictions
        test_preds_by_bag = np.empty((
            x_test.shape[0],
            self.config.num_bags,
            len(run_config.q_levels)
        ))

        train_seasons = df_train["season"].unique()

        for b in tqdm(range(self.config.num_bags), desc="Bag", leave=False):
            # Sample seasons for this bag
            bag_seasons = rng.choice(
                train_seasons,
                size=int(len(train_seasons) * self.config.bag_frac_samples),
                replace=False
            )
            bag_obs_inds = df_train["season"].isin(bag_seasons)

            for q_ind, q_level in enumerate(run_config.q_levels):
                # Fit quantile regression model
                model = lgb.LGBMRegressor(
                    verbosity=-1,
                    objective="quantile",
                    alpha=q_level,
                    random_state=int(lgb_seeds[b, q_ind])
                )
                model.fit(
                    X=x_train.loc[bag_obs_inds, :],
                    y=y_train.loc[bag_obs_inds]
                )

                # Predict on test set
                test_preds_by_bag[:, b, q_ind] = model.predict(X=x_test)

        # Aggregate predictions: median across bags
        test_pred_qs = np.median(test_preds_by_bag, axis=1)

        # Convert to DataFrame with quantile labels as columns
        test_pred_qs_df = pd.DataFrame(test_pred_qs, columns=run_config.q_labels)

        return test_pred_qs_df

    def _format_as_mchub_output(
        self,
        preds_df: pd.DataFrame,
        run_config: RunConfig,
        locations_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Format predictions for MCHub submission.

        Args:
            preds_df: DataFrame with raw predictions.
            run_config: Runtime configuration.
            locations_df: Location crosswalk for target name.

        Returns:
            DataFrame in MCHub format.
        """
        # Calculate target_end_date from horizon
        preds_df["target_end_date"] = (
            preds_df["wk_end_date"] +
            pd.to_timedelta(7 * preds_df["horizon"], unit="days")
        )
        preds_df["reference_date"] = run_config.ref_date

        # Recalculate horizon as days / 7 (for consistency)
        preds_df["horizon"] = (
            pd.to_timedelta(preds_df["target_end_date"].dt.date - run_config.ref_date).dt.days / 7
        ).astype(int)

        # Set target based on location (NYC uses ILI, others use Flu)
        preds_df["target"] = np.where(
            preds_df["location"] == "nyc",
            "ILI ED visits pct",
            "Flu ED visits pct"
        )

        # Output format columns
        preds_df["output_type"] = "quantile"
        preds_df = preds_df.rename(columns={"quantile": "output_type_id"})

        # Select and order final columns
        preds_df = preds_df[[
            "location", "reference_date", "horizon", "target_end_date",
            "target", "output_type", "output_type_id", "value"
        ]]

        return preds_df

    def _quantile_noncrossing(
        self,
        preds_df: pd.DataFrame,
        gcols: list
    ) -> pd.DataFrame:
        """Sort predictions to prevent quantile crossing.

        Args:
            preds_df: DataFrame with quantile predictions.
            gcols: Columns to group by for sorting.

        Returns:
            DataFrame with sorted quantile values.
        """
        g = preds_df.set_index(gcols).groupby(gcols)
        preds_df = g[["output_type_id", "value"]] \
            .transform(lambda x: x.sort_values()) \
            .reset_index()

        return preds_df

    def _save_predictions(
        self,
        preds_df: pd.DataFrame,
        run_config: RunConfig
    ) -> None:
        """Save predictions to file.

        Args:
            preds_df: DataFrame with predictions.
            run_config: Runtime configuration.
        """
        # Create output directory
        model_dir = run_config.output_root / f"UMass-{self.config.model_name}"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Format filename as {ref_date}-UMass-{model_name}.csv
        filename = f"{run_config.ref_date}-UMass-{self.config.model_name}.csv"
        save_path = model_dir / filename

        preds_df.to_csv(save_path, index=False)
        print(f"Saved predictions to {save_path}")
