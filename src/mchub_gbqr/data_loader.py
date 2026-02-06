"""Data loading utilities for GBQR model."""

import datetime
import io
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pymmwr

from iddata.loader import DiseaseDataLoader

from .config import ModelConfig, RunConfig


# MCHub GitHub repository
MCHUB_REPO = "reichlab/flu-metrocast"
MCHUB_TARGET_DATA_PATH = "target-data/latest-data.csv"
MCHUB_RAW_URL = f"https://raw.githubusercontent.com/{MCHUB_REPO}/main/{MCHUB_TARGET_DATA_PATH}"


def _download_latest_from_github() -> pd.DataFrame:
    """Download the latest target data directly from GitHub.

    Returns:
        DataFrame with the latest target data.
    """
    import urllib.request

    try:
        with urllib.request.urlopen(MCHUB_RAW_URL) as response:
            content = response.read().decode('utf-8')
        return pd.read_csv(io.StringIO(content))
    except Exception as e:
        raise RuntimeError(f"Failed to download from GitHub: {e}") from e


def _get_commit_at_date(repo: str, file_path: str, as_of_date: datetime.date) -> str:
    """Get the most recent commit SHA for a file on or before the given date.

    Uses GitHub CLI to query commits. The cutoff is 11:59pm ET on the as_of_date.

    Args:
        repo: GitHub repository in "owner/repo" format.
        file_path: Path to the file within the repository.
        as_of_date: Date to query (will use 11:59pm ET as cutoff).

    Returns:
        Commit SHA string.

    Raises:
        RuntimeError: If no commits found or gh CLI fails.
    """
    # Convert to datetime at 11:59pm ET (use 23:59:59 in America/New_York)
    # For simplicity, we'll use UTC offset for ET (typically -5 or -4 hours)
    # 11:59pm ET = 04:59am UTC next day (EST) or 03:59am UTC next day (EDT)
    # We'll use --until with the next day at 05:00 UTC to be safe
    next_day = as_of_date + datetime.timedelta(days=1)
    until_date = next_day.isoformat()

    # Query commits to this file up to the cutoff date
    cmd = [
        "gh", "api",
        f"/repos/{repo}/commits",
        "-q", ".[0].sha",
        "-f", f"path={file_path}",
        "-f", f"until={until_date}T04:59:59Z",
        "-f", "per_page=1"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        commit_sha = result.stdout.strip()
        if not commit_sha:
            raise RuntimeError(f"No commits found for {file_path} on or before {as_of_date}")
        return commit_sha
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"GitHub CLI failed: {e.stderr}") from e


def _fetch_file_at_commit(repo: str, file_path: str, commit_sha: str) -> str:
    """Fetch file content at a specific commit.

    Args:
        repo: GitHub repository in "owner/repo" format.
        file_path: Path to the file within the repository.
        commit_sha: Commit SHA to fetch.

    Returns:
        File content as string.
    """
    cmd = [
        "gh", "api",
        f"/repos/{repo}/contents/{file_path}",
        "-q", ".content",
        "-H", "Accept: application/vnd.github.v3.raw",
        "-f", f"ref={commit_sha}"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to fetch file: {e.stderr}") from e


def _date_to_season(date: datetime.date) -> str:
    """Convert a date to season string (e.g., '2023/24').

    Season runs from epiweek 40 of year Y to epiweek 39 of year Y+1.
    """
    ew = pymmwr.date_to_epiweek(date)
    year = ew.year
    week = ew.week

    if week >= 40:
        return f"{year}/{str(year + 1)[2:]}"
    else:
        return f"{year - 1}/{str(year)[2:]}"


def _date_to_season_week(date: datetime.date) -> int:
    """Convert a date to season week number (1-52).

    Season week 1 is epiweek 40, season week 13 is epiweek 52,
    season week 14 is epiweek 1, etc.
    """
    ew = pymmwr.date_to_epiweek(date)
    week = ew.week

    if week >= 40:
        return week - 39
    else:
        return week + 13


def load_mchub_data(
    ref_date: datetime.date,
    locations_df: pd.DataFrame,
    use_local: bool = False,
    use_versioned: bool = False,
    local_path: Optional[Path] = None
) -> pd.DataFrame:
    """Load MCHub target data from GitHub or local file.

    Args:
        ref_date: Reference date (used for season calculations and versioning).
        locations_df: Location crosswalk DataFrame.
        use_local: If True, use local file instead of downloading from GitHub.
        use_versioned: If True, fetch data as of ref_date using GitHub API versioning.
        local_path: Path to local latest-data.csv (required if use_local=True).

    Returns:
        DataFrame with columns:
            - location: MCHub location slug
            - source: "mchub"
            - geo_type: "state", "hsa", or "nc_region"
            - wk_end_date: Week ending date
            - season: Season string (e.g., "2023/24")
            - season_week: Week number within season
            - inc: Observation value (percentage)
            - target: Target name from MCHub
    """
    if use_local:
        if local_path is None:
            raise ValueError("local_path required when use_local=True")
        df = pd.read_csv(local_path)
    elif use_versioned:
        # Fetch versioned data from GitHub using commit at ref_date
        commit_sha = _get_commit_at_date(MCHUB_REPO, MCHUB_TARGET_DATA_PATH, ref_date)
        content = _fetch_file_at_commit(MCHUB_REPO, MCHUB_TARGET_DATA_PATH, commit_sha)
        df = pd.read_csv(io.StringIO(content))
    else:
        # Download latest data from GitHub (default)
        df = _download_latest_from_github()

    # Convert date column
    df["wk_end_date"] = pd.to_datetime(df["target_end_date"])

    # Add season and season_week
    df["season"] = df["wk_end_date"].apply(lambda x: _date_to_season(x.date()))
    df["season_week"] = df["wk_end_date"].apply(lambda x: _date_to_season_week(x.date()))

    # Merge with locations to get geo_type
    loc_cols = locations_df[["location", "location_type"]].copy()
    loc_cols["geo_type"] = loc_cols["location_type"].map({
        "hsa_nci_id": "hsa",
        "nc_flu_region_id": "nc_region"
    })
    # States have original_location_code == "All"
    state_locs = locations_df[locations_df["original_location_code"] == "All"]["location"].tolist()
    loc_cols.loc[loc_cols["location"].isin(state_locs), "geo_type"] = "state"

    df = df.merge(loc_cols[["location", "geo_type"]], on="location", how="left")

    # Set source and rename columns
    df["source"] = "mchub"
    df = df.rename(columns={"observation": "inc"})

    # Drop duplicates (keep first occurrence)
    df = df.drop_duplicates(subset=["location", "wk_end_date", "target"], keep="first")

    # Select and order columns
    df = df[["location", "source", "geo_type", "wk_end_date", "season", "season_week", "inc", "target"]]

    return df


def load_location_crosswalk(hub_root: Path) -> pd.DataFrame:
    """Load location crosswalk from MCHub auxiliary data.

    Args:
        hub_root: Path to MCHub repository root.

    Returns:
        DataFrame with location mapping information.
    """
    locations_path = hub_root / "auxiliary-data" / "locations.csv"
    return pd.read_csv(locations_path, dtype=str)


def load_supplementary_ilinet(as_of_date: datetime.date) -> pd.DataFrame:
    """Load ILINet data from iddata.

    Args:
        as_of_date: Reference date for data versioning.

    Returns:
        DataFrame with ILINet data, locations prefixed with "ilinet_".
    """
    loader = DiseaseDataLoader()
    df = loader.load_ilinet(scale_to_positive=False, drop_pandemic_seasons=True)

    # Drop locations with known data quality issues
    drop_locations = ["Virgin Islands", "Puerto Rico", "District of Columbia"]
    df = df[~df["location"].isin(drop_locations)]

    # Prefix locations
    df["location"] = "ilinet_" + df["location"].astype(str)
    df["source"] = "ilinet"

    # Standardize geo_type
    df["geo_type"] = df["agg_level"].map({
        "national": "national",
        "hhs": "hhs",
        "state": "state"
    })

    return df[["location", "source", "geo_type", "wk_end_date", "season", "season_week", "inc"]]


def load_supplementary_flusurvnet(as_of_date: datetime.date) -> pd.DataFrame:
    """Load FluSurvNet data from iddata.

    Args:
        as_of_date: Reference date for data versioning.

    Returns:
        DataFrame with FluSurvNet data, locations prefixed with "flusurv_".
    """
    loader = DiseaseDataLoader()
    df = loader.load_flusurv_rates(burden_adj=False)

    # Prefix locations
    df["location"] = "flusurv_" + df["location"].astype(str)
    df["source"] = "flusurvnet"

    # Standardize geo_type
    df["geo_type"] = df["agg_level"].map({
        "national": "national",
        "site": "site"
    })

    return df[["location", "source", "geo_type", "wk_end_date", "season", "season_week", "inc"]]


def load_supplementary_nhsn(as_of_date: datetime.date) -> pd.DataFrame:
    """Load NHSN data from iddata.

    Args:
        as_of_date: Reference date for data versioning.

    Returns:
        DataFrame with NHSN data, locations prefixed with "nhsn_".
    """
    loader = DiseaseDataLoader()
    df = loader.load_nhsn(disease="flu", as_of=as_of_date, drop_pandemic_seasons=True)

    # Prefix locations
    df["location"] = "nhsn_" + df["location"].astype(str)
    df["source"] = "nhsn"

    # Standardize geo_type
    df["geo_type"] = df["agg_level"].map({
        "national": "national",
        "state": "state"
    })

    return df[["location", "source", "geo_type", "wk_end_date", "season", "season_week", "inc"]]


def load_supplementary_nssp(as_of_date: datetime.date) -> pd.DataFrame:
    """Load NSSP data from iddata (extra locations beyond MCHub).

    Args:
        as_of_date: Reference date for data versioning.

    Returns:
        DataFrame with NSSP data, locations prefixed with "nssp_{agg_level}_".
    """
    loader = DiseaseDataLoader()
    df = loader.load_nssp(disease="flu", as_of=as_of_date, drop_pandemic_seasons=True)

    # Prefix locations with agg_level to avoid collisions between state FIPS and HSA NCI IDs
    # (e.g., state "11" (DC) vs HSA "11" both exist)
    df["location"] = "nssp_" + df["agg_level"].astype(str) + "_" + df["location"].astype(str)
    df["source"] = "nssp"

    # Standardize geo_type
    df["geo_type"] = df["agg_level"].map({
        "national": "national",
        "state": "state",
        "hsa": "hsa"
    })

    return df[["location", "source", "geo_type", "wk_end_date", "season", "season_week", "inc"]]


def load_all_data(
    model_config: ModelConfig,
    run_config: RunConfig,
    use_local_mchub: bool = False,
    use_versioned_mchub: bool = False
) -> pd.DataFrame:
    """Load all data sources according to configuration.

    Args:
        model_config: Model configuration with data source toggles.
        run_config: Run configuration with paths and dates.
        use_local_mchub: If True, use local MCHub data instead of GitHub.
        use_versioned_mchub: If True, fetch MCHub data as of ref_date using GitHub API.

    Returns:
        Combined DataFrame with all data sources.
    """
    frames = []

    # Load location crosswalk
    locations_df = load_location_crosswalk(run_config.hub_root)

    # Always load MCHub data (primary source)
    local_path = run_config.hub_root / "target-data" / "latest-data.csv" if use_local_mchub else None
    df_mchub = load_mchub_data(
        ref_date=run_config.ref_date,
        locations_df=locations_df,
        use_local=use_local_mchub,
        use_versioned=use_versioned_mchub,
        local_path=local_path
    )
    # Drop the target column for consistency with supplementary sources
    df_mchub_train = df_mchub.drop(columns=["target"])
    frames.append(df_mchub_train)

    # Load supplementary sources if enabled
    if model_config.use_ilinet:
        df_ilinet = load_supplementary_ilinet(run_config.ref_date)
        frames.append(df_ilinet)

    if model_config.use_flusurvnet:
        df_flusurv = load_supplementary_flusurvnet(run_config.ref_date)
        frames.append(df_flusurv)

    if model_config.use_nhsn:
        df_nhsn = load_supplementary_nhsn(run_config.ref_date)
        frames.append(df_nhsn)

    if model_config.use_nssp_extra:
        df_nssp = load_supplementary_nssp(run_config.ref_date)
        frames.append(df_nssp)

    # Combine all sources
    df_combined = pd.concat(frames, ignore_index=True)

    # Drop seasons specified in config
    if model_config.drop_seasons:
        df_combined = df_combined[~df_combined["season"].isin(model_config.drop_seasons)]

    return df_combined


def get_mchub_locations(hub_root: Path) -> list:
    """Get list of MCHub location slugs.

    Args:
        hub_root: Path to MCHub repository root.

    Returns:
        List of location slug strings.
    """
    locations_df = load_location_crosswalk(hub_root)
    return locations_df["location"].tolist()


def load_weather_data(
    weather_file: Optional[Path] = None,
    features: Optional[list] = None,
    hub_root: Optional[Path] = None
) -> pd.DataFrame:
    """Load weather data for HSA locations.

    Weather data provides temperature and humidity features that can improve
    flu forecasting, as flu transmission is affected by weather conditions.

    Args:
        weather_file: Path to weather CSV file. If None, looks in default location.
        features: List of weather feature columns to include. If None, uses all available.
        hub_root: Hub root path for finding default weather file location.

    Returns:
        DataFrame with columns:
            - location: HSA location slug
            - wk_end_date: Week ending date (Saturday)
            - [weather features]: Temperature, humidity, etc.

    Raises:
        FileNotFoundError: If weather file not found.
    """
    # Find weather file
    if weather_file is not None:
        file_path = weather_file
    elif hub_root is not None:
        # Default location: auxiliary-data/weather_by_hsa.csv (relative to hub root)
        file_path = hub_root / "auxiliary-data" / "weather_by_hsa.csv"
    else:
        raise ValueError("Must provide either weather_file or hub_root")

    if not file_path.exists():
        raise FileNotFoundError(f"Weather data file not found: {file_path}")

    # Load weather data
    df = pd.read_csv(file_path)
    df["wk_end_date"] = pd.to_datetime(df["week_end_date"])

    # Available weather features
    all_weather_cols = [
        "temp_avg_f", "temp_max_f", "temp_min_f",
        "humidity_avg_pct", "humidity_abs_gm3", "precip_total_mm"
    ]

    # Filter to requested features (if specified)
    if features is not None:
        weather_cols = [c for c in features if c in df.columns]
    else:
        weather_cols = [c for c in all_weather_cols if c in df.columns]

    # Select columns
    keep_cols = ["location", "wk_end_date"] + weather_cols
    df = df[[c for c in keep_cols if c in df.columns]]

    return df


def load_weather_with_lags(
    weather_file: Optional[Path] = None,
    features: Optional[list] = None,
    hub_root: Optional[Path] = None,
    lags: Optional[list] = None,
    rolling_windows: Optional[list] = None
) -> pd.DataFrame:
    """Load weather data with optional lagged and rolling average features.

    Creates lagged versions of weather features (e.g., last week's temperature)
    and rolling averages (e.g., trailing 3-week mean) which may be useful
    predictors since weather affects flu transmission with some delay.

    Args:
        weather_file: Path to weather CSV file.
        features: List of weather feature columns to include.
        hub_root: Hub root path for finding default weather file location.
        lags: List of lag values in weeks (e.g., [1, 2] for 1-week and 2-week lags).
              If None, no lags are created.
        rolling_windows: List of window sizes for trailing rolling averages
              (e.g., [3] for 3-week trailing average). Windows are right-aligned
              and include the current week. If None, no rolling averages created.

    Returns:
        DataFrame with weather features, lagged versions, and rolling averages.
    """
    df = load_weather_data(
        weather_file=weather_file,
        features=features,
        hub_root=hub_root
    )

    # Get weather feature columns (everything except location and date)
    weather_cols = [c for c in df.columns if c not in ["location", "wk_end_date"]]

    # Sort by location and date for proper lag/rolling calculation
    df = df.sort_values(["location", "wk_end_date"])

    # Create lagged features
    if lags is not None and len(lags) > 0:
        for lag in lags:
            for col in weather_cols:
                lag_col_name = f"{col}_lag{lag}"
                df[lag_col_name] = df.groupby("location")[col].shift(lag)

    # Create rolling average features (trailing, right-aligned)
    if rolling_windows is not None and len(rolling_windows) > 0:
        for window in rolling_windows:
            for col in weather_cols:
                roll_col_name = f"{col}_roll{window}"
                # min_periods=1 allows partial windows at start of series
                df[roll_col_name] = df.groupby("location")[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )

    return df
