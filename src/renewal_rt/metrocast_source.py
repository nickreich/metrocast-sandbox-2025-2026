"""A `iddata` DataSource for the flu-metrocast hub target series.

The metrocast targets ("Flu ED visits pct" for NSSP locations, "ILI ED visits pct" for
NYC) are the quantities we are scored on, so we forecast them directly rather than the
raw iddata NSSP feed. This wraps the hub's target-data file as a standard `DataSource`
so it plugs into `DiseaseDataLoader` / `IDModel` unchanged.

Standard iddata schema returned by `load()`:
    location, agg_level, wk_end_date, season, season_week, inc, source
"""

from __future__ import annotations

import datetime
import io
from pathlib import Path

import numpy as np
import pandas as pd

from iddata.enums import SourceType
from iddata.sources.base import DataSource
from iddata.utils import convert_epiweek_to_season, convert_epiweek_to_season_week

import pymmwr

MCHUB_RAW_URL = (
    "https://raw.githubusercontent.com/reichlab/flu-metrocast/main/target-data/latest-data.csv"
)


def _season_columns(wk_end_date: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Compute (season, season_week) from a datetime Series using iddata conventions."""
    def _ew_str(d):
        ew = pymmwr.date_to_epiweek(pd.Timestamp(d).date())
        return f"{ew.year}{ew.week:02d}"
    ew = wk_end_date.map(_ew_str).astype(str)
    return convert_epiweek_to_season(ew), convert_epiweek_to_season_week(ew)


class MetrocastTargetSource(DataSource):
    """Load flu-metrocast hub target data as a standard iddata DataSource.

    Parameters
    ----------
    hub_root : Path | None
        Path to a metrocast hub checkout. If given and `use_local`, reads
        `hub_root/target-data/latest-data.csv` and `hub_root/auxiliary-data/locations.csv`.
    use_local : bool
        If True, read the local target file; otherwise download the latest from GitHub.
    target : str
        Which target to keep ("Flu ED visits pct" by default).
    """

    source_name = SourceType.NSSP  # metrocast flu-pct is NSSP-derived

    def __init__(self, hub_root: Path | None = None, use_local: bool = True,
                 target: str = "Flu ED visits pct", include_nyc_ili: bool = False,
                 versioned: bool = False):
        self.hub_root = Path(hub_root) if hub_root is not None else None
        self.use_local = use_local
        self.target = target
        self.versioned = versioned
        # NYC's target is ILI (not flu); include it when running all locations
        self.targets = [target] + (["ILI ED visits pct"] if include_nyc_ili else [])

    def _read_raw(self, as_of: datetime.date | None = None) -> pd.DataFrame:
        if self.versioned:
            # Real-time reconstruction. The hub's versioned time-series only extends back to
            # ~2024-08, so we use the as-of vintage for the weeks it covers (which truncates at the
            # last week reported by `as_of` and carries preliminary/revisable current-season values)
            # and splice in the stable finalized history for older seasons from latest-data.csv
            # (those weeks are not meaningfully revised).
            if self.hub_root is None:
                raise ValueError("hub_root required for versioned data")
            cols = ["target_end_date", "location", "target", "observation"]
            ts = pd.read_csv(self.hub_root / "target-data" / "time-series.csv")
            ts["as_of_d"] = pd.to_datetime(ts["as_of"])
            if as_of is not None:
                ts = ts[ts["as_of_d"] <= pd.Timestamp(as_of)]
            if ts.empty:
                raise ValueError(f"No versioned data available as of {as_of}.")
            vint = (ts.sort_values("as_of_d")
                      .drop_duplicates(subset=["location", "target", "target_end_date"], keep="last")
                      [cols])
            vint_min = pd.to_datetime(vint["target_end_date"]).min()
            ld = pd.read_csv(self.hub_root / "target-data" / "latest-data.csv")
            older = ld[pd.to_datetime(ld["target_end_date"]) < vint_min][cols]
            return pd.concat([older, vint], ignore_index=True)
        if self.use_local:
            if self.hub_root is None:
                raise ValueError("hub_root required when use_local=True")
            return pd.read_csv(self.hub_root / "target-data" / "latest-data.csv")
        import urllib.request
        with urllib.request.urlopen(MCHUB_RAW_URL) as resp:
            return pd.read_csv(io.StringIO(resp.read().decode("utf-8")))

    def _agg_levels(self) -> pd.DataFrame:
        """location -> agg_level from the hub locations crosswalk."""
        if self.hub_root is None:
            return pd.DataFrame(columns=["location", "agg_level"])
        loc = pd.read_csv(self.hub_root / "auxiliary-data" / "locations.csv", dtype=str)
        agg = loc["location_type"].map(
            {"hsa_nci_id": "hsa", "nc_flu_region_id": "region"}
        ).fillna("hsa")
        agg = np.where(loc["original_location_code"] == "All", "state", agg)
        return pd.DataFrame({"location": loc["location"], "agg_level": agg})

    def load(self, as_of: datetime.date | None = None) -> pd.DataFrame:
        df = self._read_raw(as_of)
        df = df[df["target"].isin(self.targets)].copy()
        df["wk_end_date"] = pd.to_datetime(df["target_end_date"])
        if as_of is not None:
            df = df[df["wk_end_date"] <= pd.Timestamp(as_of)]

        df = df.rename(columns={"observation": "inc"})
        df["season"], df["season_week"] = _season_columns(df["wk_end_date"])
        df["source"] = self.source_name.value

        agg = self._agg_levels()
        df = df.merge(agg, on="location", how="left")
        df["agg_level"] = df["agg_level"].fillna("state")

        df = (
            df.dropna(subset=["inc"])
            .drop_duplicates(subset=["location", "wk_end_date"], keep="first")
            .sort_values(["location", "wk_end_date"])
        )
        return df[["location", "agg_level", "wk_end_date", "season", "season_week", "inc", "source"]]
