"""Renewal / Rt-based forecast model for the flu-metrocast hub (sample output).

Pipeline (per state-level location, v1):
  1. Estimate a historical instantaneous growth rate r_t from the weekly percentage
     via a penalized smoothing spline on log-incidence (GAM-equivalent), then map to
     R_t with the Euler-Lotka relation and a flu generation interval.  (rt_estimation.py)
  2. Model log R_t as a pooled seasonal climatology by season-week + a location offset
     + an AR(1) deviation.  Historical data are thin (~3 seasons of state NSSP), so the
     seasonal shape is pooled across all locations.
  3. Forecast forward by sampling AR(1) deviations around the seasonal climatology to
     get coherent R_t trajectories, then project incidence with the renewal-consistent
     step  p_{t+1} = p_t * exp(r(R_{t+1}) * 7).  Each sample index is one full-season
     trajectory -> satisfies the hub's 2026-27 sample-output requirement.

Subclasses idmodels.IDModel and uses the iddata DataSource abstraction, but overrides
run() with a metrocast-specific orchestration: the stock IDModel.run() assumes census
PopulationData, FluSight quantile output, and an NSSP percentage->proportion rescale,
none of which apply here.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import _stubs
_stubs.install()  # must precede idmodels imports

from idmodels.config import ModelConfig, RunConfig  # noqa: E402
from idmodels.features import FeaturePipeline  # noqa: E402
from idmodels.model import IDModel  # noqa: E402

from .metrocast_source import MetrocastTargetSource, _season_columns  # noqa: E402
from .rt_estimation import (  # noqa: E402
    discrete_generation_interval,
    estimate_rt_series,
    euler_lotka_r_from_R,
    make_link,
)

_DEFAULT_Q_LEVELS = [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975]
_DEFAULT_Q_LABELS = ["0.025", "0.05", "0.1", "0.25", "0.5", "0.75", "0.9", "0.95", "0.975"]


@dataclass
class RenewalModelConfig(ModelConfig):
    """Configuration for the renewal/Rt model (extends idmodels.ModelConfig)."""

    # --- output naming (team-model, e.g. epiENGAGE-renewal_rt) ---
    team_abbr: str = "epiENGAGE"

    # --- data source ---
    hub_root: Path | None = None
    use_local: bool = True
    target: str = "Flu ED visits pct"
    include_nyc: bool = False  # also load NYC's ILI target (for all-location runs)
    versioned: bool = False    # use the hub's versioned time-series (real-time vintages) instead of latest

    # --- generation interval (days) ---
    gi_mean_days: float = 3.0
    gi_sd_days: float = 1.5
    gi_max_days: int = 21

    # --- working scale ---
    transform: str = "log"       # "log" (default); OU reversion + value_max clip bound the fan
    value_max: float = 100.0     # upper bound for percentage targets (hub max)

    # --- Rt estimation ---
    log_floor: float = 0.01      # offset before the link to handle zeros
    resid_sd: float = 0.15       # spline smoothing budget (working scale); larger = smoother

    # --- Rt forecasting (seasonal climatology + AR(1)) ---
    in_season_min: int = 10      # season-week bounds (week-30 origin) used for climatology
    in_season_max: int = 45
    clim_smooth_window: int = 5  # rolling window (season-weeks) to smooth the climatology
    loc_offset_shrink: float = 0.5   # shrink location-specific Rt offset toward the pooled mean
    level_offset_shrink: float = 1.0  # location-specific level magnitude (1.0 = fully location-specific)
    ar_phi_max: float = 0.8      # cap on AR(1) persistence
    obs_sigma: float = 0.02      # small process noise on the level per step (working scale)
    nowcast_sigma: float = 0.08  # anchor (nowcast/revision) uncertainty at the seed (working scale)
    level_revert_kappa: float = 0.15  # OU reversion of the level toward seasonal climatology (lower = slower)

    # --- output ---
    output_type: str = "sample"  # "sample" (100 trajectories) or "quantile" (hub-comparable)
    n_samples: int = 100
    q_levels: list = field(default_factory=lambda: list(_DEFAULT_Q_LEVELS))
    q_labels: list = field(default_factory=lambda: list(_DEFAULT_Q_LABELS))


class RenewalRtModel(IDModel):
    """Renewal/Rt sample-trajectory model for flu-metrocast."""

    def __init__(self, model_config: RenewalModelConfig):
        super().__init__(model_config)
        self.model_config: RenewalModelConfig = model_config

    # ---- abstract-method implementations (part of the idmodels class structure) ----
    def _build_sources(self, run_config: RunConfig):
        return [MetrocastTargetSource(hub_root=self.model_config.hub_root,
                                      use_local=self.model_config.use_local,
                                      target=self.model_config.target,
                                      include_nyc_ili=self.model_config.include_nyc,
                                      versioned=self.model_config.versioned)]

    def _build_feature_pipeline(self, run_config: RunConfig) -> FeaturePipeline:
        # The renewal model builds its own state internally; no engineered features.
        return FeaturePipeline(features=[], initial_feat_names=["inc"])

    # ---- orchestration (metrocast-specific) ----
    def run(self, run_config: RunConfig) -> pd.DataFrame:
        from iddata.loader import DiseaseDataLoader

        sources = self._build_sources(run_config)
        df = DiseaseDataLoader().load(sources=sources, as_of=run_config.ref_date,
                                      ancillary=None, drop_pandemic_seasons=False)
        df = df[df["wk_end_date"] <= pd.Timestamp(run_config.ref_date)].copy()

        locs = list(run_config.states) + list(run_config.hsas)
        df = df[df["location"].isin(locs)].copy()
        if df.empty:
            raise ValueError("No data for the requested locations as of the reference date.")

        samples_df = self._fit_and_predict(df, [], run_config)
        if self.model_config.output_type == "quantile":
            preds_df = self._samples_to_quantiles(samples_df)
        else:
            preds_df = samples_df
        preds_df = self._format_output(preds_df, run_config)

        team_model = f"{self.model_config.team_abbr}-{self.model_config.model_name}"
        save_dir = run_config.output_root / team_model
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{run_config.ref_date}-{team_model}.csv"
        preds_df["output_type_id"] = preds_df["output_type_id"].astype(str)
        preds_df.to_csv(save_path, index=False)
        print(f"Saved {len(preds_df)} rows to {save_path}")
        return preds_df

    # ---- model core ----
    def _fit_and_predict(self, df: pd.DataFrame, feat_names, run_config: RunConfig) -> pd.DataFrame:
        cfg = self.model_config
        g = discrete_generation_interval(cfg.gi_mean_days, cfg.gi_sd_days, cfg.gi_max_days)
        link_fwd, link_inv = make_link(cfg.transform, value_max=cfg.value_max, floor=cfg.log_floor)
        rng = np.random.default_rng(seed=int(calendar.timegm(run_config.ref_date.timetuple())))

        # 1. per-location historical Rt + fitted incidence (zlevel = fitted on the working scale)
        hist, series = [], {}
        for loc, d in df.groupby("location"):
            d = d.sort_values("wk_end_date").reset_index(drop=True)
            if d["inc"].notna().sum() < 8:
                continue
            t_days = (d["wk_end_date"] - d["wk_end_date"].min()).dt.days.values
            fitted, r, R = estimate_rt_series(t_days, d["inc"].values, g,
                                              link_fwd, link_inv, resid_sd=cfg.resid_sd)
            d = d.assign(fitted=fitted, r=r, R=R,
                         logR=np.log(np.clip(R, 1e-3, None)),
                         zlevel=link_fwd(fitted))
            series[loc] = d
            hist.append(d[["location", "season", "season_week", "logR", "zlevel"]])
        if not series:
            raise ValueError("No location had enough data to estimate Rt.")
        hist = pd.concat(hist, ignore_index=True)

        # diagnostics: full per-location historical Rt estimates
        self.rt_history_ = pd.concat(
            [d[["location", "season", "season_week", "wk_end_date", "R"]] for d in series.values()],
            ignore_index=True,
        )

        # 2. pooled seasonal climatology of logR by season-week (in-season only)
        ins = hist[(hist.season_week >= cfg.in_season_min) & (hist.season_week <= cfg.in_season_max)]
        clim = ins.groupby("season_week")["logR"].mean()
        full_weeks = np.arange(cfg.in_season_min, cfg.in_season_max + 1)
        clim = clim.reindex(full_weeks).interpolate().ffill().bfill()
        clim = clim.rolling(cfg.clim_smooth_window, center=True, min_periods=1).mean()
        clim_map = clim.to_dict()

        def clim_at(sw: int) -> float:
            sw = int(np.clip(sw, cfg.in_season_min, cfg.in_season_max))
            return float(clim_map[sw])

        # diagnostics: pooled Rt climatology in R space (before location offsets)
        self.rt_clim_ = pd.DataFrame({"season_week": list(clim_map.keys()),
                                      "R_clim": np.exp(list(clim_map.values()))})

        # pooled seasonal climatology of the working-scale LEVEL by season-week (reversion target)
        levclim = ins.groupby("season_week")["zlevel"].mean()
        levclim = levclim.reindex(full_weeks).interpolate().ffill().bfill()
        levclim = levclim.rolling(cfg.clim_smooth_window, center=True, min_periods=1).mean()
        levclim_map = levclim.to_dict()

        def levclim_at(sw: int) -> float:
            sw = int(np.clip(sw, cfg.in_season_min, cfg.in_season_max))
            return float(levclim_map[sw])

        # location-specific level offset (shrunk toward the pooled climatology)
        loc_level_offset = {}
        for loc, d in series.items():
            di = d[(d.season_week >= cfg.in_season_min) & (d.season_week <= cfg.in_season_max)]
            if di.empty:
                loc_level_offset[loc] = 0.0
                continue
            dev = di["zlevel"].values - np.array([levclim_at(sw) for sw in di["season_week"]])
            loc_level_offset[loc] = cfg.level_offset_shrink * float(np.mean(dev))

        # location-specific offset (shrunk), and pooled AR(1) parameters
        loc_offset = {}
        e_all, elag_all = [], []
        for loc, d in series.items():
            di = d[(d.season_week >= cfg.in_season_min) & (d.season_week <= cfg.in_season_max)]
            if di.empty:
                loc_offset[loc] = 0.0
                continue
            dev = di["logR"].values - np.array([clim_at(sw) for sw in di["season_week"]])
            loc_offset[loc] = cfg.loc_offset_shrink * float(np.mean(dev))
            # residuals for AR(1), consecutive weeks within season
            e = dev - loc_offset[loc]
            for (_, seg) in di.assign(e=e).groupby("season"):
                ev = seg["e"].values
                if len(ev) >= 2:
                    e_all.append(ev[1:])
                    elag_all.append(ev[:-1])
        if e_all:
            e_cur = np.concatenate(e_all)
            e_lag = np.concatenate(elag_all)
            phi = float(np.clip(np.sum(e_cur * e_lag) / max(np.sum(e_lag ** 2), 1e-8),
                                0.0, cfg.ar_phi_max))
            sigma = float(np.std(e_cur - phi * e_lag))
        else:
            phi, sigma = 0.5, 0.1
        sigma = max(sigma, 1e-3)

        # 3. forecast target dates: horizons 0..max_horizon from the reference date
        ref = run_config.ref_date
        target_dates = [pd.Timestamp(ref) + pd.Timedelta(days=7 * h)
                        for h in range(run_config.max_horizon + 1)]
        max_target = max(target_dates)

        out = []
        rt_fc = []  # diagnostics: forecast Rt samples (location, target_end_date, season_week, R vector)
        for loc, d in series.items():
            last = d.iloc[-1]
            last_date = last["wk_end_date"]
            z0 = float(last["zlevel"])  # anchor on the working scale
            e0 = float(last["logR"]) - clim_at(last["season_week"]) - loc_offset[loc]

            # anchor the trajectories at the current (nowcast) level with revision uncertainty
            z = z0 + rng.normal(0.0, cfg.nowcast_sigma, size=cfg.n_samples)
            grid_vals = {last_date: link_inv(z)}

            # future weekly grid strictly after last_date, up to max_target
            steps = pd.date_range(last_date + pd.Timedelta(days=7), max_target, freq="7D")
            if len(steps) > 0:
                sw_steps = _season_columns(pd.Series(steps))[1].values
                e = np.full(cfg.n_samples, e0)
                for i, dt in enumerate(steps):
                    sw = sw_steps[i]
                    # sampled R trajectory: seasonal climatology + location offset + AR(1) deviation
                    e = phi * e + rng.normal(0.0, sigma, size=cfg.n_samples)
                    logR = clim_at(sw) + loc_offset[loc] + e
                    R_samp = np.exp(logR)
                    rt_fc.append((loc, dt, int(sw), R_samp.copy()))
                    r = euler_lotka_r_from_R(R_samp, g)  # per-day growth rate
                    # renewal growth step + OU reversion of the level toward seasonal climatology
                    # (a saturation / susceptible-depletion proxy that bounds long-horizon variance)
                    level_target = levclim_at(sw) + loc_level_offset[loc]
                    z = (z + r * 7.0
                         - cfg.level_revert_kappa * (z - level_target)
                         + rng.normal(0.0, cfg.obs_sigma, size=cfg.n_samples))
                    grid_vals[dt] = link_inv(z)

            grid_dates = np.array(sorted(grid_vals.keys()))
            for td in target_dates:
                if td <= last_date:
                    vals = grid_vals[last_date]
                else:
                    # nearest grid date on/after td (dates align on Saturdays, so exact)
                    key = grid_dates[np.searchsorted(grid_dates, td)]
                    vals = grid_vals[key]
                for s in range(cfg.n_samples):
                    out.append((loc, td, s + 1, float(np.clip(vals[s], 0.0, cfg.value_max))))

        # diagnostics: summarize forecast Rt trajectories by location & season-week
        self.rt_forecast_ = pd.DataFrame([
            {"location": loc, "target_end_date": dt, "season_week": sw,
             "R_med": float(np.median(Rv)),
             "R_lo": float(np.quantile(Rv, 0.05)), "R_hi": float(np.quantile(Rv, 0.95)),
             "R_lo50": float(np.quantile(Rv, 0.25)), "R_hi50": float(np.quantile(Rv, 0.75))}
            for loc, dt, sw, Rv in rt_fc
        ])

        return pd.DataFrame(out, columns=["location", "target_end_date", "output_type_id", "value"])

    def _samples_to_quantiles(self, samples_df: pd.DataFrame) -> pd.DataFrame:
        """Collapse the trajectory samples to hub quantile levels (per location & date).

        Produces the same quantile output_type used by the other hub models, so the renewal
        model can be compared directly. Quantiles of a common sample set are monotone by
        construction (no crossing).
        """
        cfg = self.model_config
        rows = []
        for (loc, td), grp in samples_df.groupby(["location", "target_end_date"]):
            qs = np.quantile(grp["value"].values, cfg.q_levels)
            for lab, val in zip(cfg.q_labels, qs):
                rows.append((loc, td, lab, float(val)))
        return pd.DataFrame(rows, columns=["location", "target_end_date", "output_type_id", "value"])

    def _format_output(self, preds_df: pd.DataFrame, run_config: RunConfig) -> pd.DataFrame:
        preds_df = preds_df.copy()
        preds_df["reference_date"] = pd.Timestamp(run_config.ref_date)
        preds_df["horizon"] = (
            (preds_df["target_end_date"] - pd.Timestamp(run_config.ref_date)).dt.days // 7
        ).astype(int)
        preds_df["target"] = np.where(preds_df["location"] == "nyc",
                                      "ILI ED visits pct", self.model_config.target)
        preds_df["output_type"] = self.model_config.output_type
        preds_df["reference_date"] = preds_df["reference_date"].dt.strftime("%Y-%m-%d")
        preds_df["target_end_date"] = preds_df["target_end_date"].dt.strftime("%Y-%m-%d")
        return preds_df[["reference_date", "target", "horizon", "target_end_date",
                         "location", "output_type", "output_type_id", "value"]]
