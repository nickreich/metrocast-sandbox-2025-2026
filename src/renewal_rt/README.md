# renewal_rt — renewal / Rt sample-trajectory model

A semi-mechanistic forecast model for the flu-metrocast hub. It estimates the effective
reproduction number **Rt** from the weekly NSSP "% ED visits due to flu" signal, forecasts
Rt forward from a pooled seasonal climatology, and projects the percentage forward with a
renewal-consistent growth step to produce **full-season sample trajectories** (the hub's
2026-27 sample-output format). It can also emit standard **quantiles** for comparison with
the other hub models.

Built on the `idmodels` / `iddata` class structure (subclasses `idmodels.IDModel`; reads the
hub target series through a custom `iddata` `DataSource`). **v1 is state-level.**

## Notation

For location ℓ and week ending `t`:

| Symbol | Meaning |
|--------|---------|
| **`p`** (`p_{ℓ,t}`) | the **forecast target itself**: the weekly *percentage of ED visits due to flu* (0–100), from NSSP. This is what the hub scores. |
| `z` | working-scale level, `z = log(p)` (log link; `logit(p/100)` optional). All dynamics run on `z`. |
| `r_t` | instantaneous exponential **growth rate** (per day) `= dz/dt`. |
| `g_a` | discretized **daily generation-interval** pmf (gamma). |
| `R_t` | instantaneous **reproduction number**, `R = 1 / Σ_a g_a e^{-r a}` (Euler–Lotka). |
| `clim_R(w)` | **pooled** seasonal climatology of `log R` by season-week `w` (see pooling note). |
| `b_ℓ` | location-specific **Rt** offset (`loc_offset`, shrunk by `loc_offset_shrink`). |
| `e_t` | AR(1) deviation of `log R` from `clim_R + b_ℓ`. |
| `clim_lev(w)` | **pooled** seasonal climatology of the **level** `z` by season-week. |
| `a_ℓ` | location-specific **level** offset (`loc_level_offset`, scaled by `level_offset_shrink`). |
| `κ` | `level_revert_kappa` — weekly strength of the level's reversion to its seasonal target. |
| `φ`, `σ` | AR(1) persistence and innovation sd of the Rt deviations. |

## Method

1. **Rt estimation (GAM-equivalent).** For each location, fit a penalized smoothing spline to
   `z = log(p)` vs. time; its derivative is the instantaneous growth rate `r_t`. This follows the
   result that EpiEstim-style instantaneous-Rt estimation is closely equivalent to a GAM / spline
   regression of log-incidence [4, 5]. Because the target is a percentage, not counts, we rely on
   `R` being invariant to a constant multiplicative scaling of the signal (the smooth trend absorbs
   slow drift in the ED-visit denominator).
2. **r → Rt via Euler–Lotka** [2]. `R = 1 / Σ_a g_a e^{-r a}`, with `g` a discretized daily flu
   generation interval (gamma, mean 3.0 d, sd 1.5 d [3]). At weekly resolution a direct renewal
   convolution is ill-posed (most GI mass sits at lag 0), so we use this growth-rate form of the
   renewal equation for **both** estimation (`r → R`) and projection (`R → r`); using the same `g`
   in both directions makes the forecast nearly insensitive to the GI (see Status).
3. **Rt forecast (v1: seasonal + AR).** Sample the Rt trajectory as
   `log R_{t+1} = clim_R(w) + b_ℓ + e_{t+1}`, `e_{t+1} = φ e_t + η`, `η ~ N(0, σ²)`.
   The seasonal climatology and AR parameters are **pooled across all locations** (history is thin,
   ~3 seasons of state NSSP); `b_ℓ` restores each location's average Rt level. This is the
   "consistent-across-locations" reversion for **Rt**.
4. **Renewal projection with saturation.** Step the level:
   `z_{t+1} = z_t + 7·r(R_{t+1}) − κ·(z_t − [clim_lev(w) + a_ℓ]) + ξ`, `ξ ~ N(0, σ_obs²)`, then
   `p = exp(z)`. The `−κ(…)` term is an **Ornstein–Uhlenbeck** mean reversion [6] of the level
   toward its seasonal target — a susceptible-depletion / saturation proxy that bounds long-horizon
   variance. The reversion **target is location-specific in magnitude** (`clim_lev` supplies the
   pooled seasonal *shape*; `a_ℓ`, fully location-specific by default, supplies that location's own
   seasonal *level*), because the seasonal percentage differs markedly across locations even though
   the Rt seasonality is shared. Each sample index is one coherent full-season trajectory.

**Pooling note.** Both climatologies (`clim_R`, `clim_lev`) are computed by pooling *all* locations
and seasons at each season-week (a simple mean, smoothed over a rolling window). Location specificity
enters only through the offsets `b_ℓ` (Rt, shrunk 0.5) and `a_ℓ` (level, shrink 1.0 = fully
location-specific).

## Files

| File | Purpose |
|------|---------|
| `rt_estimation.py` | generation interval, spline growth rate, Euler–Lotka `r↔R`, link functions |
| `metrocast_source.py` | `MetrocastTargetSource` — hub target series as an `iddata` `DataSource` |
| `renewal_model.py` | `RenewalRtModel` (subclasses `idmodels.IDModel`) + `RenewalModelConfig` |
| `main.py` | CLI runner |
| `plot_forecasts.py` | diagnostic trajectory plot (% ED visits) vs. observed |
| `plot_rt.py` | diagnostic Rt-by-season-week plot: history vs. current estimate vs. forecast |
| `backtest.py` | retrospective sweep: all locations, biweekly reference dates over a season → forecast CSVs + all-location trajectory grids |
| `score.py` | WIS + interval-coverage scoring (versioned real-time data) vs. all flu-metrocast hub baselines |
| `_stubs.py` | stubs idmodels' unused deps (lightgbm/sarix/timeseriesutils/tqdm) at import |

## Setup

Uses the **latest local `../idmodels` and `../iddata` source** in a dedicated venv (kept separate
from the gbqr models, which pin the older released idmodels):

```bash
uv venv src/renewal_rt/.venv --python 3.11
VENV=src/renewal_rt/.venv/bin/python
uv pip install --python $VENV -e ../iddata   --no-deps
uv pip install --python $VENV -e ../idmodels --no-deps
uv pip install --python $VENV -r src/renewal_rt/requirements.txt
```

`lightgbm`, `sarix`, `timeseriesutils`, and `tqdm` are intentionally **not** installed — the
renewal model never uses them, and `_stubs.py` registers minimal stand-ins so `import idmodels`
(which eagerly imports GBQR/SARIX) succeeds without them.

## Usage (from the repo root)

```bash
# 100 full-season sample trajectories (2026-27 hub format)
PYTHONPATH=$(pwd) src/renewal_rt/.venv/bin/python src/renewal_rt/main.py \
    --today_date 2026-01-28 --use_local_mchub --max_horizon 26 --output_type sample

# hub-comparable quantiles (writes to a separate epiENGAGE-renewal_rt_q/ dir)
PYTHONPATH=$(pwd) src/renewal_rt/.venv/bin/python src/renewal_rt/main.py \
    --today_date 2026-01-28 --use_local_mchub --output_type quantile --model_name renewal_rt_q

# diagnostic plot
PYTHONPATH=$(pwd) src/renewal_rt/.venv/bin/python src/renewal_rt/plot_forecasts.py \
    --forecast model-output/epiENGAGE-renewal_rt/2026-01-31-epiENGAGE-renewal_rt.csv

# retrospective backtest: ALL 77 locations, biweekly reference dates across the 2025/26 season
# (writes per-date forecast CSVs + all-location trajectory grids under plots/backtest/)
PYTHONPATH=$(pwd) src/renewal_rt/.venv/bin/python src/renewal_rt/backtest.py \
    --first_ref 2025-11-08 --last_ref 2026-05-16 --season_end 2026-06-27
```

For all-location runs (including NYC, whose target is ILI) set `include_nyc=True` on the config and
pass every location slug in `RunConfig.states`; `backtest.py` does both.

```bash
# WIS + coverage scoring vs. every hub baseline, using real-time versioned data
PYTHONPATH=$(pwd) src/renewal_rt/.venv/bin/python src/renewal_rt/score.py
```

`versioned=True` reconstructs the real-time data vintage at each reference date from the hub's
`time-series.csv` (which only extends to ~2024-08), splicing in stable finalized history from
`latest-data.csv` for older seasons. Scoring is against `oracle-output.csv` on cells common to each
pair of models.

Output: `model-output/epiENGAGE-<model_name>/{ref_date}-epiENGAGE-<model_name>.csv`, columns
`reference_date, target, horizon, target_end_date, location, output_type, output_type_id, value`.

## Parameters (`RenewalModelConfig`)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `include_nyc` | `False` | also load NYC's ILI target so all 77 locations can be run (else flu-only, 76 locations). |
| `versioned` | `False` | use real-time data vintages (hub `time-series.csv` + finalized older history) instead of `latest-data.csv`. |
| `transform` | `"log"` | working scale for `z`; `"logit"` bounds predictions to `(0, value_max)`. |
| `gi_mean_days`, `gi_sd_days`, `gi_max_days` | 3.0, 1.5, 21 | daily generation-interval gamma (mean, sd, truncation) [3]. |
| `resid_sd` | 0.15 | smoothing-spline residual sd on the working scale (larger = smoother growth rate). |
| `in_season_min`/`max` | 10 / 45 | season-week window (week-30 origin) used to build climatologies. |
| `clim_smooth_window` | 5 | rolling window (season-weeks) smoothing both climatologies. |
| `loc_offset_shrink` | 0.5 | shrinkage of the location **Rt** offset `b_ℓ` toward the pooled mean. |
| `level_offset_shrink` | 1.0 | scaling of the location **level** offset `a_ℓ` (1.0 = fully location-specific magnitude). |
| `ar_phi_max` | 0.8 | cap on AR(1) persistence `φ` of Rt deviations. |
| `level_revert_kappa` (`κ`) | 0.15 | weekly OU reversion of the level to its seasonal target; lower = slower reversion (half-life ≈ 2.4 / 4.3 / 6.6 wk at 0.25 / 0.15 / 0.10). |
| `obs_sigma`, `nowcast_sigma` | 0.02, 0.08 | per-step process noise and anchor (nowcast/revision) noise (working scale). |
| `output_type`, `n_samples` | `"sample"`, 100 | emit 100 trajectories, or hub-comparable quantiles. |

## Status & limitations (v1)

- **State level only** (13 states). HSA / NC-region / NYC (ILI) geographies are not yet fit; the
  `MetrocastTargetSource` returns them but the runner restricts to states.
- **Thin history** (~3 seasons of state NSSP) → seasonal Rt climatology is pooled across all
  locations and is rough. In hindcasts from before the peak the model captures **timing** well but
  tends to be **conservative on peak height** (tuned by `level_revert_kappa`).
- **Generation interval is a fixed assumption but results are robust to it.** Sweeping the GI mean
  from 2.3 to 4.5 days changes median forecasts by a median of 0.2% (max ~4%), because the same `g`
  is used for `r → R` and `R → r` and nearly cancels. So the fixed GI is not a material source of
  error here.
- **Weather covariates deferred** (planned v2). Weather forecasts only extend ~2 weeks while
  trajectories run the full season, so weather mainly helps the near horizons.

## Next steps

- **Borrow longer history for the seasonal climatology** from other `iddata` sources. NSSP flu-%
  only goes back ~3 seasons, but ILINet (weighted ILI %, back to 1997) and FluSurv-NET
  (hospitalization rates, back to ~2003) cover ~20+ seasons. Because Rt is scale-invariant, the
  **seasonal Rt *shape*** estimated from these long series (at state / HHS-region level) can serve
  as a prior/climatology for the NSSP model; only the *level* climatology must stay NSSP-specific.
  The existing `data_loader` already exposes ILINet/FluSurv-NET/NHSN loaders. *(Not yet implemented.)*
- Add weather / climate covariates to the Rt model (short-horizon lift).
- Consider approaches to sharpen the relationship between Rt and incidence, perhaps by including an exponent. 
- Add school holiday timings (varying by state?) as a covariate.
- Extend to HSA and NC-region geographies (custom source already returns them) and NYC ILI.
- Hierarchical partial pooling; season-varying `κ` (weak on the rise, strong post-peak) to fix the
  peak-height conservatism; retrospective WIS/energy-score evaluation vs. the gbqr baselines.

## References

1. Cori A, Ferguson NM, Fraser C, Cauchemez S. *A new framework and software to estimate
   time-varying reproduction numbers during epidemics.* Am J Epidemiol. 2013;178(9):1505–1512.
   doi:10.1093/aje/kwt133 — instantaneous-Rt framework (EpiEstim).
2. Wallinga J, Lipsitch M. *How generation intervals shape the relationship between growth rates
   and reproductive numbers.* Proc R Soc B. 2007;274(1609):599–604. doi:10.1098/rspb.2006.3754 —
   the Euler–Lotka `r ↔ R` mapping used here.
3. Cowling BJ, Fang VJ, Riley S, Peiris JSM, Leung GM. *Estimation of the serial interval of
   influenza.* Epidemiology. 2009;20(3):344–347. doi:10.1097/EDE.0b013e31819d1092 — influenza
   serial/generation interval (mean ≈ 2.6–3.6 d), source for the GI defaults.
4. EpiEstim ≈ GAM / spline-regression equivalence. *Epidemics* (2025).
   https://www.sciencedirect.com/science/article/pii/S1755436525000453
   *(exact author list to be confirmed from the article.)*
5. Pircalabelu E. *A spline-based time-varying reproduction number for modelling epidemiological
   outbreaks.* J R Stat Soc Ser C. 2023;72(5). doi:10.1093/jrsssc/qlad027 — supporting reference
   for spline/GAM-based Rt.
6. Uhlenbeck GE, Ornstein LS. *On the theory of the Brownian motion.* Phys Rev. 1930;36:823 —
   the mean-reverting (OU) process used for the level-reversion term.
