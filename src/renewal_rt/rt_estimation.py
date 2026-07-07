"""Rt estimation from a weekly incidence-proxy series.

The approach follows the result that EpiEstim-style instantaneous-Rt estimation is
closely equivalent to a GAM (penalized spline) regression of log-incidence
(Nash et al., Epidemics 2025). Concretely:

  1. Fit a penalized smoothing spline to log(incidence proxy) vs. time (in DAYS).
     Its first derivative is the instantaneous exponential growth rate r_t (per day).
  2. Map r_t -> R_t with the Euler-Lotka relation using a flu generation interval:
         R = 1 / sum_a g_a * exp(-r * a)
     where g_a is the discretized (daily) generation-interval pmf.

Because R is invariant to a constant multiplicative scaling of the signal, using the
NSSP percentage as an incidence proxy is valid to the extent the ED-visit denominator
is stable within a season; the smooth trend absorbs slow denominator drift.

Working at weekly resolution with a ~3-day generation interval, a direct weekly renewal
convolution is ill-posed (most GI mass falls at lag 0). We therefore use the Euler-Lotka
growth-rate form of the renewal equation for BOTH directions (r->R for estimation,
R->r for projection), which is resolution-independent and internally consistent.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.stats import gamma as gamma_dist


def discrete_generation_interval(mean_days: float = 3.0,
                                 sd_days: float = 1.5,
                                 max_days: int = 21) -> np.ndarray:
    """Discretized daily generation-interval pmf g[a], a = 1..max_days (g[0] = 0).

    Uses a gamma distribution matched to (mean, sd), integrated over daily bins and
    renormalized over a >= 1. Default (mean 3.0d, sd 1.5d) is a standard influenza
    generation/serial-interval choice.
    """
    shape = (mean_days / sd_days) ** 2
    scale = sd_days ** 2 / mean_days
    edges = np.arange(0, max_days + 1)  # 0,1,...,max_days
    cdf = gamma_dist.cdf(edges, a=shape, scale=scale)
    pmf = np.diff(cdf)              # mass in (a-1, a] for a = 1..max_days
    g = np.concatenate([[0.0], pmf])  # index 0..max_days; g[0] = 0
    g[1:] = g[1:] / g[1:].sum()       # renormalize over a >= 1
    return g


def euler_lotka_R_from_r(r, g: np.ndarray) -> np.ndarray:
    """Map growth rate r (per day) to reproduction number R via Euler-Lotka.

    R = 1 / sum_{a>=1} g_a exp(-r a).
    """
    r = np.clip(np.asarray(r, dtype=float), -2.0, 2.0)  # per-day; guard against spline-edge blowups
    a = np.arange(len(g))
    # denom[i] = sum_a g_a exp(-r_i a); broadcast over r
    denom = (g[None, :] * np.exp(-np.outer(r, a))).sum(axis=1)
    return 1.0 / denom


def euler_lotka_r_from_R(R, g: np.ndarray, r_lo: float = -0.5, r_hi: float = 0.5) -> np.ndarray:
    """Invert Euler-Lotka: find growth rate r (per day) such that R(r) = R.

    Monotone in r, solved by vectorized bisection on [r_lo, r_hi] (per-day growth;
    +/-0.5/day is far beyond any real weekly flu dynamics).
    """
    R = np.asarray(R, dtype=float)
    lo = np.full_like(R, r_lo)
    hi = np.full_like(R, r_hi)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        Rmid = euler_lotka_R_from_r(mid, g)
        too_low = Rmid < R
        lo = np.where(too_low, mid, lo)
        hi = np.where(too_low, hi, mid)
    return 0.5 * (lo + hi)


def spline_growth_rate(t_days: np.ndarray, log_inc: np.ndarray,
                       resid_sd: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """Penalized smoothing spline of log_inc vs. time (days); return (fitted, growth_rate).

    growth_rate = d/dt log_inc (per day) evaluated at the input points. `resid_sd` is the
    target residual standard deviation on the log scale (larger -> smoother); it sets the
    UnivariateSpline smoothing budget s = n * resid_sd^2.
    """
    t_days = np.asarray(t_days, dtype=float)
    log_inc = np.asarray(log_inc, dtype=float)
    n = len(t_days)
    if n < 5:
        # too few points for a cubic smoother: fall back to a local log-difference
        r = np.gradient(log_inc, t_days)
        return log_inc.copy(), r
    s = n * (resid_sd ** 2)
    spl = UnivariateSpline(t_days, log_inc, k=3, s=s)
    fitted = spl(t_days)
    r = spl.derivative()(t_days)
    return fitted, r


def make_link(kind: str = "logit", value_max: float = 100.0, floor: float = 0.01):
    """Return (forward, inverse) functions mapping percentage <-> working scale.

    - "log":   z = log(p + floor);         p = exp(z) - floor        (unbounded above)
    - "logit": z = logit(p / value_max);   p = value_max * sigmoid(z) (bounded to (0, value_max))

    For the small percentages that flu ED-visit shares take (p/100 << 1), logit(p/100) differs
    from log(p) only by a near-constant, so the growth rate dz/dt matches the incidence growth
    rate used by Euler-Lotka; the logit link only bites near the [0, value_max] boundary.
    """
    if kind == "log":
        def fwd(p):
            return np.log(np.clip(np.asarray(p, float), 0, None) + floor)

        def inv(z):
            return np.maximum(np.exp(np.asarray(z, float)) - floor, 0.0)
    elif kind == "logit":
        eps = 1e-4

        def fwd(p):
            q = np.clip(np.asarray(p, float) / value_max, eps, 1 - eps)
            return np.log(q / (1 - q))

        def inv(z):
            return value_max / (1.0 + np.exp(-np.asarray(z, float)))
    else:
        raise ValueError(f"unknown link '{kind}'")
    return fwd, inv


def estimate_rt_series(t_days: np.ndarray, inc: np.ndarray, g: np.ndarray,
                       link_fwd, link_inv, resid_sd: float = 0.15):
    """Full pipeline for one location: incidence proxy -> (fitted_inc, r_per_day, R).

    inc is the weekly percentage series (>= 0). The growth rate is the derivative of the
    working-scale (link) trend; returns arrays aligned to the inputs.
    """
    z = link_fwd(np.asarray(inc, dtype=float))
    fitted_z, r = spline_growth_rate(t_days, z, resid_sd=resid_sd)
    R = euler_lotka_R_from_r(r, g)
    return link_inv(fitted_z), r, R
