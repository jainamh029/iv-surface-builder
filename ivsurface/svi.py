"""Raw-SVI calibration with static no-arbitrage constraints.

Raw SVI total variance (Gatheral 2004), with k = ln(K/F):

    w(k) = a + b * ( rho*(k - m) + sqrt((k - m)^2 + sigma^2) )

Calibration per expiry uses the quasi-explicit reduction of Zeliade (2009):
with y = (k-m)/sigma the model is *linear* in (a, d, c) where d = b*rho*sigma and
c = b*sigma, so for fixed (m, sigma) the fit is a small linear least-squares
problem. Only (m, sigma) are searched numerically. Linear constraints enforced
in the inner problem:

* w >= 0 everywhere            (a + c - |d| >= 0, a sufficient condition)
* Lee wing bound               (c + |d| <= 2 sigma  <=>  b(1+|rho|) <= 2)
* calendar spread vs the previous (shorter) expiry: w_T(k) >= w_{T-1}(k), on the
  overlap of the two slices' observed strike ranges

Butterfly arbitrage (negative risk-neutral density) is *not* linear, so it is
checked via Gatheral's g(k) >= 0 condition, and a constrained 5-parameter
repair is attempted when it fails.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

G_TOL = -1e-6
CAL_MARGIN = 0.05                    # calendar enforced on the overlap of both slices' data, +/- this margin
BF_MARGIN = 0.10                     # butterfly checked on [k_min - margin, k_max + margin]


@dataclass(frozen=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def w(self, k):
        x = np.asarray(k, float) - self.m
        return self.a + self.b * (self.rho * x + np.sqrt(x * x + self.sigma ** 2))

    def dw(self, k):
        x = np.asarray(k, float) - self.m
        return self.b * (self.rho + x / np.sqrt(x * x + self.sigma ** 2))

    def d2w(self, k):
        x = np.asarray(k, float) - self.m
        return self.b * self.sigma ** 2 / (x * x + self.sigma ** 2) ** 1.5

    def g(self, k):
        """Gatheral's butterfly function; the density is non-negative iff g >= 0."""
        k = np.asarray(k, float)
        w = np.maximum(self.w(k), 1e-12)
        wp, wpp = self.dw(k), self.d2w(k)
        return (1 - k * wp / (2 * w)) ** 2 - wp ** 2 / 4 * (1 / w + 0.25) + wpp / 2

    def to_dict(self):
        return asdict(self)


@dataclass
class SliceFit:
    expiry: str
    T: float
    F: float
    params: SVIParams
    n: int
    rmse_vol: float        # unweighted RMSE of fitted vs market IV (decimal, 0.01 = 1 vol pt)
    max_err_vol: float
    min_g: float           # min of the butterfly function on the check grid
    butterfly_ok: bool
    calendar_ok: bool      # w_T >= w_{T-1} on the check grid
    repaired: bool = False
    k_min: float = 0.0     # observed log-moneyness range; outside it the fit is extrapolation
    k_max: float = 0.0


# --------------------------------------------------------------------------
# inner linear problem
# --------------------------------------------------------------------------
_A_BASE = np.array([[0, 0, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1],
                    [0, 1, 1], [1, -1, 1], [1, 1, 1]], float)


def _inner(k, w, om, m, sig, wscale, cal):
    """Solve for (a, d, c)/wscale at fixed (m, sigma). Returns (theta, sse) or (None, inf)."""
    y = (k - m) / sig
    r = np.sqrt(y * y + 1.0)
    X = np.column_stack([np.ones_like(y), y, r])
    ws = w / wscale
    sw = np.sqrt(om)
    Xw, yw = X * sw[:, None], ws * sw

    A = _A_BASE
    bvec = np.array([0, -2 * sig / wscale, -2 * sig / wscale, 0, 0, 0, 0], float)
    if cal is not None:
        kg, wprev = cal
        yg = (kg - m) / sig
        A = np.vstack([A, np.column_stack([np.ones_like(yg), yg, np.sqrt(yg * yg + 1.0)])])
        bvec = np.concatenate([bvec, wprev / wscale])

    theta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
    if np.min(A @ theta - bvec) < -1e-10:
        res = minimize(lambda t: np.sum((Xw @ t - yw) ** 2), theta,
                       jac=lambda t: 2 * Xw.T @ (Xw @ t - yw), method="SLSQP",
                       constraints=[{"type": "ineq", "fun": lambda t: A @ t - bvec,
                                     "jac": lambda t: A}],
                       options={"maxiter": 200, "ftol": 1e-12})
        theta = res.x
        if np.min(A @ theta - bvec) < -1e-7:
            return None, np.inf
    return theta, float(np.sum((Xw @ theta - yw) ** 2))


def _to_params(theta, m, sig, wscale) -> SVIParams:
    a, d, c = theta * wscale
    b = c / sig
    rho = float(np.clip(d / c, -0.9999, 0.9999)) if c > 1e-14 else 0.0
    return SVIParams(float(a), float(b), rho, float(m), float(sig))


# --------------------------------------------------------------------------
# butterfly repair (nonlinear, 5 parameters)
# --------------------------------------------------------------------------
def _repair_butterfly(p0, k, w, om, wscale, kb, cal):
    def unpack(x):
        return SVIParams(x[0] * wscale, x[1] * wscale, x[2], x[3], x[4])

    def obj(x):
        return np.sum(om * (unpack(x).w(k) / wscale - w / wscale) ** 2)

    def cons(x):
        p = unpack(x)
        out = [p.g(kb), [2 - p.b * (1 + p.rho), 2 - p.b * (1 - p.rho),
                         p.a + p.b * p.sigma * np.sqrt(1 - p.rho ** 2)]]
        if cal is not None:
            out.append((p.w(cal[0]) - cal[1]) / wscale)
        return np.concatenate([np.atleast_1d(o) for o in out])

    span = k.max() - k.min()
    x0 = [p0.a / wscale, p0.b / wscale, p0.rho, p0.m, p0.sigma]
    bounds = [(-2, 5), (0, 50), (-0.999, 0.999), (k.min() - span, k.max() + span), (1e-3, 2.0)]
    res = minimize(obj, x0, method="SLSQP", bounds=bounds,
                   constraints=[{"type": "ineq", "fun": cons}],
                   options={"maxiter": 300, "ftol": 1e-12})
    cand = unpack(res.x)
    return cand if np.min(cons(res.x)) > -1e-7 else None


def calendar_grid(k_min_a, k_max_a, k_min_b, k_max_b, n=61):
    """Grid over the overlap of two slices' observed ranges (+ margin), or None if they don't overlap.

    Outside the overlap one slice is pure extrapolation, so constraining against it would
    distort the slice that actually has data there.
    """
    lo, hi = max(k_min_a, k_min_b) - CAL_MARGIN, min(k_max_a, k_max_b) + CAL_MARGIN
    return np.linspace(lo, hi, n) if lo < hi else None


def _bf_grid(k):
    return np.linspace(k.min() - BF_MARGIN, k.max() + BF_MARGIN, 80)


def _rmse(p: SVIParams, k, w, T):
    """RMSE of fitted vs market implied vol (decimal), given total variances w at maturity T."""
    return float(np.sqrt(np.mean((np.sqrt(np.maximum(p.w(k), 1e-12) / T) - np.sqrt(w / T)) ** 2)))


# --------------------------------------------------------------------------
# slice and surface calibration
# --------------------------------------------------------------------------
def fit_slice(k, w, om, T: float, cal=None) -> tuple[SVIParams, bool, bool]:
    """Fit one expiry. ``cal`` = (k_grid, w_prev) imposes w(k_grid) >= w_prev.

    Returns (params, repaired, calendar_constrained).
    """
    k, w, om = (np.asarray(x, float) for x in (k, w, om))
    om = om / om.mean()          # absolute weight scale is irrelevant; keep the optimiser well-scaled
    wscale = float(w.max())
    span = max(k.max() - k.min(), 0.1)

    def search(cal_):
        def obj(x):
            return _inner(k, w, om, x[0], x[1], wscale, cal_)[1] / max(np.sum(om * (w / wscale) ** 2), 1e-300)

        starts = [(m0, s0) for m0 in np.linspace(k.min(), k.max(), 5) for s0 in (0.03, 0.08, 0.2, 0.5)]
        scored = sorted(starts, key=lambda s: obj(s))[:3]
        bounds = [(k.min() - 0.5 * span, k.max() + 0.5 * span), (2e-3, 1.5)]
        best = min((minimize(obj, s, method="Nelder-Mead", bounds=bounds,
                             options={"xatol": 1e-5, "fatol": 1e-12, "maxiter": 300})
                    for s in scored), key=lambda r: r.fun)
        return best

    best = search(cal)
    constrained = cal is not None
    if not np.isfinite(best.fun) and cal is not None:      # calendar infeasible: fit freely
        best, constrained = search(None), False
        cal = None
    theta, _ = _inner(k, w, om, best.x[0], best.x[1], wscale, cal)
    if theta is None:
        raise RuntimeError("SVI slice calibration failed")
    params = _to_params(theta, best.x[0], best.x[1], wscale)

    repaired = False
    kb = _bf_grid(k)
    if params.g(kb).min() < G_TOL:
        fixed = _repair_butterfly(params, k, w, om, wscale, kb, cal)
        # accept only if the repaired curve still describes the market about as well
        r0 = _rmse(params, k, w, T)
        if fixed is not None and _rmse(fixed, k, w, T) <= max(2 * r0, r0 + 3e-3):
            params, repaired = fixed, True
    return params, repaired, constrained


def calibrate(quotes: pd.DataFrame) -> list[SliceFit]:
    """Fit every expiry in ``quotes`` (output of ``prepare``), shortest first."""
    fits: list[SliceFit] = []
    prev: SliceFit | None = None
    for expiry, g in quotes.sort_values("T").groupby("expiry", sort=False):
        k, iv, T = g["k"].values, g["iv"].values, float(g["T"].iloc[0])
        w = iv ** 2 * T
        om = g["weight"].values / (2 * iv * T) ** 2     # inverse-variance in total-variance units
        cal = None
        if prev is not None:
            kg = calendar_grid(prev.k_min, prev.k_max, k.min(), k.max())
            cal = None if kg is None else (kg, prev.params.w(kg))
        params, repaired, constrained = fit_slice(k, w, om, T, cal)

        iv_fit = np.sqrt(np.maximum(params.w(k), 1e-12) / T)
        err = iv_fit - iv
        min_g = float(params.g(_bf_grid(k)).min())
        cal_ok = True if cal is None else bool(np.min(params.w(cal[0]) - cal[1]) >= -1e-9)
        fits.append(SliceFit(expiry=str(expiry), T=T, F=float(g["F"].iloc[0]), params=params,
                             n=len(g), rmse_vol=float(np.sqrt(np.mean(err ** 2))),
                             max_err_vol=float(np.abs(err).max()), min_g=min_g,
                             butterfly_ok=min_g >= G_TOL, calendar_ok=cal_ok, repaired=repaired,
                             k_min=float(k.min()), k_max=float(k.max())))
        prev = fits[-1]
    return fits
