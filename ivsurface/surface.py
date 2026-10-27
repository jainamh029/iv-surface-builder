"""Queryable implied-vol surface assembled from per-expiry SVI slices.

Across maturities, total variance is interpolated *linearly in T at fixed
log-moneyness k*. If every slice is ordered (w_T(k) non-decreasing in T), the
interpolant is too, so calendar no-arbitrage carries over from the slices.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .svi import SliceFit, SVIParams, calendar_grid


class IVSurface:
    def __init__(self, fits: list[SliceFit], spot: float, asof: str, rate: float):
        if not fits:
            raise ValueError("no calibrated slices")
        self.fits = sorted(fits, key=lambda f: f.T)
        self.spot, self.asof, self.rate = spot, asof, rate
        self.T = np.array([f.T for f in self.fits])
        self._logF = np.log([f.F for f in self.fits])

    # -- evaluation ---------------------------------------------------------
    def forward(self, T):
        return np.exp(np.interp(T, self.T, self._logF))

    def total_variance(self, k, T: float) -> np.ndarray:
        """w(k, T) for scalar T; flat-vol scaling before the first slice."""
        k = np.asarray(k, float)
        T = float(T)
        if T > self.T[-1] + 1e-12:
            raise ValueError(f"T={T:.4f} is beyond the last calibrated expiry ({self.T[-1]:.4f})")
        if T <= self.T[0]:
            return self.fits[0].params.w(k) * T / self.T[0]
        j = int(np.searchsorted(self.T, T))
        lo, hi = self.fits[j - 1], self.fits[j]
        lam = (T - lo.T) / (hi.T - lo.T)
        return (1 - lam) * lo.params.w(k) + lam * hi.params.w(k)

    def iv(self, k, T: float) -> np.ndarray:
        return np.sqrt(np.maximum(self.total_variance(k, T), 0.0) / T)

    def iv_strike(self, K, T: float) -> np.ndarray:
        return self.iv(np.log(np.asarray(K, float) / self.forward(T)), T)

    def atm_vol(self, T: float) -> float:
        return float(self.iv(0.0, T))

    def data_range(self, T: float) -> tuple[float, float]:
        """Observed log-moneyness range at maturity T (interpolated between slices)."""
        lo = np.interp(T, self.T, [f.k_min for f in self.fits])
        hi = np.interp(T, self.T, [f.k_max for f in self.fits])
        return float(lo), float(hi)

    def grid(self, n_k: int = 101, n_T: int = 60, mask_extrapolation: bool = True):
        """Rectangular (k, T, IV) grid; points outside the observed strike range are NaN."""
        ks = np.array([f.k_min for f in self.fits] + [f.k_max for f in self.fits])
        k = np.linspace(np.percentile(ks, 10), np.percentile(ks, 90), n_k)
        T = np.linspace(self.T[0], self.T[-1], n_T)
        Z = np.vstack([self.iv(k, t) for t in T])
        if mask_extrapolation:
            for i, t in enumerate(T):
                lo, hi = self.data_range(t)
                Z[i, (k < lo) | (k > hi)] = np.nan
        return k, T, Z

    # -- diagnostics --------------------------------------------------------
    def arbitrage_report(self) -> pd.DataFrame:
        rows = [dict(expiry=f.expiry, days=f.T * 365, n=f.n, rmse_vol_pts=100 * f.rmse_vol,
                     max_err_vol_pts=100 * f.max_err_vol, min_g=f.min_g,
                     butterfly_ok=f.butterfly_ok, calendar_ok=f.calendar_ok,
                     repaired=f.repaired, **{k: v for k, v in f.params.to_dict().items()})
                for f in self.fits]
        return pd.DataFrame(rows)

    def calendar_violations(self, n_k: int = 200) -> int:
        """Count (k, adjacent-slice) points where total variance falls with maturity.

        Checked on the overlap of adjacent slices' observed ranges (+ margin). Between slices the
        surface interpolates linearly in T, so ordered neighbours imply an ordered surface.
        """
        bad = 0
        for lo, hi in zip(self.fits[:-1], self.fits[1:]):
            kg = calendar_grid(lo.k_min, lo.k_max, hi.k_min, hi.k_max, n_k)
            if kg is not None:
                bad += int(np.sum(hi.params.w(kg) - lo.params.w(kg) < -1e-9))
        return bad

    # -- persistence --------------------------------------------------------
    def to_json(self, path: str | Path) -> None:
        payload = dict(spot=self.spot, asof=self.asof, rate=self.rate,
                       slices=[dict(asdict(f), params=f.params.to_dict()) for f in self.fits])
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "IVSurface":
        d = json.loads(Path(path).read_text())
        fits = [SliceFit(**{**s, "params": SVIParams(**s["params"])}) for s in d["slices"]]
        return cls(fits, d["spot"], d["asof"], d["rate"])
