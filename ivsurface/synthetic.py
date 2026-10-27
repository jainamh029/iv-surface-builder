"""Synthetic option chain generated from a known, arbitrage-free SSVI surface.

Lets the pipeline run offline and, more importantly, lets tests check that the
calibrated surface recovers the *true* volatilities despite bid/ask noise and
deliberately injected bad quotes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .black import black_price
from .data import NY, RAW_COLUMNS, year_fraction

DEFAULT_DAYS = (7, 14, 21, 30, 45, 60, 90, 120, 180, 270, 365)


def atm_vol(T):
    """Upward-sloping ATM term structure."""
    return 0.15 + 0.05 * (1.0 - np.exp(-2.0 * np.asarray(T, float)))


def ssvi_total_variance(k, T, rho=-0.7, eta=1.0, gamma=0.45):
    """Gatheral-Jacquier SSVI total variance w(k, T) (arbitrage-free for these params)."""
    k, T = np.broadcast_arrays(np.asarray(k, float), np.asarray(T, float))
    theta = atm_vol(T) ** 2 * T
    phi = eta / (theta ** gamma * (1.0 + theta) ** (1.0 - gamma))
    pk = phi * k
    return 0.5 * theta * (1.0 + rho * pk + np.sqrt((pk + rho) ** 2 + 1.0 - rho ** 2))


def true_iv(k, T):
    T = np.asarray(T, float)
    return np.sqrt(ssvi_total_variance(k, T) / T)


def make_chain(spot=500.0, rate=0.04, div_yield=0.013, days=DEFAULT_DAYS, seed=7,
               bad_quote_frac=0.04) -> pd.DataFrame:
    """Build a noisy, tick-rounded chain with injected zero-bid/crossed/outlier quotes."""
    rng = np.random.default_rng(seed)
    asof = pd.Timestamp("2026-10-26 12:00", tz=NY)
    strikes = np.arange(np.round(0.7 * spot / 5) * 5, 1.3 * spot + 1, 5.0)

    rows = []
    for d in days:
        expiry = (asof + pd.Timedelta(days=d)).date().isoformat()
        T = year_fraction(expiry, asof)
        F = spot * np.exp((rate - div_yield) * T)
        df = np.exp(-rate * T)
        for typ, cp in (("C", 1.0), ("P", -1.0)):
            vol = true_iv(np.log(strikes / F), T)
            mid = black_price(F, strikes, T, vol, cp, df)
            half = 0.01 + 0.015 * mid
            mid = np.maximum(mid + rng.normal(0.0, 0.3, strikes.size) * half, 0.0)
            bid = np.maximum(np.round(mid - half, 2), 0.0)
            ask = np.round(mid + half, 2)
            for K, b, a in zip(strikes, bid, ask):
                rows.append(dict(expiry=expiry, T=T, type=typ, strike=K, bid=b, ask=a,
                                 last=round((a + b) / 2, 2),
                                 volume=int(rng.integers(0, 2000)),
                                 openInterest=int(rng.integers(0, 20000)),
                                 yahoo_iv=0.0, spot=spot, asof=asof.isoformat()))
    df_ = pd.DataFrame(rows)

    # inject bad quotes the cleaning stage is supposed to catch
    n = len(df_)
    bad = rng.choice(n, size=int(bad_quote_frac * n), replace=False)
    kind = rng.integers(0, 3, size=bad.size)
    for idx, kd in zip(bad, kind):
        b, a = float(df_.at[idx, "bid"]), float(df_.at[idx, "ask"])
        if kd == 0:
            df_.at[idx, "bid"] = 0.0                                    # one-sided
        elif kd == 1:
            df_.at[idx, "bid"], df_.at[idx, "ask"] = a + 0.05, b        # crossed
        else:
            m = (a + b) / 2 * 1.6 + 0.10                                # stale outlier
            df_.at[idx, "bid"], df_.at[idx, "ask"] = round(m - 0.02, 2), round(m + 0.02, 2)
    return df_[RAW_COLUMNS]
