"""Black-76 pricing and a vectorised implied-volatility solver.

Everything is expressed in terms of the forward F, so dividends and rates
enter only through F and the discount factor. ``cp`` is +1 for calls, -1 for puts.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _broadcast(*xs):
    return np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in xs))


def _d1_d2(F, K, T, vol):
    s = vol * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * s * s) / s
    return d1, d1 - s


def black_price(F, K, T, vol, cp, df=1.0):
    """Black-76 option price (discounted by ``df``)."""
    F, K, T, vol, cp = _broadcast(F, K, T, vol, cp)
    d1, d2 = _d1_d2(F, K, T, vol)
    return df * cp * (F * ndtr(cp * d1) - K * ndtr(cp * d2))


def black_vega(F, K, T, vol, df=1.0):
    """Sensitivity of the price to a unit change in volatility."""
    F, K, T, vol = _broadcast(F, K, T, vol)
    d1, _ = _d1_d2(F, K, T, vol)
    return df * F * np.exp(-0.5 * d1 * d1) / _SQRT_2PI * np.sqrt(T)


def implied_vol(price, F, K, T, cp, df=1.0, lo=1e-4, hi=5.0, iters=100):
    """Invert Black-76 for volatility by vectorised bisection.

    Black price is strictly increasing in vol, so bisection is unconditionally
    stable. Returns NaN where the quote violates static no-arbitrage bounds
    (at/below intrinsic value, at/above the upper bound) or needs vol > ``hi``.
    """
    price, F, K, T, cp = _broadcast(price, F, K, T, cp)
    p = price / df
    intrinsic = np.maximum(cp * (F - K), 0.0)
    upper = np.where(cp > 0, F, K)
    valid = (p > intrinsic + 1e-12) & (p < upper) & (T > 0)

    lo_v = np.full(p.shape, lo)
    hi_v = np.full(p.shape, hi)
    valid &= black_price(F, K, T, hi_v, cp) >= p  # root lies inside [lo, hi]

    for _ in range(iters):
        mid = 0.5 * (lo_v + hi_v)
        too_low = black_price(F, K, T, mid, cp) < p
        lo_v = np.where(too_low, mid, lo_v)
        hi_v = np.where(too_low, hi_v, mid)

    out = np.where(valid, 0.5 * (lo_v + hi_v), np.nan)
    return out[()]
