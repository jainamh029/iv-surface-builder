"""Option-chain acquisition: Yahoo Finance (yfinance) and CSV round-tripping.

A raw chain is a long DataFrame with one row per contract and columns
``expiry, T, type, strike, bid, ask, last, volume, openInterest, yahoo_iv, spot, asof``.
"""
from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

NY = ZoneInfo("America/New_York")
SECONDS_PER_YEAR = 365.0 * 86400.0
RAW_COLUMNS = ["expiry", "T", "type", "strike", "bid", "ask", "last",
               "volume", "openInterest", "yahoo_iv", "spot", "asof"]


def year_fraction(expiry: str | pd.Timestamp, asof: pd.Timestamp) -> float:
    """ACT/365 time to a 16:00 New York expiry."""
    exp = pd.Timestamp(expiry).tz_localize(None).replace(hour=16)
    exp = exp.tz_localize(NY)
    return max((exp - asof).total_seconds(), 0.0) / SECONDS_PER_YEAR


def select_expiries(days: list[float], n: int) -> list[int]:
    """Pick ``n`` indices whose days-to-expiry are roughly log-spaced.

    Index options list weeklies/dailies densely near the front; taking the
    first ``n`` would leave the long end of the surface empty.
    """
    if len(days) <= n:
        return list(range(len(days)))
    logd = np.log(np.maximum(days, 0.5))
    targets = np.linspace(logd.min(), logd.max(), n)
    chosen = {int(np.argmin(np.abs(logd - t))) for t in targets}
    return sorted(chosen)


def _spot(tk) -> float:
    try:
        px = float(tk.fast_info["last_price"])
        if px > 0:
            return px
    except Exception:  # noqa: BLE001 - yfinance raises assorted errors
        pass
    hist = tk.history(period="5d")
    return float(hist["Close"].dropna().iloc[-1])


def fetch_chain(symbol: str, max_expiries: int = 12, min_days: float = 3,
                max_days: float = 400) -> pd.DataFrame:
    """Download a live chain for ``symbol`` from Yahoo Finance.

    Equity/ETF options (SPY, QQQ, ...) are American-style. For European-style
    contracts use an index such as ``^SPX``.
    """
    import yfinance as yf

    tk = yf.Ticker(symbol)
    asof = pd.Timestamp.now(tz=NY)
    spot = _spot(tk)

    listed = list(tk.options)
    if not listed:
        raise RuntimeError(f"Yahoo returned no option expiries for {symbol!r}")
    days = [year_fraction(e, asof) * 365.0 for e in listed]
    keep = [i for i, d in enumerate(days) if min_days <= d <= max_days]
    if not keep:
        raise RuntimeError(f"No expiries within [{min_days}, {max_days}] days")
    picked = [keep[i] for i in select_expiries([days[i] for i in keep], max_expiries)]

    frames = []
    for i in picked:
        chain = tk.option_chain(listed[i])
        for typ, side in (("C", chain.calls), ("P", chain.puts)):
            frames.append(side.assign(type=typ, expiry=listed[i], T=days[i] / 365.0))
    raw = pd.concat(frames, ignore_index=True)

    raw = raw.rename(columns={"lastPrice": "last", "impliedVolatility": "yahoo_iv"})
    raw["spot"] = spot
    raw["asof"] = asof.isoformat()
    for col in ("bid", "ask", "last", "volume", "openInterest", "yahoo_iv"):
        raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0.0)
    return raw[RAW_COLUMNS].sort_values(["T", "type", "strike"]).reset_index(drop=True)


def save_chain(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_chain(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df[RAW_COLUMNS]
