"""Raw chain -> clean implied-vol quotes.

Pipeline: validate quotes -> forward per expiry from put-call parity -> keep
only OTM options -> invert Black-76 -> drop IV outliers. Every dropped row
carries a ``reason`` so you can see exactly what was filtered and why.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .black import black_vega, implied_vol

REASONS = ["crossed", "no_quote", "wide_spread", "low_price", "short_dated",
           "low_interest", "no_forward", "not_otm", "far_otm", "iv_failed",
           "iv_outlier", "thin_slice"]


@dataclass
class PrepConfig:
    rate: float = 0.04            # flat continuous rate used only for discounting
    min_days: float = 7.0         # very short expiries are dominated by microstructure
    max_abs_k: float = 0.5        # drop |ln(K/F)| beyond this
    max_rel_spread: float = 0.5   # (ask-bid)/mid
    min_price: float = 0.05       # mids below ~a few ticks carry no information
    min_open_interest: int = 0
    use_last: bool = False        # fall back to last trade when there is no two-sided quote
    min_pairs: int = 3            # call/put pairs needed to estimate a forward
    n_parity_strikes: int = 10    # strikes nearest spot used for the forward
    hampel_window: int = 5
    hampel_nsigma: float = 4.0
    hampel_floor: float = 0.02    # never flag residuals smaller than 2 vol points
    min_points: int = 6           # per-expiry minimum to attempt a smile fit


@dataclass
class PrepResult:
    quotes: pd.DataFrame      # clean OTM quotes with iv, k, weight
    audit: pd.DataFrame       # every raw row with its filter reason ('ok' if kept)
    forwards: pd.DataFrame    # per-expiry forward, discount factor, implied dividend yield
    config: PrepConfig

    def diagnostics(self) -> pd.DataFrame:
        """Rows kept/dropped per filter reason."""
        counts = self.audit["reason"].value_counts()
        order = ["ok"] + REASONS
        return (counts.reindex(order).fillna(0).astype(int)
                .rename_axis("reason").reset_index(name="rows"))

    def per_expiry(self) -> pd.DataFrame:
        a = self.audit.assign(kept=self.audit["reason"].eq("ok"))
        return a.groupby("expiry").agg(T=("T", "first"), raw=("kept", "size"),
                                       kept=("kept", "sum")).reset_index()


def _mid(raw: pd.DataFrame, use_last: bool):
    bid, ask, last = raw["bid"], raw["ask"], raw["last"]
    two_sided = (bid > 0) & (ask > 0) & (ask >= bid)
    mid = ((bid + ask) / 2).where(two_sided)
    if use_last:
        mid = mid.fillna(last.where(last > 0))
    return mid, two_sided


def estimate_forwards(audit: pd.DataFrame, ok: pd.Series, mid: pd.Series,
                      spot: float, cfg: PrepConfig) -> pd.DataFrame:
    """F = K + (C - P)/DF near the money, median over strikes (put-call parity)."""
    rows = []
    for expiry, g in audit[ok].groupby("expiry"):
        T = g["T"].iloc[0]
        df = np.exp(-cfg.rate * T)
        calls = mid[g.index[g["type"] == "C"]].groupby(g["strike"]).first()
        puts = mid[g.index[g["type"] == "P"]].groupby(g["strike"]).first()
        pairs = pd.concat([calls.rename("c"), puts.rename("p")], axis=1, join="inner")
        if len(pairs) < cfg.min_pairs:
            continue
        near = pairs.iloc[np.argsort(np.abs(pairs.index.values - spot))[:cfg.n_parity_strikes]]
        F = float(np.median(near.index.values + (near["c"] - near["p"]).values / df))
        q = cfg.rate - np.log(F / spot) / T
        rows.append(dict(expiry=expiry, T=T, F=F, df=df, implied_div_yield=q,
                         n_pairs=len(near)))
    return pd.DataFrame(rows, columns=["expiry", "T", "F", "df", "implied_div_yield", "n_pairs"])


def _hampel(k: np.ndarray, iv: np.ndarray, window: int, nsigma: float, floor: float):
    """Flag IV points far from a rolling median of their strike-neighbours."""
    order = np.argsort(k)
    s = pd.Series(iv[order])
    med = s.rolling(window, center=True, min_periods=3).median()
    resid = (s - med).values
    scale = 1.4826 * np.nanmedian(np.abs(resid))
    thr = max(nsigma * scale, floor)
    flag = np.zeros(len(iv), dtype=bool)
    flag[order] = np.abs(resid) > thr
    return flag


def prepare(raw: pd.DataFrame, cfg: PrepConfig | None = None) -> PrepResult:
    cfg = cfg or PrepConfig()
    audit = raw.reset_index(drop=True).copy()
    audit["reason"] = "ok"
    spot = float(audit["spot"].iloc[0])

    def flag(mask, name):
        hit = mask.fillna(False) & audit["reason"].eq("ok")
        audit.loc[hit, "reason"] = name

    mid, two_sided = _mid(audit, cfg.use_last)
    audit["mid"] = mid

    # --- quote hygiene -----------------------------------------------------
    flag((audit["bid"] > 0) & (audit["ask"] < audit["bid"]), "crossed")
    flag(mid.isna(), "no_quote")
    flag(two_sided & ((audit["ask"] - audit["bid"]) / mid > cfg.max_rel_spread), "wide_spread")
    flag(mid < cfg.min_price, "low_price")
    flag(audit["T"] * 365.0 < cfg.min_days, "short_dated")
    flag(audit["openInterest"] < cfg.min_open_interest, "low_interest")

    # --- forwards from put-call parity ------------------------------------
    fwd = estimate_forwards(audit, audit["reason"].eq("ok"), mid, spot, cfg)
    fmap = fwd.set_index("expiry")
    audit["F"] = audit["expiry"].map(fmap["F"])
    audit["df"] = audit["expiry"].map(fmap["df"])
    flag(audit["F"].isna(), "no_forward")

    # --- OTM selection: puts below the forward, calls at/above ------------
    audit["k"] = np.log(audit["strike"] / audit["F"])
    otm = ((audit["type"] == "C") & (audit["k"] >= 0)) | ((audit["type"] == "P") & (audit["k"] < 0))
    flag(~otm, "not_otm")
    flag(audit["k"].abs() > cfg.max_abs_k, "far_otm")

    # --- implied vol -------------------------------------------------------
    live = audit["reason"].eq("ok")
    cp = np.where(audit["type"] == "C", 1.0, -1.0)
    audit["iv"] = np.nan
    audit.loc[live, "iv"] = implied_vol(audit.loc[live, "mid"].values, audit.loc[live, "F"].values,
                                        audit.loc[live, "strike"].values, audit.loc[live, "T"].values,
                                        cp[live.values], audit.loc[live, "df"].values)
    flag(audit["iv"].isna(), "iv_failed")

    # --- IV outliers per expiry -------------------------------------------
    for _, g in audit[audit["reason"].eq("ok")].groupby("expiry"):
        if len(g) >= cfg.hampel_window:
            bad = _hampel(g["k"].values, g["iv"].values, cfg.hampel_window,
                          cfg.hampel_nsigma, cfg.hampel_floor)
            audit.loc[g.index[bad], "reason"] = "iv_outlier"

    # --- need enough survivors to fit a smile -----------------------------
    n_ok = audit[audit["reason"].eq("ok")].groupby("expiry").size()
    thin = audit["expiry"].map(n_ok).fillna(0) < cfg.min_points
    flag(thin, "thin_slice")

    # --- quote weights: inverse-variance using the bid/ask width in vol units
    vega = black_vega(audit["F"], audit["strike"], audit["T"], audit["iv"], audit["df"])
    width = ((audit["ask"] - audit["bid"]) / vega).where(two_sided, 0.02)
    audit["vega"] = vega
    audit["iv_err"] = width.clip(lower=0.005, upper=0.10)

    quotes = audit[audit["reason"].eq("ok")].copy()
    quotes["weight"] = 1.0 / quotes["iv_err"] ** 2
    quotes["days"] = quotes["T"] * 365.0
    cols = ["expiry", "T", "days", "k", "strike", "F", "df", "type", "mid", "bid", "ask",
            "iv", "iv_err", "weight", "vega", "volume", "openInterest", "yahoo_iv"]
    quotes = quotes[cols].sort_values(["T", "k"]).reset_index(drop=True)
    fwd = fwd[fwd["expiry"].isin(quotes["expiry"])].reset_index(drop=True)
    return PrepResult(quotes=quotes, audit=audit, forwards=fwd, config=cfg)
