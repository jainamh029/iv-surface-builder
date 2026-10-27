"""Cross-check this project's SVI calibrator against the svi-py library (import name: pysvi).

svi-py needs Python >= 3.13, so run this from a separate environment:

    uv venv --python 3.13 .venv-xcheck
    uv pip install --python .venv-xcheck/bin/python svi-py pandas matplotlib
    .venv-xcheck/bin/python scripts/crosscheck_svi_py.py outputs/<run>

Both calibrators get the *same* cleaned quotes and the *same* forwards, so the comparison
isolates the slice fitter. Fit quality, butterfly and calendar checks are all computed here,
independently of either library's own diagnostics.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pysvi import ArbitrageFreedom, calibrate_slice, get_model

BF_MARGIN, CAL_MARGIN = 0.10, 0.05      # same audit windows as ivsurface.svi
INK, MINE, LIB_Q, LIB_B = "#1f2937", "#2563eb", "#f59e0b", "#10b981"


def w_svi(p, k):
    x = np.asarray(k, float) - p["m"]
    return p["a"] + p["b"] * (p["rho"] * x + np.sqrt(x * x + p["sigma"] ** 2))


def min_g(p, k):
    """Gatheral's butterfly function on the observed range +/- margin (>= 0 means no arbitrage)."""
    kk = np.linspace(k.min() - BF_MARGIN, k.max() + BF_MARGIN, 200)
    x = kk - p["m"]
    s = np.sqrt(x * x + p["sigma"] ** 2)
    w = np.maximum(w_svi(p, kk), 1e-12)
    wp, wpp = p["b"] * (p["rho"] + x / s), p["b"] * p["sigma"] ** 2 / s ** 3
    return float(((1 - kk * wp / (2 * w)) ** 2 - wp ** 2 / 4 * (1 / w + 0.25) + wpp / 2).min())


def main(run_dir: Path) -> None:
    q = pd.read_csv(run_dir / "quotes_clean.csv")
    mine = pd.read_csv(run_dir / "svi_params.csv").set_index("expiry")
    # (model, calibrate kwargs). "tuned" uses svi-py's own implied-vol objective + multi-start,
    # i.e. it optimises the same quantity (vol error) that this script reports.
    tuned = dict(objective="implied_vol", initialization="multi_start")
    models = {"svi-py default": (get_model("svi", ArbitrageFreedom.QUASI), {}),
              "svi-py default + no-butterfly": (get_model("svi", ArbitrageFreedom.NO_BUTTERFLY), {}),
              "svi-py tuned": (get_model("svi", ArbitrageFreedom.QUASI), tuned),
              "svi-py tuned + no-butterfly": (get_model("svi", ArbitrageFreedom.NO_BUTTERFLY), tuned)}

    rows, curves = [], {}
    for expiry, g in q.groupby("expiry", sort=False):
        T, k, iv = float(g["T"].iloc[0]), g["k"].values, g["iv"].values
        fits = {"this project": {c: float(mine.loc[expiry, c]) for c in ("a", "b", "rho", "m", "sigma")}}
        secs = {"this project": np.nan}
        df = g.rename(columns={"T": "maturity", "F": "implied_forward"})
        for name, (model, kw) in models.items():
            t0 = time.perf_counter()
            p = calibrate_slice(df, model, **kw)
            secs[name] = time.perf_counter() - t0
            fits[name] = None if p is None else {c: float(p[c]) for c in ("a", "b", "rho", "m", "sigma")}
        for name, p in fits.items():
            if p is None:
                rows.append(dict(expiry=expiry, days=T * 365, fitter=name, failed=True))
                continue
            err = np.sqrt(np.maximum(w_svi(p, k), 1e-12) / T) - iv
            rows.append(dict(expiry=expiry, days=T * 365, fitter=name, failed=False, n=len(k),
                             rmse_vol_pts=100 * np.sqrt(np.mean(err ** 2)),
                             max_err_vol_pts=100 * np.abs(err).max(), min_g=min_g(p, k),
                             seconds=secs[name], **p))
        curves[expiry] = dict(T=T, k=k, iv=iv, fits=fits)

    res = pd.DataFrame(rows)
    res.to_csv(run_dir / "crosscheck_svi_py.csv", index=False)

    # calendar audit: adjacent fitted slices, on the overlap of observed ranges
    cal = {}
    ex = list(curves)
    for name in res["fitter"].unique():
        bad = 0
        for a, b in zip(ex[:-1], ex[1:]):
            pa, pb = curves[a]["fits"][name], curves[b]["fits"][name]
            if pa is None or pb is None:
                continue
            lo = max(curves[a]["k"].min(), curves[b]["k"].min()) - CAL_MARGIN
            hi = min(curves[a]["k"].max(), curves[b]["k"].max()) + CAL_MARGIN
            if lo < hi:
                kg = np.linspace(lo, hi, 61)
                bad += int(np.sum(w_svi(pb, kg) - w_svi(pa, kg) < -1e-9))
        cal[name] = bad

    summ = res[~res["failed"].fillna(False)].groupby("fitter").agg(
        slices=("expiry", "size"), mean_rmse=("rmse_vol_pts", "mean"),
        worst_max_err=("max_err_vol_pts", "max"), slices_with_butterfly_arb=("min_g", lambda s: int((s < -1e-6).sum())),
        mean_seconds=("seconds", "mean"))
    summ["calendar_violations"] = pd.Series(cal)

    # curve-to-curve agreement with this project's fit, over each slice's observed strikes
    diff = {}
    for name in res["fitter"].unique():
        d = []
        for e in ex:
            a, b = curves[e]["fits"]["this project"], curves[e]["fits"][name]
            if b is not None:
                kk = np.linspace(curves[e]["k"].min(), curves[e]["k"].max(), 200)
                T = curves[e]["T"]
                d.append(np.sqrt(np.mean((np.sqrt(w_svi(a, kk) / T) - np.sqrt(np.maximum(w_svi(b, kk), 1e-12) / T)) ** 2)))
        diff[name] = 100 * float(np.mean(d))
    summ["curve_diff_vs_mine"] = pd.Series(diff)
    summ["failed_slices"] = res.groupby("fitter")["failed"].apply(lambda s: int(s.fillna(False).sum()))
    summ = summ.reindex(["this project", *models]).round(4)
    summ.to_csv(run_dir / "crosscheck_summary.csv")
    print(summ.to_string())

    # --- figure -------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    cols = {"this project": MINE, "svi-py default": LIB_Q, "svi-py default + no-butterfly": "#fcd34d",
            "svi-py tuned": LIB_B, "svi-py tuned + no-butterfly": "#6ee7b7"}
    x = np.arange(len(ex))
    for i, (name, c) in enumerate(cols.items()):
        r = res[res["fitter"] == name].set_index("expiry").reindex(ex)
        ax[0].bar(x + (i - 2) * 0.16, r["rmse_vol_pts"], 0.16, color=c, label=name)
    ax[0].set(title="Fit error on market quotes", ylabel="RMSE (vol points)", xticks=x,
              xticklabels=[f"{curves[e]['T'] * 365:.0f}d" for e in ex])
    ax[0].legend(frameon=False, fontsize=7)
    ax[0].set_yscale("log")
    for j, pick in enumerate([ex[len(ex) // 3], ex[-1]]):
        a, c_ = ax[1 + j], curves[pick]
        kk = np.linspace(c_["k"].min(), c_["k"].max(), 200)
        a.scatter(c_["k"], c_["iv"] * 100, s=7, color="#9ca3af", label="market", zorder=1)
        for name, col in cols.items():
            p = c_["fits"][name]
            if p is not None:
                a.plot(kk, np.sqrt(np.maximum(w_svi(p, kk), 1e-12) / c_["T"]) * 100, color=col, lw=1.6,
                       ls="-" if name == "this project" else ("--" if "tuned" in name else ":"), label=name)
        a.set(title=f"{pick}  ({c_['T'] * 365:.0f}d)", xlabel="ln(K/F)", ylabel="IV %")
        a.legend(frameon=False, fontsize=7)
    for a in ax:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle("Cross-check: this project's SVI calibrator vs svi-py (same quotes, same forwards)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(run_dir / "crosscheck_svi_py.png", dpi=150)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
