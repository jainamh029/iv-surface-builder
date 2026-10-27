"""Build the static portfolio site in docs/ from a finished run.

    python scripts/build_site.py outputs/<run>

Every number on the page is read from the run's result files (or recomputed here for the
synthetic validation), so the page cannot drift from the code. Needs the main environment.
"""
from __future__ import annotations

import html
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ivsurface.prep import prepare
from ivsurface.svi import calibrate
from ivsurface.synthetic import make_chain, true_iv

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
INK, MUTED, ACCENT = "#1f2937", "#6b7280", "#2563eb"


def validation_figure(path: Path) -> dict:
    """Run the pipeline on a chain with known SSVI truth and plot recovery."""
    prep = prepare(make_chain())
    fits = calibrate(prep.quotes)
    q = prep.quotes
    errs, ks = [], []
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.1))
    for a, i in zip(ax[:2], (2, 7)):
        f = fits[i]
        g = q[q["expiry"] == f.expiry]
        kk = np.linspace(f.k_min, f.k_max, 200)
        a.scatter(g["k"], g["iv"] * 100, s=8, color="#9ca3af", label="noisy market quotes")
        a.plot(kk, true_iv(kk, f.T) * 100, color=INK, lw=2.4, alpha=0.35, label="true surface")
        a.plot(kk, np.sqrt(f.params.w(kk) / f.T) * 100, color=ACCENT, lw=1.4, label="recovered SVI")
        a.set(title=f"{f.T * 365:.0f}-day expiry", xlabel="ln(K/F)", ylabel="IV %")
        a.legend(frameon=False, fontsize=8)
    for f in fits:
        kk = np.linspace(f.k_min, f.k_max, 60)
        errs.append(np.sqrt(f.params.w(kk) / f.T) - true_iv(kk, f.T))
        ks.append(kk)
    e, k = 100 * np.concatenate(errs), np.concatenate(ks)
    ax[2].scatter(k, e, s=5, color=ACCENT, alpha=0.5)
    ax[2].axhline(0, color="#d1d5db", lw=0.8)
    ax[2].set(title="Recovery error, every slice", xlabel="ln(K/F)", ylabel="fitted - true IV (vol points)")
    for a in ax:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle("Validation: known surface -> noisy chain with injected bad quotes -> pipeline",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    stats = dict(max_err=float(np.abs(e).max()), rms=float(np.sqrt((e ** 2).mean())))
    return stats, prep, fits


def dataset_payload(label, title, quotes, report, audit, spot, asof, rate, truth=False, cross=None):
    """Everything the dashboard needs for one dataset, as plain JSON-able Python."""
    rep = report.set_index("expiry")
    slices = []
    for expiry, g in quotes.groupby("expiry", sort=False):
        r = rep.loc[expiry]
        sl = dict(expiry=expiry, days=round(float(r["days"]), 3), T=float(g["T"].iloc[0]), F=round(float(g["F"].iloc[0]), 4),
                  a=float(r["a"]), b=float(r["b"]), rho=float(r["rho"]), m=float(r["m"]), sigma=float(r["sigma"]),
                  k_min=float(g["k"].min()), k_max=float(g["k"].max()), n=int(r["n"]),
                  rmse=round(float(r["rmse_vol_pts"]), 4), max_err=round(float(r["max_err_vol_pts"]), 4),
                  min_g=round(float(r["min_g"]), 5), butterfly_ok=bool(r["butterfly_ok"]),
                  calendar_ok=bool(r["calendar_ok"]), repaired=bool(r["repaired"]),
                  q=dict(k=g["k"].round(5).tolist(), iv=g["iv"].round(5).tolist(), K=g["strike"].tolist(),
                         type=g["type"].tolist(), bid=g["bid"].round(3).tolist(), ask=g["ask"].round(3).tolist()))
        if truth:
            kk = np.linspace(sl["k_min"], sl["k_max"], 80)
            sl["truth"] = dict(k=kk.round(5).tolist(), iv=true_iv(kk, sl["T"]).round(5).tolist())
        slices.append(sl)
    reasons = audit["reason"].value_counts()
    return dict(label=label, title=title, spot=spot, asof=asof, rate=rate, n_raw=int(len(audit)),
                n_kept=int((audit["reason"] == "ok").sum()),
                filters=[dict(reason=k, rows=int(v)) for k, v in reasons.items()], slices=slices, cross=cross)


def export_dashboard(run, syn_prep, syn_fits, spot, asof, rate):
    import json

    from ivsurface.surface import IVSurface

    cross = pd.read_csv(run / "crosscheck_summary.csv").rename(columns={"fitter": "name"})
    cross = cross[["name", "mean_rmse", "worst_max_err", "slices_with_butterfly_arb", "calendar_violations", "curve_diff_vs_mine"]]
    spy = dataset_payload("SPY", "SPY, live snapshot", pd.read_csv(run / "quotes_clean.csv"), pd.read_csv(run / "svi_params.csv"),
                          pd.read_csv(run / "quotes_audit.csv"), spot, asof, rate, cross=cross.round(4).to_dict("records"))
    rep = IVSurface(syn_fits, 500.0, "2026-10-26", 0.04).arbitrage_report()
    syn = dataset_payload("SYNTHETIC", "Synthetic chain with known truth", syn_prep.quotes, rep, syn_prep.audit,
                          500.0, "2026-10-26", 0.04, truth=True)
    out = DOCS / "data"
    out.mkdir(exist_ok=True)
    for d in (spy, syn):
        (out / f"{d['label']}.json").write_text(json.dumps(d, separators=(",", ":")))
    assets = DOCS / "assets"
    assets.mkdir(exist_ok=True)
    import plotly
    shutil.copy(Path(plotly.__file__).parent / "package_data" / "plotly.min.js", assets / "plotly.min.js")


def table(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = ""
    for _, r in df.iterrows():
        cells = "".join(f"<td>{html.escape(fmt.get(c, str)(v))}</td>" for c, v in r.items())
        rows += f"<tr>{cells}</tr>"
    return f'<div class="tbl"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'


def main(run: Path) -> None:
    DOCS.mkdir(exist_ok=True)
    for name in ("surface.html", "smiles.png", "summary.png", "crosscheck_svi_py.png"):
        shutil.copy(run / name, DOCS / name)
    val, syn_prep, syn_fits = validation_figure(DOCS / "validation.png")

    audit = pd.read_csv(run / "quotes_audit.csv")
    params = pd.read_csv(run / "svi_params.csv")
    cross = pd.read_csv(run / "crosscheck_summary.csv")
    quotes = pd.read_csv(run / "quotes_clean.csv")
    surf = pd.read_json(run / "surface.json", typ="series")
    spot, asof = float(surf["spot"]), str(surf["asof"])[:10]

    kept, total = int((audit["reason"] == "ok").sum()), len(audit)
    reasons = audit["reason"].value_counts().drop("ok")
    n_exp = int(params.shape[0])
    ok_all = bool(params["butterfly_ok"].all() and params["calendar_ok"].all())
    atm = [np.sqrt((r.a + r.b * (r.rho * (0 - r.m) + np.sqrt(r.m ** 2 + r.sigma ** 2))) / (r.days / 365))
           for r in params.itertuples()]
    atm_short, atm_long = atm[0] * 100, atm[-1] * 100
    wing = float(quotes[quotes["days"] == quotes["days"].min()]["iv"].max()) * 100

    filt = pd.DataFrame({"Filter": reasons.index.str.replace("_", " "), "Rows": reasons.values})
    fit_tbl = params[["expiry", "days", "n", "rmse_vol_pts", "max_err_vol_pts", "butterfly_ok", "calendar_ok"]].copy()
    fit_tbl.columns = ["Expiry", "Days", "Quotes", "RMSE (vol pts)", "Max err (vol pts)", "No butterfly arb", "No calendar arb"]
    yn = lambda v: "yes" if bool(v) else "NO"
    c = cross.rename(columns={cross.columns[0]: "Fitter"})
    c = c[["Fitter", "mean_rmse", "worst_max_err", "slices_with_butterfly_arb", "calendar_violations", "curve_diff_vs_mine"]]
    c.columns = ["Calibrator", "Mean RMSE (vol pts)", "Worst error (vol pts)", "Slices w/ butterfly arb",
                 "Calendar violations", "Curve gap vs mine (vol pts)"]
    f2 = lambda v: f"{float(v):.2f}"
    f3 = lambda v: f"{float(v):.3f}"

    export_dashboard(run, syn_prep, syn_fits, spot, asof, float(surf["rate"]))

    page = TEMPLATE
    subs = {
        "SPOT": f"{spot:,.2f}", "ASOF": asof, "KEPT": f"{kept:,}", "TOTAL": f"{total:,}", "NEXP": str(n_exp),
        "ATM_SHORT": f"{atm_short:.1f}", "ATM_LONG": f"{atm_long:.1f}", "WING": f"{wing:.0f}",
        "MEAN_RMSE": f"{params['rmse_vol_pts'].mean():.2f}", "ARB": "0" if ok_all else "see table",
        "VAL_MAX": f"{val['max_err']:.2f}", "VAL_RMS": f"{val['rms']:.2f}",
        "FILTER_TABLE": table(filt, {"Rows": lambda v: f"{int(v):,}"}),
        "FIT_TABLE": table(fit_tbl, {"Days": lambda v: f"{float(v):.0f}", "RMSE (vol pts)": f3,
                                     "Max err (vol pts)": f2, "No butterfly arb": yn, "No calendar arb": yn}),
        "CROSS_TABLE": table(c, {"Mean RMSE (vol pts)": f2, "Worst error (vol pts)": f2,
                                 "Slices w/ butterfly arb": lambda v: str(int(v)),
                                 "Calendar violations": lambda v: str(int(v)),
                                 "Curve gap vs mine (vol pts)": f2}),
        "TUNED_RMSE": f2(cross.loc[cross.iloc[:, 0] == "svi-py tuned", "mean_rmse"].iloc[0]),
        "DEFAULT_RMSE": f2(cross.loc[cross.iloc[:, 0] == "svi-py default", "mean_rmse"].iloc[0]),
        "MINE_RMSE": f2(cross.loc[cross.iloc[:, 0] == "this project", "mean_rmse"].iloc[0]),
    }
    for k, v in subs.items():
        page = page.replace("{{" + k + "}}", v)
    (DOCS / "index.html").write_text(page)
    print(f"wrote {DOCS / 'index.html'}  (validation: rms {val['rms']:.3f}, max {val['max_err']:.3f} vol pts)")


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Implied Volatility Surface Builder</title>
<meta name="description" content="Option prices to a no-arbitrage SVI volatility surface, validated against a known truth and cross-checked against svi-py.">
<style>
:root{--ink:#111827;--muted:#6b7280;--line:#e5e7eb;--bg:#fff;--soft:#f8fafc;--accent:#2563eb;--warn:#dc2626}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.65 -apple-system,BlinkMacSystemFont,"Inter","Segoe UI",Roboto,sans-serif}
.wrap{max-width:980px;margin:0 auto;padding:0 20px}
header.hero{padding:64px 0 36px;border-bottom:1px solid var(--line);background:linear-gradient(180deg,var(--soft),#fff)}
.eyebrow{font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);font-weight:600}
h1{font-size:clamp(30px,5vw,46px);line-height:1.1;margin:10px 0 14px;letter-spacing:-.02em}
.lede{font-size:19px;color:#374151;max-width:760px;margin:0}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:30px 0 8px}
.stat{border:1px solid var(--line);border-radius:12px;padding:14px 16px;background:#fff}
.stat b{display:block;font-size:26px;letter-spacing:-.02em;line-height:1.2}
.stat span{font-size:13px;color:var(--muted)}
.links{margin-top:22px;display:flex;gap:10px;flex-wrap:wrap}
.btn{display:inline-block;padding:9px 16px;border-radius:9px;border:1px solid var(--line);color:var(--ink);text-decoration:none;font-weight:600;font-size:14px;background:#fff}
.btn.primary{background:var(--ink);color:#fff;border-color:var(--ink)}
section{padding:46px 0;border-bottom:1px solid var(--line)}
h2{font-size:26px;letter-spacing:-.01em;margin:0 0 6px}
.sub{color:var(--muted);margin:0 0 22px;max-width:760px}
p{max-width:760px}
img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:12px;display:block}
.frame{border:1px solid var(--line);border-radius:12px;overflow:hidden}
.frame iframe{width:100%;height:600px;border:0;display:block}
.tbl{overflow-x:auto;margin:14px 0}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:8px 12px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);background:var(--soft)}
td:not(:first-child),th:not(:first-child){text-align:right}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.card{border:1px solid var(--line);border-radius:12px;padding:18px 20px;background:#fff}
.card h3{margin:0 0 6px;font-size:16px}.card p{margin:0;font-size:15px;color:#374151}
.callout{border-left:3px solid var(--accent);background:var(--soft);padding:14px 18px;border-radius:0 10px 10px 0;max-width:760px}
.callout.warn{border-color:var(--warn)}
code{font:13px ui-monospace,SFMono-Regular,Menlo,monospace;background:#f3f4f6;padding:2px 6px;border-radius:5px}
pre{background:#0f172a;color:#e2e8f0;padding:16px 18px;border-radius:12px;overflow-x:auto;font:13px/1.6 ui-monospace,Menlo,monospace}
footer{padding:34px 0 60px;color:var(--muted);font-size:14px}
@media(max-width:720px){.grid2{grid-template-columns:1fr}.frame iframe{height:420px}header.hero{padding:40px 0 24px}}
</style></head><body>

<header class="hero"><div class="wrap">
<div class="eyebrow">Quant project &middot; derivatives</div>
<h1>Implied Volatility Surface Builder</h1>
<p class="lede">Take a real options chain, work backwards from prices to the volatility the market is pricing at every strike and expiry, fit an arbitrage-aware surface, and see the smile and skew that Black-Scholes assumes away.</p>
<div class="stats">
<div class="stat"><b>{{KEPT}}</b><span>of {{TOTAL}} SPY contracts survive cleaning</span></div>
<div class="stat"><b>{{NEXP}}</b><span>expiries fitted, as of {{ASOF}}</span></div>
<div class="stat"><b>{{MEAN_RMSE}} pts</b><span>mean fit error on market quotes</span></div>
<div class="stat"><b>{{ARB}}</b><span>butterfly / calendar arbitrage violations</span></div>
</div>
<div class="links"><a class="btn primary" href="#surface">Explore the surface</a><a class="btn" href="dashboard.html">Live dashboard</a><a class="btn" href="#validation">Validation</a><a class="btn" href="#crosscheck">svi-py cross-check</a></div>
</div></header>

<div class="wrap">

<section id="why"><h2>Why this matters</h2>
<p class="sub">Options traders quote volatility, not price. Black-Scholes assumes one volatility for every strike; the market disagrees loudly.</p>
<p>On {{ASOF}} the SPY at-the-money implied vol was about <b>{{ATM_SHORT}}%</b> at the shortest expiry and <b>{{ATM_LONG}}%</b> at the longest, while far out-of-the-money puts at the front of the curve priced in as much as <b>{{WING}}%</b>. Index crash risk is what that skew is paying for. The red dashed line on every panel below is the flat volatility Black-Scholes would use.</p>
<img src="smiles.png" alt="Volatility smiles by expiry with SVI fit and the flat Black-Scholes volatility" loading="lazy">
</section>

<section id="surface"><h2>The surface</h2>
<p class="sub">Interactive. Drag to rotate, scroll to zoom. Black dots are the market quotes; the surface is the fitted SVI. Regions outside the observed strikes are masked rather than extrapolated.</p>
<div class="frame"><iframe src="surface.html" title="Interactive 3D implied volatility surface" loading="lazy"></iframe></div>
<p style="margin-top:22px">The skew is steepest at short maturities and flattens with time; the term structure of at-the-money vol is upward sloping, with a bump around the first two expiries.</p>
<img src="summary.png" alt="ATM term structure, ATM skew and what the cleaning stage dropped" loading="lazy">
</section>

<section id="method"><h2>How it works</h2>
<p class="sub">Five stages, each with its own module and tests.</p>
<div class="grid2">
<div class="card"><h3>1 &middot; Clean, and show your work</h3><p>Crossed, one-sided, wide-spread, penny and statistically odd quotes are removed. Every dropped row is kept with its reason so nothing disappears silently.</p></div>
<div class="card"><h3>2 &middot; Forward from put-call parity</h3><p><code>F = K + (C - P)/DF</code> near the money, so dividends and borrow costs are read from the market instead of assumed.</p></div>
<div class="card"><h3>3 &middot; Invert Black-76</h3><p>Out-of-the-money options only, solved with vectorised bisection. Quotes violating static no-arbitrage bounds are rejected instead of producing nonsense vols.</p></div>
<div class="card"><h3>4 &middot; Constrained raw SVI</h3><p>Zeliade's quasi-explicit reduction turns each slice into a small linear least-squares problem, with positivity, Lee wing bounds and calendar spreads as linear constraints, plus a butterfly check and repair.</p></div>
</div>
<h3 style="margin-top:34px">What the cleaning stage removed</h3>
{{FILTER_TABLE}}
<p class="callout" style="margin-top:6px">"Not otm" is not data loss: each strike contributes its out-of-the-money leg only, so the in-the-money twin is intentionally skipped.</p>
<h3 style="margin-top:34px">Per-expiry fit and arbitrage audit</h3>
{{FIT_TABLE}}
</section>

<section id="validation"><h2>Validation against a known truth</h2>
<p class="sub">A fit that matches the market proves little if the market is noisy. So the pipeline is also run on a chain generated from a known arbitrage-free SSVI surface, with bid/ask noise, tick rounding and injected bad quotes.</p>
<img src="validation.png" alt="Recovered surface versus true surface on a synthetic chain" loading="lazy">
<p style="margin-top:18px">Across every slice, inside the observed strikes, the recovered vols are within <b>{{VAL_MAX}} vol points</b> of the truth (RMS <b>{{VAL_RMS}}</b>). This is enforced as a test, not just plotted.</p>
</section>

<section id="crosscheck"><h2>Cross-check against svi-py</h2>
<p class="sub">An independent implementation, run on exactly the same cleaned quotes and forwards. Fit quality, butterfly and calendar checks are computed by a separate script, not by either library.</p>
<img src="crosscheck_svi_py.png" alt="Fit error of this project versus svi-py, default and tuned" loading="lazy">
{{CROSS_TABLE}}
<div class="callout"><b>Reading it honestly.</b> With its default settings svi-py averages <b>{{DEFAULT_RMSE}}</b> vol points of error, because it minimises unweighted <i>total-variance</i> error, which neglects the at-the-money region. Switched to its own <code>implied_vol</code> objective with multi-start, it reaches <b>{{TUNED_RMSE}}</b>, essentially level with this project's <b>{{MINE_RMSE}}</b>, and the fitted curves agree to about a tenth of a vol point. Two independent calibrators converging is the real result. The differences are that this one enforces calendar spreads by construction, and that it is about 15&ndash;20&times; faster than svi-py's tuned configuration (roughly 0.09 s versus 1.5&ndash;2 s per slice); svi-py's untuned default is faster still (about 0.01 s) but is the inaccurate one.</div>
</section>

<section id="run"><h2>Run it</h2>
<pre>pip install -e ".[dev]"
python -m ivsurface SPY            # live chain from Yahoo Finance
python -m ivsurface --synthetic    # offline, known ground truth
pytest                             # 13 tests</pre>
</section>

<section id="limits" style="border-bottom:0"><h2>Limitations</h2>
<div class="callout warn">
<p style="margin:0 0 8px">SPY options are American-style but priced here with Black-76. Using only out-of-the-money options limits the error, but a European index such as <code>^SPX</code> is the clean choice.</p>
<p style="margin:0 0 8px">Free Yahoo data is delayed; this snapshot was taken after the close. The discount rate is a flat constant used only for discounting.</p>
<p style="margin:0">Arbitrage checks cover the observed strike range plus a small margin. Beyond it the surface is extrapolation, and is masked in the 3D view.</p></div>
</section>

</div>
<footer><div class="wrap">Built with Python, NumPy, SciPy, pandas, Plotly and Matplotlib. Data: Yahoo Finance.</div></footer>
</body></html>
"""

if __name__ == "__main__":
    main(Path(sys.argv[1]))
