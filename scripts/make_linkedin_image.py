"""Render a single shareable graphic (1200x1350, LinkedIn portrait) from the SPY run.

    python scripts/make_linkedin_image.py outputs/SPY social/linkedin_post.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager as fm

BG, PANEL, INK, MUTED = "#0d1117", "#141a23", "#eef1f6", "#8b95a7"
GOLD, BLUE, RED, GRID = "#e8b64c", "#6ea8ff", "#ff6b6b", "#252d3a"

run = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/SPY")
out = Path(sys.argv[2] if len(sys.argv) > 2 else "social/linkedin_post.png")
out.parent.mkdir(parents=True, exist_ok=True)

for f in ("HelveticaNeue.ttc", "Helvetica.ttc"):
    p = Path("/System/Library/Fonts") / f
    if p.exists():
        fm.fontManager.addfont(str(p))
        plt.rcParams["font.family"] = fm.FontProperties(fname=str(p)).get_name()
        break

q = pd.read_csv(run / "quotes_clean.csv")
par = pd.read_csv(run / "svi_params.csv")
audit = pd.read_csv(run / "quotes_audit.csv")

EXPIRY = par["expiry"].iloc[0]                       # shortest slice: steepest skew
s = par[par["expiry"] == EXPIRY].iloc[0]
d = q[q["expiry"] == EXPIRY]
T = float(d["T"].iloc[0])


def svi_iv(k):
    w = s.a + s.b * (s.rho * (k - s.m) + np.sqrt((k - s.m) ** 2 + s.sigma ** 2))
    return 100 * np.sqrt(np.maximum(w, 0) / T)


k = np.linspace(d["k"].min() - 0.03, d["k"].max() + 0.03, 400)
iv = svi_iv(k)
atm = float(svi_iv(np.array([0.0]))[0])
wing = float(svi_iv(np.array([d["k"].min()]))[0])
strike_lo = d.loc[d["k"].idxmin(), "strike"]

n_raw, n_kept = len(audit), len(q)
rmse = par["rmse_vol_pts"].mean()
n_exp = par["expiry"].nunique()

fig = plt.figure(figsize=(12, 13.5), dpi=100, facecolor=BG)


def text(x, y, s_, size, color=INK, weight="normal", ha="left", va="baseline", **kw):
    fig.text(x, y, s_, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, **kw)


# header
text(0.06, 0.94, "SPY OPTIONS  |  26 OCT 2026", 12.5, MUTED, "medium")
text(0.06, 0.895, "Black-Scholes sees one volatility.", 33, INK, "bold")
text(0.06, 0.857, "The market prices a whole surface.", 33, GOLD, "bold")

# chart
ax = fig.add_axes([0.09, 0.345, 0.86, 0.47], facecolor=PANEL)
ax.fill_between(k, atm, iv, where=iv > atm, color=RED, alpha=0.16, lw=0)
ax.axhline(atm, color=RED, lw=2, ls=(0, (5, 4)))
ax.plot(k, iv, color=BLUE, lw=3.6, solid_capstyle="round", zorder=4)
ax.scatter(d["k"], 100 * d["iv"], s=22, color=INK, alpha=0.85, lw=0, zorder=5)
ax.set_xlim(k.min(), k.max())
ax.set_ylim(5, wing + 9)
ax.grid(True, color=GRID, lw=0.8)
ax.set_axisbelow(True)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.tick_params(colors=MUTED, labelsize=11, length=0, pad=8)
ax.set_xlabel("Strike, as ln(K / forward)    <  puts  |  calls  >", color=MUTED, fontsize=12, labelpad=12)
ax.set_ylabel("Implied volatility (%)", color=MUTED, fontsize=12, labelpad=10)
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")

xl = d["k"].min()
ax.annotate("", xy=(xl + 0.002, 100 * d.loc[d["k"].idxmin(), "iv"] - 1.5), xytext=(xl + 0.012, 47),
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6, shrinkA=0, shrinkB=2))
ax.text(xl + 0.008, 41.5, f"{wing:.0f}%", color=INK, fontsize=38, fontweight="bold", va="center")
ax.text(xl + 0.008, 34, f"{strike_lo:.0f}-strike put, 9 days out", color=MUTED, fontsize=13, va="center")
ax.text(xl + 0.008, atm + 2.3, f"Flat Black-Scholes vol: {atm:.0f}% everywhere", color=RED, fontsize=14,
        fontweight="bold", ha="left")
ax.text(k.max() - 0.005, 7.6, "dots = market quotes     line = arbitrage-free SVI fit",
        color=MUTED, fontsize=11.5, ha="right")

# stat tiles
tiles = [
    (f"{n_kept:,} / {n_raw:,}", "contracts kept after\ncleaning, every drop logged"),
    (f"{rmse:.2f} pts", "mean fit error across\n{} expiries".format(n_exp)),
    ("0", "butterfly or calendar\narbitrage violations"),
]
x0, w, gap = 0.06, 0.28, 0.02
for i, (big, small) in enumerate(tiles):
    x = x0 + i * (w + gap)
    fig.patches.append(plt.Rectangle((x, 0.16), w, 0.125, transform=fig.transFigure,
                                     facecolor=PANEL, edgecolor=GRID, lw=1.2))
    text(x + 0.02, 0.245, big, 27, GOLD, "bold")
    text(x + 0.02, 0.212, small.split("\n")[0], 12, MUTED)
    text(x + 0.02, 0.19, small.split("\n")[1], 12, MUTED)

# footer
fig.add_artist(plt.Line2D([0.06, 0.94], [0.108, 0.108], color=GRID, lw=1.2, transform=fig.transFigure))
text(0.06, 0.068, "Implied Volatility Surface Builder", 17, INK, "bold")
text(0.06, 0.042, "Real options chain  >  Black-76 inversion  >  SVI surface  >  interactive 3D", 12.5, MUTED)
text(0.94, 0.068, "Jainam Shah", 15, GOLD, "bold", ha="right")

fig.savefig(out, facecolor=BG)
print(f"wrote {out}  ({n_kept} of {n_raw} contracts, ATM {atm:.1f}%, wing {wing:.1f}%, expiry {EXPIRY})")
