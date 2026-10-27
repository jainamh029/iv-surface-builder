"""Figures: interactive 3D surface, per-expiry smiles, term structure / skew / filter audit."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from .prep import PrepResult
from .surface import IVSurface

INK, MUTED, ACCENT, WARN = "#1f2937", "#6b7280", "#2563eb", "#dc2626"


def plot_surface_3d(surface: IVSurface, quotes, path: str | Path, title: str = "") -> Path:
    """Interactive Plotly surface (fitted SVI) with the market quotes overlaid."""
    k, T, Z = surface.grid()
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=k, y=T * 365, z=Z * 100, colorscale="Viridis", opacity=0.92,
        colorbar=dict(title="IV %", len=0.6),
        hovertemplate="ln(K/F)=%{x:.3f}<br>days=%{y:.0f}<br>IV=%{z:.1f}%<extra>SVI fit</extra>"))
    fig.add_trace(go.Scatter3d(
        x=quotes["k"], y=quotes["days"], z=quotes["iv"] * 100, mode="markers",
        marker=dict(size=2.2, color="#111827"), name="market quotes",
        hovertemplate="K=%{customdata[0]:.0f} %{customdata[1]}<br>days=%{y:.0f}<br>IV=%{z:.1f}%<extra>quote</extra>",
        customdata=np.column_stack([quotes["strike"], quotes["type"]])))
    fig.update_layout(
        title=title or "Implied volatility surface", margin=dict(l=0, r=0, t=50, b=0),
        scene=dict(xaxis_title="log-moneyness ln(K/F)", yaxis_title="days to expiry",
                   zaxis_title="implied vol (%)", camera=dict(eye=dict(x=1.7, y=-1.7, z=0.9))),
        legend=dict(x=0.02, y=0.98))
    path = Path(path)
    fig.write_html(path, include_plotlyjs=True)
    return path


def plot_smiles(surface: IVSurface, quotes, path: str | Path, n: int = 9, title: str = "") -> Path:
    """Market IV vs fitted SVI for ~n expiries, against the flat vol Black-Scholes would imply."""
    idx = np.unique(np.linspace(0, len(surface.fits) - 1, min(n, len(surface.fits))).round().astype(int))
    cols = 3
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.2 * rows), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, i in zip(axes.flat, idx):
        ax.set_visible(True)
        f = surface.fits[i]
        g = quotes[quotes["expiry"] == f.expiry]
        k = np.linspace(f.k_min - 0.03, f.k_max + 0.03, 200)
        ax.scatter(g["k"], g["iv"] * 100, s=9, color=MUTED, alpha=0.8, label="market", zorder=3)
        ax.plot(k, np.sqrt(f.params.w(k) / f.T) * 100, color=ACCENT, lw=1.8, label="SVI fit")
        atm = np.sqrt(f.params.w(0.0) / f.T) * 100
        ax.axhline(atm, color=WARN, lw=1, ls="--", label="flat vol (Black-Scholes)")
        ax.axvline(0, color="#d1d5db", lw=0.8)
        ax.set_title(f"{f.expiry}  ({f.T * 365:.0f}d)", fontsize=10, color=INK)
        ax.set_xlabel("ln(K/F)", fontsize=8)
        ax.set_ylabel("IV %", fontsize=8)
        ax.tick_params(labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(title or "Volatility smiles", x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = Path(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_summary(surface: IVSurface, prep: PrepResult, path: str | Path, title: str = "") -> Path:
    """ATM term structure, ATM skew, and what the cleaning stage dropped."""
    days = surface.T * 365
    atm = np.array([np.sqrt(f.params.w(0.0) / f.T) for f in surface.fits]) * 100
    skew = np.array([f.params.dw(0.0) / (2 * np.sqrt(f.params.w(0.0) * f.T)) for f in surface.fits])

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].plot(days, atm, "o-", color=ACCENT, ms=4)
    ax[0].set(title="ATM implied vol term structure", xlabel="days to expiry", ylabel="IV %")
    ax[1].plot(days, skew, "o-", color=WARN, ms=4)
    ax[1].axhline(0, color="#d1d5db", lw=0.8)
    ax[1].set(title="ATM skew  dIV/dk  (negative = puts richer)", xlabel="days to expiry", ylabel="dIV / d ln(K/F)")

    d = prep.diagnostics()
    d = d[(d["reason"] != "ok") & (d["rows"] > 0)].sort_values("rows")
    ax[2].barh(d["reason"], d["rows"], color=MUTED)
    for y, v in enumerate(d["rows"]):
        ax[2].text(v, y, f" {v}", va="center", fontsize=8, color=INK)
    kept, total = int((prep.audit["reason"] == "ok").sum()), len(prep.audit)
    ax[2].set(title=f"Rows dropped by filter  (kept {kept} of {total})", xlabel="rows")
    for a in ax:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.tick_params(labelsize=9)
    fig.suptitle(title or "Surface summary", x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = Path(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
