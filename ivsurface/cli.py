"""Command line entry point:  python -m ivsurface SPY  |  --synthetic  |  --from-csv chain.csv"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import data, plots
from .prep import PrepConfig, prepare
from .surface import IVSurface
from .svi import calibrate
from .synthetic import make_chain


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ivsurface", description="Build an implied volatility surface.")
    p.add_argument("symbol", nargs="?", default="SPY", help="Yahoo ticker (SPY, QQQ, ^SPX, ...)")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--synthetic", action="store_true", help="use a generated SSVI chain (offline)")
    src.add_argument("--from-csv", type=Path, help="reload a chain saved by a previous run")
    p.add_argument("--out", type=Path, default=Path("outputs"), help="output directory")
    p.add_argument("--max-expiries", type=int, default=12)
    p.add_argument("--max-days", type=float, default=400)
    p.add_argument("--rate", type=float, default=0.04, help="flat continuous risk-free rate for discounting")
    p.add_argument("--min-days", type=float, default=7, help="drop expiries shorter than this")
    p.add_argument("--max-abs-k", type=float, default=0.5, help="drop |ln(K/F)| beyond this")
    p.add_argument("--max-spread", type=float, default=0.5, help="max (ask-bid)/mid")
    p.add_argument("--use-last", action="store_true",
                   help="use last trade when there is no two-sided quote (needed when the market is closed)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.synthetic:
        label, raw = "SYNTHETIC", make_chain()
    elif args.from_csv:
        # data/SPY_20261026_2302.csv -> "SPY"
        label, raw = args.from_csv.stem.split("_")[0], data.load_chain(args.from_csv)
    else:
        label = args.symbol.replace("^", "")
        print(f"Fetching {args.symbol} option chain from Yahoo Finance ...")
        raw = data.fetch_chain(args.symbol, args.max_expiries, max_days=args.max_days)
        stamp = pd.Timestamp(raw["asof"].iloc[0]).strftime("%Y%m%d_%H%M")
        saved = data.save_chain(raw, Path("data") / f"{label}_{stamp}.csv")
        print(f"  saved raw chain -> {saved}")

    cfg = PrepConfig(rate=args.rate, min_days=args.min_days, max_abs_k=args.max_abs_k,
                     max_rel_spread=args.max_spread, use_last=args.use_last)
    prep = prepare(raw, cfg)
    if prep.quotes.empty:
        print(prep.diagnostics().to_string(index=False))
        raise SystemExit("No usable quotes after cleaning. If the market is closed, retry with --use-last.")

    spot, asof = float(raw["spot"].iloc[0]), str(raw["asof"].iloc[0])
    print(f"\n{label}: spot {spot:.2f}, as of {asof}, {raw['expiry'].nunique()} expiries, {len(raw)} contracts")
    print("\nQuote filtering:")
    print(prep.diagnostics().query("rows > 0").to_string(index=False))

    print("\nForwards from put-call parity:")
    f = prep.forwards.assign(days=lambda d: d["T"] * 365)[["expiry", "days", "F", "implied_div_yield", "n_pairs"]]
    print(f.round(4).to_string(index=False))

    print("\nCalibrating raw-SVI slices ...")
    surface = IVSurface(calibrate(prep.quotes), spot, asof, args.rate)
    report = surface.arbitrage_report()
    print(report[["expiry", "days", "n", "rmse_vol_pts", "max_err_vol_pts", "min_g",
                  "butterfly_ok", "calendar_ok", "repaired"]].round(3).to_string(index=False))
    viol = surface.calendar_violations()
    print(f"\nCalendar-arbitrage violations on interpolated grid: {viol}")

    out = args.out / label
    out.mkdir(parents=True, exist_ok=True)
    title = f"{label}  spot {spot:.2f}  ({asof[:16]})"
    prep.quotes.to_csv(out / "quotes_clean.csv", index=False)
    prep.audit.to_csv(out / "quotes_audit.csv", index=False)
    report.to_csv(out / "svi_params.csv", index=False)
    surface.to_json(out / "surface.json")
    html = plots.plot_surface_3d(surface, prep.quotes, out / "surface.html", title)
    plots.plot_smiles(surface, prep.quotes, out / "smiles.png", title=title)
    plots.plot_summary(surface, prep, out / "summary.png", title=title)
    print(f"\nWrote results to {out}/   (open {html.name} for the interactive 3D surface)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
