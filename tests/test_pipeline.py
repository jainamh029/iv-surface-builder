import numpy as np
import pytest

from ivsurface.black import black_price, implied_vol
from ivsurface.prep import PrepConfig, prepare
from ivsurface.surface import IVSurface
from ivsurface.svi import SVIParams, calibrate, fit_slice
from ivsurface.synthetic import make_chain, true_iv


# ---------------------------------------------------------------- Black-76
def test_implied_vol_roundtrip():
    F, T = 100.0, 0.5
    K = np.array([90, 95, 100, 105, 110.0])   # deep ITM options have ~zero time value: vol unidentifiable
    for cp in (1.0, -1.0):
        for vol in (0.08, 0.25, 0.9):
            px = black_price(F, K, T, vol, cp, df=0.98)
            np.testing.assert_allclose(implied_vol(px, F, K, T, cp, df=0.98), vol, atol=1e-8)


def test_put_call_parity():
    F, K, T, vol, df = 100.0, 95.0, 0.75, 0.3, 0.97
    c, p = black_price(F, K, T, vol, 1.0, df), black_price(F, K, T, vol, -1.0, df)
    assert c - p == pytest.approx(df * (F - K))


def test_implied_vol_rejects_arbitrage_violations():
    # below intrinsic, at the upper bound, and zero price must all give NaN
    out = implied_vol(np.array([4.0, 100.0, 0.0]), 100.0, np.array([90.0, 90.0, 90.0]), 0.5, 1.0)
    assert np.isnan(out).all()


# --------------------------------------------------------------------- SVI
def test_svi_slice_recovers_known_parameters():
    true = SVIParams(a=0.01, b=0.15, rho=-0.5, m=0.02, sigma=0.15)
    k = np.linspace(-0.4, 0.3, 40)
    p, _, _ = fit_slice(k, true.w(k), np.ones_like(k), T=0.5)
    np.testing.assert_allclose(p.w(k), true.w(k), atol=2e-5)
    assert p.g(np.linspace(-0.5, 0.4, 100)).min() >= -1e-6


def test_svi_slice_respects_calendar_constraint():
    prev = SVIParams(a=0.02, b=0.12, rho=-0.4, m=0.0, sigma=0.2)
    k = np.linspace(-0.3, 0.2, 30)
    w = 0.9 * prev.w(k)              # market data that would violate calendar vs prev
    p, _, constrained = fit_slice(k, w, np.ones_like(k), T=0.5, cal=(k, prev.w(k)))
    assert constrained
    assert np.min(p.w(k) - prev.w(k)) >= -1e-9


# ------------------------------------------------------- end-to-end (synthetic)
@pytest.fixture(scope="module")
def run():
    prep = prepare(make_chain())
    fits = calibrate(prep.quotes)
    return prep, fits, IVSurface(fits, 500.0, "t", 0.04)


def test_cleaning_catches_injected_bad_quotes(run):
    prep, _, _ = run
    reasons = prep.audit["reason"].value_counts()
    assert reasons.get("crossed", 0) > 0
    assert reasons.get("no_quote", 0) > 0
    assert reasons.get("not_otm", 0) > 0
    assert (prep.quotes["iv"] > 0).all() and prep.quotes["iv"].notna().all()


def test_parity_forward_and_dividend_yield(run):
    prep, _, _ = run
    assert prep.forwards["implied_div_yield"].between(0.010, 0.016).all()   # truth: 1.3%


def test_clean_ivs_match_truth(run):
    q = run[0].quotes
    err = q["iv"] - true_iv(q["k"], q["T"])
    assert np.sqrt((err ** 2).mean()) < 0.003      # < 0.3 vol points despite bid/ask noise


def test_surface_matches_truth_inside_observed_range(run):
    _, fits, surf = run
    for f in fits:
        k = np.linspace(f.k_min, f.k_max, 40)
        err = surf.iv(k, f.T) - true_iv(k, f.T)
        assert np.abs(err).max() < 0.01            # < 1 vol point everywhere the data lives


def test_no_static_arbitrage(run):
    _, fits, surf = run
    assert all(f.butterfly_ok and f.calendar_ok for f in fits)
    assert surf.calendar_violations() == 0


def test_interpolation_hits_slices_and_is_monotone_in_total_variance(run):
    _, fits, surf = run
    f = fits[3]
    np.testing.assert_allclose(surf.total_variance(0.0, f.T), f.params.w(0.0))
    T = np.linspace(surf.T[0], surf.T[-1], 50)
    w = np.array([surf.total_variance(0.0, t) for t in T])
    assert (np.diff(w) > 0).all()
    with pytest.raises(ValueError):
        surf.total_variance(0.0, surf.T[-1] * 1.5)


def test_json_roundtrip(run, tmp_path):
    _, _, surf = run
    surf.to_json(tmp_path / "s.json")
    again = IVSurface.from_json(tmp_path / "s.json")
    np.testing.assert_allclose(again.iv(0.0, 0.3), surf.iv(0.0, 0.3))


def test_use_last_rescues_closed_market():
    raw = make_chain(bad_quote_frac=0.0)
    raw[["bid", "ask"]] = 0.0                      # market closed: no two-sided quotes
    assert prepare(raw).quotes.empty
    assert not prepare(raw, PrepConfig(use_last=True)).quotes.empty
