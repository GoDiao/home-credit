"""scorecard.py unit tests"""
import numpy as np
import pandas as pd
import pytest

from config import BASE_SCORE, BASE_ODDS, PDO
from scorecard import (
    pd_to_score,
    score_to_pd,
    compute_score_bins,
    compute_lift_gains,
    compute_scorecard_summary,
)


def test_pd_to_score_basic():
    pd_vals = np.array([0.02, 0.05, 0.10, 0.50])
    scores = pd_to_score(pd_vals)
    assert scores.shape == pd_vals.shape
    # Higher PD → lower score
    assert scores[0] > scores[-1]


def test_pd_to_score_at_base_odds():
    # PD at base_odds: PD = 1 / (1 + base_odds)
    pd_at_base = 1.0 / (1 + BASE_ODDS)
    scores = pd_to_score(np.array([pd_at_base]))
    np.testing.assert_allclose(scores[0], BASE_SCORE, atol=0.5)


def test_pd_to_score_pdo_property():
    """Every PDO points, odds should double (PD halved approximately)."""
    pd_val = np.array([0.05])
    s1 = pd_to_score(pd_val)[0]
    # Score increased by PDO → odds doubled → PD should be lower
    pd_new = score_to_pd(np.array([s1 + PDO]))[0]
    pd_orig = score_to_pd(np.array([s1]))[0]
    odds_new = (1 - pd_new) / pd_new
    odds_orig = (1 - pd_orig) / pd_orig
    np.testing.assert_allclose(odds_new / odds_orig, 2.0, rtol=0.01)


def test_score_to_pd_roundtrip():
    pd_vals = np.array([0.01, 0.05, 0.10, 0.30, 0.80])
    scores = pd_to_score(pd_vals)
    pd_back = score_to_pd(scores)
    np.testing.assert_allclose(pd_back, pd_vals, atol=1e-6)


def test_pd_to_score_clips():
    scores = pd_to_score(np.array([0.999]))
    assert scores[0] <= 850
    scores = pd_to_score(np.array([0.001]))
    assert scores[0] >= 300


def test_compute_score_bins():
    rng = np.random.default_rng(42)
    n = 2000
    scores = rng.normal(600, 80, size=n)
    target = (rng.random(n) < 0.08).astype(float)
    df = compute_score_bins(scores, target, n_bins=10)
    assert isinstance(df, pd.DataFrame)
    assert "bad_rate" in df.columns
    assert "ks" in df.columns
    assert df["count"].sum() == n
    assert df["bad"].sum() == int(target.sum())


def test_compute_score_bins_all_good():
    scores = np.linspace(400, 700, 500)
    target = np.zeros(500)
    df = compute_score_bins(scores, target, n_bins=5)
    assert (df["bad"] == 0).all()
    assert (df["bad_rate"] == 0).all()


def test_compute_lift_gains():
    rng = np.random.default_rng(42)
    n = 3000
    scores = rng.normal(600, 80, size=n)
    target = (rng.random(n) < 0.08).astype(float)
    result = compute_lift_gains(scores, target, n_bins=10)
    assert "bins" in result
    assert result["total_good"] + result["total_bad"] == n
    assert len(result["bins"]) == 10
    # Last bin should have cumulative close to 1.0
    assert result["bins"][-1]["cumulative_bad_pct"] == pytest.approx(1.0, abs=0.01)


def test_compute_lift_gains_monotone_capture():
    rng = np.random.default_rng(42)
    n = 2000
    scores = rng.normal(600, 80, size=n)
    target = (rng.random(n) < 0.08).astype(float)
    result = compute_lift_gains(scores, target, n_bins=10)
    caps = [b["cumulative_bad_pct"] for b in result["bins"]]
    # Should be monotonically non-decreasing
    for i in range(1, len(caps)):
        assert caps[i] >= caps[i - 1] - 1e-6


def test_compute_scorecard_summary():
    rng = np.random.default_rng(42)
    n = 1000
    scores = rng.normal(600, 80, size=n)
    target = (rng.random(n) < 0.08).astype(float)
    summary = compute_scorecard_summary(scores, target)
    assert summary["total"] == n
    assert summary["score_mean"] > 0
    assert summary["ks_stat"] >= 0
    assert summary["good_mean_score"] != summary["bad_mean_score"] or n == 0
