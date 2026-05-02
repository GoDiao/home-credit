"""Scorecard module — PD-to-Score mapping, score binning, and Lift/Gains analysis.

Standard industry formula:
    Score = BaseScore - PDO / ln(2) * ln(PD / (1 - PD) * BaseOdds)

At BaseOdds the score equals BaseScore. Every PDO points, the odds double.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from config import BASE_SCORE, BASE_ODDS, PDO, SCORE_MIN, SCORE_MAX

logger = logging.getLogger(__name__)

# Pre-compute scaling factor
_FACTOR = PDO / np.log(2)


def pd_to_score(
    pd_values: np.ndarray | pd.Series,
    base_score: float = BASE_SCORE,
    base_odds: float = BASE_ODDS,
    pdo: float = PDO,
    score_min: float = SCORE_MIN,
    score_max: float = SCORE_MAX,
) -> np.ndarray:
    """Convert PD (probability of default) to credit score.

    Higher score = lower risk.  Uses the standard logistic scaling.
    """
    factor = pdo / np.log(2)
    offset = base_score + factor * np.log(base_odds)

    pd_arr = np.asarray(pd_values, dtype=np.float64)
    pd_clipped = np.clip(pd_arr, 1e-10, 1 - 1e-10)
    odds = (1 - pd_clipped) / pd_clipped  # good-to-bad odds

    scores = base_score + factor * (np.log(odds) - np.log(base_odds))
    return np.clip(scores, score_min, score_max)


def score_to_pd(
    scores: np.ndarray | pd.Series,
    base_score: float = BASE_SCORE,
    base_odds: float = BASE_ODDS,
    pdo: float = PDO,
) -> np.ndarray:
    """Convert credit score back to PD (inverse of pd_to_score)."""
    factor = pdo / np.log(2)
    s = np.asarray(scores, dtype=np.float64)
    log_odds = (s - base_score) / factor + np.log(base_odds)
    odds = np.exp(log_odds)
    return 1 / (1 + odds)


def compute_score_bins(
    scores: np.ndarray,
    target: np.ndarray,
    n_bins: int = 20,
) -> pd.DataFrame:
    """Bin scores and compute per-bin statistics.

    Returns DataFrame with columns:
        bin_idx, score_min, score_max, count, good, bad, bad_rate,
        cumulative_good, cumulative_bad, cumulative_bad_rate,
        good_pct, bad_pct, ks
    """
    scores = np.asarray(scores, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    edges = np.linspace(scores.min() - 1e-6, scores.max() + 1e-6, n_bins + 1)
    bin_idx = np.digitize(scores, edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    rows = []
    for i in range(n_bins):
        mask = bin_idx == i
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        bad = int(target[mask].sum())
        good = cnt - bad
        rows.append({
            "bin_idx": i,
            "score_min": float(scores[mask].min()),
            "score_max": float(scores[mask].max()),
            "count": cnt,
            "good": good,
            "bad": bad,
            "bad_rate": bad / cnt if cnt else 0,
        })

    df = pd.DataFrame(rows)

    # Cumulative stats (from lowest score = highest risk)
    df = df.sort_values("score_min").reset_index(drop=True)
    df["cumulative_good"] = df["good"].cumsum()
    df["cumulative_bad"] = df["bad"].cumsum()
    df["cumulative_total"] = df["count"].cumsum()
    df["cumulative_bad_rate"] = df["cumulative_bad"] / df["cumulative_total"]

    total_good = int(df["good"].sum())
    total_bad = int(df["bad"].sum())

    # Lift components
    df["good_pct"] = df["cumulative_good"] / total_good if total_good else 0
    df["bad_pct"] = df["cumulative_bad"] / total_bad if total_bad else 0
    df["ks"] = np.abs(df["good_pct"] - df["bad_pct"])

    return df


def compute_lift_gains(
    scores: np.ndarray,
    target: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute Lift, Gains, and Capture Rate by score decile.

    Returns dict with keys: bins, total_good, total_bad, population_bad_rate.
    """
    scores = np.asarray(scores, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    total = len(scores)
    total_bad = int(target.sum())
    total_good = total - total_bad
    pop_bad_rate = total_bad / total if total else 0

    # Sort by score ascending (lowest score = highest risk)
    order = np.argsort(scores)
    sorted_target = target[order]
    sorted_scores = scores[order]

    bin_size = max(1, total // n_bins)
    rows = []
    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else total
        if start >= total:
            break
        chunk = sorted_target[start:end]
        cnt = len(chunk)
        bad = int(chunk.sum())
        good = cnt - bad
        bad_rate = bad / cnt if cnt else 0
        lift = bad_rate / pop_bad_rate if pop_bad_rate else 0

        rows.append({
            "decile": i + 1,
            "count": cnt,
            "good": good,
            "bad": bad,
            "bad_rate": bad_rate,
            "lift": lift,
            "score_min": float(sorted_scores[start:end].min()),
            "score_max": float(sorted_scores[start:end].max()),
            "cumulative_bad_pct": 0.0,
            "cumulative_good_pct": 0.0,
        })

    df = pd.DataFrame(rows)

    # Cumulative capture rates
    df["cumulative_bad"] = df["bad"].cumsum()
    df["cumulative_good"] = df["good"].cumsum()
    df["cumulative_bad_pct"] = df["cumulative_bad"] / total_bad if total_bad else 0
    df["cumulative_good_pct"] = df["cumulative_good"] / total_good if total_good else 0

    return {
        "bins": df.to_dict(orient="records"),
        "total_good": total_good,
        "total_bad": total_bad,
        "population_bad_rate": pop_bad_rate,
        "n_bins": n_bins,
    }


def compute_scorecard_summary(
    scores: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    """High-level scorecard statistics for dashboard display."""
    scores = np.asarray(scores, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    good_mask = target == 0
    bad_mask = target == 1

    return {
        "total": len(scores),
        "total_good": int(good_mask.sum()),
        "total_bad": int(bad_mask.sum()),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "score_median": float(np.median(scores)),
        "score_p25": float(np.percentile(scores, 25)),
        "score_p75": float(np.percentile(scores, 75)),
        "good_mean_score": float(scores[good_mask].mean()) if good_mask.any() else 0,
        "bad_mean_score": float(scores[bad_mask].mean()) if bad_mask.any() else 0,
        "ks_stat": _compute_ks(scores, target),
    }


def _compute_ks(scores: np.ndarray, target: np.ndarray) -> float:
    """Compute KS statistic from scores and binary target."""
    good_mask = target == 0
    bad_mask = target == 1
    if not good_mask.any() or not bad_mask.any():
        return 0.0

    thresholds = np.sort(np.unique(scores))
    ks_values = []
    n_good = good_mask.sum()
    n_bad = bad_mask.sum()

    for t in thresholds:
        tpr = (scores[bad_mask] <= t).sum() / n_bad  # bad captured below
        fpr = (scores[good_mask] <= t).sum() / n_good  # good captured below
        ks_values.append(abs(tpr - fpr))

    return float(max(ks_values)) if ks_values else 0.0
