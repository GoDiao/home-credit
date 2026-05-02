"""utils.py 单元测试"""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    calculate_ks, calculate_gini, calculate_all_metrics,
    calculate_psi, calculate_woe_iv,
    plot_roc_curve, plot_ks_curve, plot_gini_curve
)


class TestCalculateKS:
    def test_perfect_separation(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        ks = calculate_ks(y_true, y_pred)
        assert ks > 0.9

    def test_random_predictions(self):
        np.random.seed(42)
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0.4, 0.5, 0.45, 0.55, 0.5, 0.45, 0.55, 0.5])
        ks = calculate_ks(y_true, y_pred)
        assert 0 <= ks <= 1

    def test_all_same_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.5, 0.5, 0.5, 0.5])
        ks = calculate_ks(y_true, y_pred)
        assert ks == 0.0


class TestCalculateGini:
    def test_gini_from_auc(self):
        assert calculate_gini(0.5) == 0.0
        assert calculate_gini(1.0) == 1.0
        assert calculate_gini(0.75) == 0.5

    def test_gini_range(self):
        assert calculate_gini(0.8) == pytest.approx(0.6)


class TestCalculateAllMetrics:
    def test_returns_dict(self):
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.6, 0.4, 0.9])
        metrics = calculate_all_metrics(y_true, y_pred)
        assert isinstance(metrics, dict)
        assert 'AUC' in metrics
        assert 'KS' in metrics
        assert 'Gini' in metrics

    def test_metrics_range(self):
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.6, 0.4, 0.9])
        metrics = calculate_all_metrics(y_true, y_pred)
        assert 0 <= metrics['AUC'] <= 1
        assert 0 <= metrics['KS'] <= 1
        assert 0 <= metrics['Gini'] <= 1


class TestPSI:
    def test_same_distribution(self):
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 1000))
        psi = calculate_psi(data, data, bins=10)
        assert psi == pytest.approx(0.0, abs=0.01)

    def test_different_distribution(self):
        np.random.seed(42)
        expected = pd.Series(np.random.normal(0, 1, 1000))
        actual = pd.Series(np.random.normal(2, 1, 1000))
        psi = calculate_psi(expected, actual, bins=10)
        assert psi > 0.1

    def test_zero_variance(self):
        expected = pd.Series([1.0] * 100)
        actual = pd.Series([1.0] * 100)
        psi = calculate_psi(expected, actual, bins=10)
        assert psi == 0.0


class TestWOEIV:
    def test_returns_tuple(self):
        np.random.seed(42)
        df = pd.DataFrame({
            'feature': np.random.normal(0, 1, 500),
            'target': np.random.binomial(1, 0.2, 500)
        })
        result, iv = calculate_woe_iv(df, 'feature', 'target', bins=5)
        assert isinstance(result, pd.DataFrame)
        assert isinstance(iv, float)
        assert iv >= 0

    def test_strong_predictor(self):
        df = pd.DataFrame({
            'feature': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            'target': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        })
        _, iv = calculate_woe_iv(df, 'feature', 'target', bins=3)
        assert iv > 0.1


class TestPlotGiniCurve:
    def test_creates_plot(self, tmp_path):
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.6, 0.4, 0.9])
        save_path = str(tmp_path / "gini.png")
        import matplotlib
        matplotlib.use('Agg')
        plot_gini_curve(y_true, y_pred, save_path=save_path)
        assert (tmp_path / "gini.png").exists()
