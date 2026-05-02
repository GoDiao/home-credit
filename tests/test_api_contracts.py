"""API contract tests — verify service functions return correct schema.

Uses direct function calls instead of TestClient (avoids httpx dependency).
Each test validates that the response dict contains expected keys and value types.

Note: Scorecard/model tests only use "xgboost" to avoid slow repeated model loading.
"""
import pytest

from api.services import (
    get_overview,
    get_model_metrics,
    get_feature_importance,
    get_psi_data,
    get_model_registry,
    get_scorecard_summary,
    get_scorecard_distribution,
    get_scorecard_lift,
    get_monitoring_health,
)


# =========================================================
# /api/overview
# =========================================================
class TestOverview:
    def test_returns_required_keys(self):
        data = get_overview()
        for key in ["n_samples", "n_features", "n_default", "default_rate",
                     "models", "pipeline", "target_distribution", "dataset_info"]:
            assert key in data, f"Missing key: {key}"

    def test_models_is_list(self):
        data = get_overview()
        assert isinstance(data["models"], list)

    def test_pipeline_is_list(self):
        data = get_overview()
        assert isinstance(data["pipeline"], list)
        assert len(data["pipeline"]) > 0

    def test_numeric_fields(self):
        data = get_overview()
        assert isinstance(data["n_samples"], (int, float))
        assert isinstance(data["default_rate"], (int, float))


# =========================================================
# /api/model/{type}/metrics — only xgboost (fast, already cached)
# =========================================================
class TestModelMetrics:
    def test_returns_metrics_keys(self):
        data = get_model_metrics("xgboost")
        for key in ["model_type", "threshold", "auc", "ks", "gini", "accuracy",
                     "precision", "recall", "f1", "confusion_matrix"]:
            assert key in data, f"Missing key: {key}"

    def test_metrics_range(self):
        data = get_model_metrics("xgboost")
        assert 0 <= data["auc"] <= 1
        assert 0 <= data["ks"] <= 1
        assert 0 <= data["gini"] <= 1

    def test_confusion_matrix_keys(self):
        data = get_model_metrics("xgboost")
        cm = data["confusion_matrix"]
        for key in ["tp", "tn", "fp", "fn", "tp_pct", "fp_pct", "fn_pct", "tn_pct"]:
            assert key in cm, f"Missing CM key: {key}"


# =========================================================
# /api/features/importance
# =========================================================
class TestFeatureImportance:
    def test_returns_required_keys(self):
        data = get_feature_importance("xgboost", top_n=10)
        assert "features" in data
        assert "importance" in data
        assert isinstance(data["features"], list)
        assert isinstance(data["importance"], list)

    def test_top_n_respected(self):
        data = get_feature_importance("xgboost", top_n=5)
        assert len(data["features"]) <= 5


# =========================================================
# /api/monitoring/psi
# =========================================================
class TestPSIData:
    def test_returns_required_keys(self):
        data = get_psi_data(0.25)
        for key in ["total_features", "stable_count", "marginal_count",
                     "unstable_count", "features", "psi_values", "threshold"]:
            assert key in data, f"Missing key: {key}"

    def test_lists_same_length(self):
        data = get_psi_data(0.25)
        assert len(data["features"]) == len(data["psi_values"])

    def test_counts_add_up(self):
        data = get_psi_data(0.25)
        assert data["stable_count"] + data["marginal_count"] + data["unstable_count"] == data["total_features"]

    def test_threshold_echoed(self):
        data = get_psi_data(0.15)
        assert data["threshold"] == 0.15


# =========================================================
# /api/monitoring/health
# =========================================================
class TestMonitoringHealth:
    def test_returns_required_keys(self):
        data = get_monitoring_health(0.25)
        for key in ["status", "status_label", "color", "total_features",
                     "stable_count", "marginal_count", "unstable_count",
                     "stable_pct", "marginal_pct", "unstable_pct",
                     "top_drift", "alerts", "threshold"]:
            assert key in data, f"Missing key: {key}"

    def test_status_values(self):
        data = get_monitoring_health(0.25)
        assert data["status"] in ("healthy", "warning", "critical")
        assert data["color"] in ("emerald", "amber", "rose")

    def test_alerts_structure(self):
        data = get_monitoring_health(0.25)
        assert isinstance(data["alerts"], list)
        for alert in data["alerts"]:
            assert "level" in alert
            assert "message" in alert
            assert "action" in alert

    def test_top_drift_structure(self):
        data = get_monitoring_health(0.25)
        assert isinstance(data["top_drift"], list)
        for item in data["top_drift"]:
            assert "feature" in item
            assert "psi" in item


# =========================================================
# /api/models/registry
# =========================================================
class TestModelRegistry:
    def test_returns_registry(self):
        data = get_model_registry()
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_model_record_fields(self):
        data = get_model_registry()
        if not data["models"]:
            pytest.skip("No models registered")
        m = data["models"][0]
        for key in ["model_name", "model_type", "feature_count", "metrics",
                     "trained_at", "code_version"]:
            assert key in m, f"Missing key: {key} in model record"


# =========================================================
# /api/scorecard/* — only xgboost to avoid slow repeated loads
# =========================================================
class TestScorecardSummary:
    def test_returns_summary_keys(self):
        data = get_scorecard_summary("xgboost")
        for key in ["total", "total_good", "total_bad", "score_mean", "score_std",
                     "score_min", "score_max", "ks_stat", "good_mean_score", "bad_mean_score"]:
            assert key in data, f"Missing key: {key}"


class TestScorecardDistribution:
    def test_returns_distribution_keys(self):
        data = get_scorecard_distribution("xgboost", n_bins=10)
        assert "bins" in data
        assert "config" in data
        assert isinstance(data["bins"], list)

    def test_bin_fields(self):
        data = get_scorecard_distribution("xgboost", n_bins=10)
        if not data["bins"]:
            pytest.skip("No bins")
        b = data["bins"][0]
        for key in ["score_min", "score_max", "count", "good", "bad", "bad_rate"]:
            assert key in b, f"Missing bin key: {key}"

    def test_config_fields(self):
        data = get_scorecard_distribution("xgboost", n_bins=10)
        cfg = data["config"]
        for key in ["base_score", "base_odds", "pdo", "score_min", "score_max"]:
            assert key in cfg, f"Missing config key: {key}"


class TestScorecardLift:
    def test_returns_lift_keys(self):
        data = get_scorecard_lift("xgboost", n_bins=10)
        for key in ["bins", "total_good", "total_bad", "population_bad_rate", "n_bins"]:
            assert key in data, f"Missing key: {key}"

    def test_bin_fields(self):
        data = get_scorecard_lift("xgboost", n_bins=10)
        if not data["bins"]:
            pytest.skip("No bins")
        b = data["bins"][0]
        for key in ["decile", "count", "good", "bad", "bad_rate", "lift",
                     "cumulative_bad_pct", "cumulative_good_pct"]:
            assert key in b, f"Missing lift bin key: {key}"
