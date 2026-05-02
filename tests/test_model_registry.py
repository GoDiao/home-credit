"""model_registry.py 单元测试"""
import json
from pathlib import Path

import numpy as np

from model_registry import (
    _json_safe,
    build_model_record,
    features_hash,
    get_git_commit,
    load_registry,
    register_model,
    save_registry,
    upsert_model_record,
)


def test_features_hash_is_stable_and_order_sensitive():
    assert features_hash(["a", "b"]) == features_hash(["a", "b"])
    assert features_hash(["a", "b"]) != features_hash(["b", "a"])


def test_features_hash_none_and_empty():
    assert features_hash(None) is None
    assert features_hash([]) is None


def test_build_model_record_json_safe(tmp_path):
    artifact = tmp_path / "model.pkl"
    artifact.write_bytes(b"model")
    record = build_model_record(
        model_name="Demo",
        model_type="demo",
        model_path=artifact,
        feature_names=["f1", "f2"],
        metrics={"AUC": 0.75},
        dataset_rows=100,
    )
    assert record["model_name"] == "Demo"
    assert record["feature_count"] == 2
    assert record["features_hash"]
    assert record["artifact"]["size_mb"] >= 0
    json.dumps(record)


def test_build_model_record_no_artifact(tmp_path):
    record = build_model_record(
        model_name="Ghost",
        model_type="ghost",
        model_path=tmp_path / "nonexistent.pkl",
        feature_names=None,
        metrics=None,
        dataset_rows=None,
    )
    assert record["feature_count"] == 0
    assert record["features_hash"] is None
    assert "artifact" not in record
    assert record["metrics"] == {}
    assert record["dataset_rows"] is None


def test_build_model_record_with_extra(tmp_path):
    artifact = tmp_path / "m.pkl"
    artifact.write_bytes(b"x")
    record = build_model_record(
        model_name="E",
        model_type="e",
        model_path=artifact,
        extra={"calibration_method": "isotonic"},
    )
    assert record["extra"]["calibration_method"] == "isotonic"


def test_upsert_model_record_replaces_same_model_type(tmp_path):
    path = tmp_path / "registry.json"
    first = {"model_name": "Demo v1", "model_type": "demo", "metrics": {"AUC": 0.7}}
    second = {"model_name": "Demo v2", "model_type": "demo", "metrics": {"AUC": 0.8}}

    upsert_model_record(first, path=path)
    registry = upsert_model_record(second, path=path)

    assert path.exists()
    assert len(registry["models"]) == 1
    assert registry["models"][0]["model_name"] == "Demo v2"

    loaded = load_registry(path)
    assert loaded["models"][0]["metrics"]["AUC"] == 0.8


def test_upsert_multiple_model_types(tmp_path):
    path = tmp_path / "registry.json"
    upsert_model_record({"model_name": "A", "model_type": "a", "metrics": {}}, path=path)
    upsert_model_record({"model_name": "B", "model_type": "b", "metrics": {}}, path=path)
    registry = load_registry(path)
    assert len(registry["models"]) == 2
    names = {m["model_name"] for m in registry["models"]}
    assert names == {"A", "B"}


def test_upsert_sets_updated_at(tmp_path):
    path = tmp_path / "registry.json"
    upsert_model_record({"model_name": "X", "model_type": "x", "metrics": {}}, path=path)
    registry = load_registry(path)
    assert "updated_at" in registry


def test_register_model_returns_record(tmp_path):
    artifact = tmp_path / "model.pkl"
    artifact.write_bytes(b"test")
    record = register_model(
        model_name="Convenience",
        model_type="conv",
        model_path=artifact,
        feature_names=["a"],
        metrics={"AUC": 0.9},
    )
    assert record["model_name"] == "Convenience"
    assert record["model_type"] == "conv"
    assert record["feature_count"] == 1
    assert record["metrics"]["AUC"] == 0.9


def test_save_and_load_empty_registry(tmp_path):
    path = tmp_path / "registry.json"
    save_registry({"version": 1, "models": []}, path=path)
    assert load_registry(path) == {"version": 1, "models": []}


def test_load_registry_missing_file(tmp_path):
    path = tmp_path / "nope.json"
    result = load_registry(path)
    assert result == {"version": 1, "models": []}


def test_load_registry_corrupt_models_field(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"version": 1, "models": "not_a_list"}')
    result = load_registry(path)
    assert result["models"] == []


def test_json_safe_numpy():
    assert _json_safe(np.int64(5)) == 5
    assert _json_safe(np.float64(3.14)) == 3.14
    assert _json_safe(np.nan) is None
    assert _json_safe(np.inf) is None
    assert _json_safe(Path("/tmp/x")) == str(Path("/tmp/x"))
    assert _json_safe([np.int64(1), np.float64(2.0)]) == [1, 2.0]
    assert _json_safe({"k": np.int64(3)}) == {"k": 3}


def test_get_git_commit():
    result = get_git_commit()
    assert isinstance(result, str)
    assert len(result) >= 4 or result == "unknown"