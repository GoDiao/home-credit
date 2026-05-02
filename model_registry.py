"""Lightweight model registry utilities.

The registry is intentionally file-based so it works in local notebooks,
scripts, FastAPI, and the dashboard without adding infrastructure.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from config import MODEL_DIR, RANDOM_STATE, TARGET


REGISTRY_PATH = MODEL_DIR / "model_registry.json"


def _json_safe(value: Any) -> Any:
    """Convert numpy/path values into JSON-safe Python primitives."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return value


def get_git_commit() -> str:
    """Return current git commit if available, otherwise ``unknown``."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def features_hash(feature_names: Iterable[str] | None) -> str | None:
    """Create a stable hash for the ordered feature list."""
    if not feature_names:
        return None
    payload = "\n".join(str(f) for f in feature_names)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    """Load the registry file, returning an empty structure if absent."""
    if not path.exists():
        return {"version": 1, "models": []}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "models" not in data or not isinstance(data["models"], list):
        data["models"] = []
    data.setdefault("version", 1)
    return data


def save_registry(registry: dict[str, Any], path: Path = REGISTRY_PATH) -> None:
    """Persist registry JSON with deterministic formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(registry), f, ensure_ascii=False, indent=2)


def build_model_record(
    *,
    model_name: str,
    model_type: str,
    model_path: str | Path,
    feature_names: Iterable[str] | None = None,
    metrics: dict[str, Any] | None = None,
    dataset_rows: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a normalized model registry record."""
    features = list(feature_names or [])
    path = Path(model_path)
    record = {
        "model_name": model_name,
        "model_type": model_type,
        "model_path": str(path),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_count": len(features),
        "features_hash": features_hash(features),
        "metrics": metrics or {},
        "dataset_rows": dataset_rows,
        "target": TARGET,
        "random_state": RANDOM_STATE,
        "code_version": get_git_commit(),
    }
    if path.exists():
        stat = path.stat()
        record["artifact"] = {
            "size_mb": round(stat.st_size / (1024 * 1024), 3),
            "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        }
    if extra:
        record["extra"] = extra
    return _json_safe(record)


def upsert_model_record(record: dict[str, Any], path: Path = REGISTRY_PATH) -> dict[str, Any]:
    """Insert or update a model record by ``model_type``."""
    registry = load_registry(path)
    models = [m for m in registry.get("models", []) if m.get("model_type") != record.get("model_type")]
    models.append(record)
    models.sort(key=lambda m: str(m.get("model_name", m.get("model_type", ""))))
    registry["models"] = models
    registry["updated_at"] = datetime.now(timezone.utc).isoformat()
    save_registry(registry, path)
    return registry


def register_model(**kwargs: Any) -> dict[str, Any]:
    """Convenience helper: build and upsert a model record."""
    record = build_model_record(**kwargs)
    upsert_model_record(record)
    return record