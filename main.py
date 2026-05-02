"""Project pipeline CLI for the Home Credit risk modeling project."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path
from typing import Callable

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR, REPORT_DIR, setup_logging


PROJECT_ROOT = Path(__file__).resolve().parent


def _require_file(path: Path, hint: str) -> None:
    """Fail fast with an actionable message when an upstream artifact is missing."""
    if not path.exists():
        raise FileNotFoundError(f"缺少必要文件: {path}\n提示: {hint}")


def _run_module_main(module_name: str) -> None:
    """Import a project module and execute its main() function."""
    module = __import__(module_name)
    main_func = getattr(module, "main", None)
    if main_func is None:
        raise AttributeError(f"模块 {module_name} 没有 main() 函数")
    main_func()


def run_process() -> None:
    """Run raw application data processing."""
    _require_file(
        RAW_DATA_DIR / "application_train.csv",
        "请先将 Kaggle 原始数据放入 data/raw/，至少需要 application_train.csv。",
    )
    _run_module_main("data_processing")


def run_features() -> None:
    """Run feature engineering on processed application data."""
    _require_file(
        PROCESSED_DATA_DIR / "train_processed.csv",
        "请先运行: uv run python main.py --stage process",
    )
    _run_module_main("feature_engineering")


def run_train() -> None:
    """Run the main PD model training flow."""
    _require_file(
        PROCESSED_DATA_DIR / "train_with_features.csv",
        "请先运行: uv run python main.py --stage features",
    )
    _run_module_main("pd_model")


def run_lightgbm() -> None:
    """Run the standalone LightGBM training script."""
    _require_file(
        PROCESSED_DATA_DIR / "train_with_features.csv",
        "请先运行: uv run python main.py --stage features",
    )
    runpy.run_path(str(PROJECT_ROOT / "train_lightgbm.py"), run_name="__main__")


def run_stacking() -> None:
    """Run the standalone Stacking Ensemble training script."""
    _require_file(
        PROCESSED_DATA_DIR / "train_with_features.csv",
        "请先运行: uv run python main.py --stage features",
    )
    _require_file(
        PROJECT_ROOT / "outputs" / "models" / "pd_model_lightgbm.pkl",
        "Stacking 脚本需要已训练 LightGBM，请先运行: uv run python main.py --stage lightgbm",
    )
    runpy.run_path(str(PROJECT_ROOT / "train_stacking.py"), run_name="__main__")


def run_policy() -> None:
    """Run cut-off policy simulation."""
    _require_file(
        PROCESSED_DATA_DIR / "train_with_features.csv",
        "请先运行: uv run python main.py --stage features",
    )
    _run_module_main("policy_simulation")


def run_monitoring() -> None:
    """Run portfolio monitoring reports."""
    _require_file(
        PROCESSED_DATA_DIR / "train_processed.csv",
        "请先运行: uv run python main.py --stage process",
    )
    _run_module_main("portfolio_monitoring")


def run_api() -> None:
    """Start the FastAPI development server on the standard port 8000."""
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)


def run_all() -> None:
    """Run the full reproducible pipeline in dependency order."""
    run_process()
    run_features()
    run_train()
    run_policy()
    run_monitoring()


STAGE_RUNNERS: dict[str, Callable[[], None]] = {
    "process": run_process,
    "features": run_features,
    "train": run_train,
    "lightgbm": run_lightgbm,
    "stacking": run_stacking,
    "policy": run_policy,
    "monitoring": run_monitoring,
    "api": run_api,
    "all": run_all,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Home Credit 风控建模 Pipeline CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        choices=sorted(STAGE_RUNNERS),
        default="all",
        help="要执行的流水线阶段",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = parse_args(argv)
    print(f"\n🚀 Running stage: {args.stage}")
    STAGE_RUNNERS[args.stage]()
    print(f"\n✅ Stage completed: {args.stage}")
    print(f"📁 Reports directory: {REPORT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
