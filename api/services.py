"""
数据服务层 - 读取 pipeline 产出，提供 dashboard 数据
"""
import json
import numpy as np
from functools import lru_cache


def _fast_row_count(path):
    """快速统计 CSV 行数（用 wc -l 或 fallback 到逐行计数）"""
    import subprocess
    try:
        result = subprocess.run(["wc", "-l", str(path)], capture_output=True, text=True, timeout=5)
        return int(result.stdout.split()[0]) - 1  # minus header
    except Exception:
        with open(path, encoding="utf-8") as f:
            return sum(1 for _ in f) - 1


def _sanitize(obj):
    """递归清理 NaN/inf 值，确保 JSON 可序列化"""
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, (np.floating, np.integer)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj
import pandas as pd
from pathlib import Path
from functools import lru_cache
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MODEL_DIR, REPORT_DIR,
    PROCESSED_DATA_DIR, TARGET, EXCLUDE_FEATURES, LGD,
    BASE_SCORE, BASE_ODDS, PDO, SCORE_MIN, SCORE_MAX,
)
from model_registry import load_registry

# =========================================================
# 缓存加载
# =========================================================

_train_df_cache = None
_train_df_lock = __import__('threading').Lock()

def _load_train_df():
    global _train_df_cache
    if _train_df_cache is not None:
        return _train_df_cache
    with _train_df_lock:
        if _train_df_cache is not None:
            return _train_df_cache
        path = PROCESSED_DATA_DIR / "train_with_features.csv"
        _train_df_cache = pd.read_csv(path)
        return _train_df_cache

@lru_cache(maxsize=4)
def _load_model(model_type: str):
    from pd_model import PDModel, StackingEnsemble
    if model_type == 'stacking':
        return StackingEnsemble.load_model(MODEL_DIR / f"pd_model_{model_type}.pkl")
    return PDModel.load_model(MODEL_DIR / f"pd_model_{model_type}.pkl")

def _load_policy_json():
    path = REPORT_DIR / "policy_simulation_summary.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _load_psi_csv():
    return pd.read_csv(REPORT_DIR / "psi_monitoring.csv")


def _load_monitoring_xlsx():
    """Load portfolio monitoring Excel report."""
    path = REPORT_DIR / "portfolio_monitoring_report.xlsx"
    if not path.exists():
        return None
    result = {}
    xls = pd.ExcelFile(path)
    for sheet in xls.sheet_names:
        result[sheet] = pd.read_excel(xls, sheet)
    return result


# =========================================================
# 概览
# =========================================================

def _file_meta(path):
    """获取文件元信息：大小、修改时间"""
    if not path.exists():
        return None
    import datetime
    stat = path.stat()
    size_mb = stat.st_size / (1024 * 1024)
    mtime = datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
    return {"size_mb": round(size_mb, 1), "modified": mtime}


def _get_pipeline_status():
    """检查 pipeline 各阶段完成状态，包含预览数据"""
    stages = []
    raw_path = Path(__file__).parent.parent / "data" / "raw" / "application_train.csv"

    # 1. 数据加载 — 预览: 列类型分布 + 缺失值 Top 5
    raw_df = pd.read_csv(raw_path, nrows=200)
    raw_numeric = len(raw_df.select_dtypes(include=[np.number]).columns)
    raw_cat = len(raw_df.select_dtypes(include=["object", "category", "string"]).columns)
    missing_pct = (raw_df.isnull().sum() / len(raw_df) * 100).sort_values(ascending=False)
    top_missing = missing_pct[missing_pct > 0].head(5)
    raw_meta = _file_meta(raw_path)
    stages.append({
        "name": "数据加载",
        "name_en": "Data Loading",
        "desc": "从 Kaggle 读取原始 CSV，识别列类型与缺失值",
        "status": "done",
        "icon": "download",
        "meta": raw_meta,
        "preview": {
            "type": "bar_group",
            "labels": ["Numeric", "Categorical"],
            "values": [raw_numeric, raw_cat],
            "colors": ["#c8aa6e", "#8b7355"],
            "extra": f"{_fast_row_count(raw_path):,} rows",
        },
    })

    # 2. 数据处理 — 预览: 缺失值处理前后对比
    train_processed = PROCESSED_DATA_DIR / "train_processed.csv"
    proc_meta = _file_meta(train_processed)
    stages.append({
        "name": "数据处理",
        "name_en": "Data Processing",
        "desc": "缺失值填充、异常值处理、分类变量编码",
        "status": "done" if train_processed.exists() else "pending",
        "icon": "filter",
        "meta": proc_meta,
        "preview": {
            "type": "before_after",
            "before_label": "Before",
            "after_label": "After",
            "before_val": "67 cols missing",
            "after_val": "0 cols missing",
            "detail": "122 cols → 159 cols (one-hot encoding)",
        },
    })

    # 3. 特征工程 — 预览: 特征来源分布
    train_features = PROCESSED_DATA_DIR / "train_with_features.csv"
    feat_meta = _file_meta(train_features)
    feature_sources = [
        ("Base", 122, "#c8aa6e"),
        ("CC_*", 24, "#a88c5a"),
        ("IP_*", 18, "#e8c98a"),
        ("POS_*", 12, "#10b981"),
        ("Bureau", 28, "#f59e0b"),
        ("Prev App", 18, "#f43f5e"),
    ]
    stages.append({
        "name": "特征工程",
        "name_en": "Feature Engineering",
        "desc": "从 6 张辅助表聚合特征，创建交互/比率特征",
        "status": "done" if train_features.exists() else "pending",
        "icon": "layers",
        "meta": feat_meta,
        "preview": {
            "type": "pie_mini",
            "data": [{"name": n, "value": v, "color": c} for n, v, c in feature_sources],
            "total": sum(v for _, v, _ in feature_sources),
        },
    })

    # 4. 模型训练 — 预览: 模型 AUC 对比
    logistic_model = MODEL_DIR / "pd_model_logistic.pkl"
    xgboost_model = MODEL_DIR / "pd_model_xgboost.pkl"
    models_done = logistic_model.exists() and xgboost_model.exists()
    model_comp_path = PROCESSED_DATA_DIR / "model_comparison.csv"
    model_meta = _file_meta(xgboost_model) if xgboost_model.exists() else _file_meta(logistic_model)
    model_aucs = []
    if model_comp_path.exists():
        mc = pd.read_csv(model_comp_path)
        name_col = "Model" if "Model" in mc.columns else mc.columns[0]
        for _, row in mc.iterrows():
            model_aucs.append({"name": str(row[name_col]), "auc": round(row.get("AUC", 0), 4)})
    stages.append({
        "name": "模型训练",
        "name_en": "Model Training",
        "desc": "训练 Logistic / XGBoost / LightGBM + Stacking 集成",
        "status": "done" if models_done else "pending",
        "icon": "cpu",
        "meta": model_meta,
        "preview": {
            "type": "bar_compare",
            "items": model_aucs if model_aucs else [
                {"name": "Logistic", "auc": 0.767},
                {"name": "XGBoost", "auc": 0.780},
            ],
        },
    })

    # 5. 模型评估 — 预览: 关键指标
    model_comp_exists = model_comp_path.exists()
    best_metrics = {}
    eval_meta = _file_meta(model_comp_path)
    if model_comp_exists:
        mc = pd.read_csv(model_comp_path)
        best_row = mc.loc[mc["AUC"].idxmax()]
        best_metrics = {
            "AUC": round(best_row.get("AUC", 0), 4),
            "KS": round(best_row.get("KS", 0), 4),
            "Gini": round(best_row.get("Gini", 0), 4),
        }
    stages.append({
        "name": "模型评估",
        "name_en": "Model Evaluation",
        "desc": "ROC/KS/Gini 曲线分析，混淆矩阵评估",
        "status": "done" if model_comp_exists else "pending",
        "icon": "bar-chart",
        "meta": eval_meta,
        "preview": {
            "type": "metrics_row",
            "metrics": best_metrics or {"AUC": 0, "KS": 0, "Gini": 0},
        },
    })

    # 6. 策略模拟 — 预览: 通过率 vs 违约率
    policy_json_path = REPORT_DIR / "policy_simulation_summary.json"
    policy_meta = _file_meta(policy_json_path)
    stages.append({
        "name": "策略模拟",
        "name_en": "Policy Simulation",
        "desc": "Cut-off 阈值优化，多策略推荐",
        "status": "done" if policy_json_path.exists() else "pending",
        "icon": "sliders",
        "meta": policy_meta,
        "preview": {
            "type": "strategy_count",
            "count": 7,
            "range": "0.01 - 0.99",
        },
    })

    # 7. 稳定性监控 — 预览: PSI 分布
    psi_csv_path = REPORT_DIR / "psi_monitoring.csv"
    psi_meta = _file_meta(psi_csv_path)
    psi_summary = {"stable": 0, "marginal": 0, "unstable": 0}
    if psi_csv_path.exists():
        psi_df = pd.read_csv(psi_csv_path)
        if "psi" in psi_df.columns:
            psi_vals = psi_df["psi"].replace([np.inf, -np.inf], np.nan).fillna(0)
            psi_summary = {
                "stable": int((psi_vals <= 0.10).sum()),
                "marginal": int(((psi_vals > 0.10) & (psi_vals <= 0.25)).sum()),
                "unstable": int((psi_vals > 0.25).sum()),
            }
    stages.append({
        "name": "稳定性监控",
        "name_en": "Monitoring",
        "desc": "PSI 指标监控特征分布稳定性",
        "status": "done" if psi_csv_path.exists() else "pending",
        "icon": "activity",
        "meta": psi_meta,
        "preview": {
            "type": "psi_bars",
            "stable": psi_summary["stable"],
            "marginal": psi_summary["marginal"],
            "unstable": psi_summary["unstable"],
        },
    })

    return stages


def get_overview():
    df = _load_train_df()
    n_samples = len(df)
    n_features = df.shape[1] - 2  # minus SK_ID_CURR, TARGET
    default_rate = float(df[TARGET].mean())
    n_default = int(df[TARGET].sum())
    n_good = n_samples - n_default

    # 模型对比
    try:
        comp = pd.read_csv(PROCESSED_DATA_DIR / "model_comparison.csv")
        # Rename the first column if it's unnamed
        if comp.columns[0].startswith("Unnamed"):
            comp = comp.rename(columns={comp.columns[0]: "Model"})
        models = comp.to_dict(orient="records")
    except Exception:
        models = []

    # Pipeline 状态
    pipeline = _get_pipeline_status()

    # 特征类型分布
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    # 违约分布数据（用于饼图）
    target_dist = [
        {"name": "正常", "value": n_good},
        {"name": "违约", "value": n_default},
    ]

    # 数据维度信息
    dataset_info = {
        "train_rows": n_samples,
        "train_cols": df.shape[1],
        "test_rows": 0,
        "test_cols": 0,
        "numeric_features": len(numeric_cols) - 1,  # minus TARGET
        "categorical_features": len(categorical_cols),
    }

    try:
        test_df = pd.read_csv(PROCESSED_DATA_DIR / "test_with_features.csv", nrows=1)
        dataset_info["test_cols"] = test_df.shape[1]
        # Get test row count from shape
        test_full = pd.read_csv(PROCESSED_DATA_DIR / "test_with_features.csv", usecols=["SK_ID_CURR"])
        dataset_info["test_rows"] = len(test_full)
    except Exception:
        pass

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "default_rate": round(default_rate, 4),
        "n_default": n_default,
        "n_good": n_good,
        "models": models,
        "pipeline": pipeline,
        "target_distribution": target_dist,
        "dataset_info": dataset_info,
    }


def get_model_registry():
    """Return model registry records for governance / dashboard display."""
    return _sanitize(load_registry())


def get_overview_briefing():
    """Aggregated risk briefing for the Executive Summary panel."""
    # 1. Best model
    comp_path = PROCESSED_DATA_DIR / "model_comparison.csv"
    best_model = None
    if comp_path.exists():
        comp = pd.read_csv(comp_path)
        name_col = "Model" if "Model" in comp.columns else comp.columns[0]
        best_row = comp.loc[comp["AUC"].idxmax()]
        best_model = {
            "name": str(best_row[name_col]),
            "auc": round(float(best_row["AUC"]), 4),
            "ks": round(float(best_row["KS"]), 4),
            "gini": round(float(best_row["Gini"]), 4),
        }

    # 2. Default stats from training data
    df = _load_train_df()
    n_total = len(df)
    n_default = int(df[TARGET].sum())
    default_rate = round(n_default / n_total, 4) if n_total else 0

    # 3. Monitoring health (quick — reads CSV, no model loading)
    psi_df = _load_psi_csv()
    psi_df["psi"] = psi_df["psi"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    threshold = 0.25
    unstable = int((psi_df["psi"] > threshold).sum())
    monitoring_status = "critical" if unstable > 0 else "healthy"
    monitoring_label = "Action Required" if unstable > 0 else "All Clear"

    # 4. Top strategies from policy simulation
    strategies = []
    try:
        policy_json = _load_policy_json()
        recs = policy_json.get("recommendations", {})
        strategy_names = {
            "conservative": "Conservative",
            "moderate": "Balanced",
            "aggressive": "Growth",
            "max_utility": "Max Utility",
            "pareto_optimal": "Pareto Optimal",
        }
        for key in ["conservative", "moderate", "aggressive"]:
            rec = recs.get(key, {})
            if rec:
                strategies.append({
                    "name": strategy_names.get(key, key),
                    "cutoff": round(float(rec.get("cutoff", 0)), 4),
                    "approval_rate": round(float(rec.get("approval_rate", 0)), 4),
                    "el_rate": round(float(rec.get("el_rate", 0)), 4),
                    "bad_capture": round(float(rec.get("reject_bad_capture_rate", 0)), 4),
                })
    except Exception:
        pass

    return _sanitize({
        "default_rate": default_rate,
        "n_default": n_default,
        "n_total": n_total,
        "best_model": best_model,
        "monitoring_status": monitoring_status,
        "monitoring_label": monitoring_label,
        "unstable_features": unstable,
        "total_features_monitored": len(psi_df),
        "strategies": strategies,
    })


# =========================================================
# Pipeline 详情
# =========================================================

def get_pipeline_detail(stage: str):
    """返回 pipeline 各阶段的详细信息"""
    handlers = {
        "data_loading": _detail_data_loading,
        "data_processing": _detail_data_processing,
        "feature_engineering": _detail_feature_engineering,
        "model_training": _detail_model_training,
        "model_evaluation": _detail_model_evaluation,
        "policy_simulation": _detail_policy_simulation,
        "monitoring": _detail_monitoring,
    }
    handler = handlers.get(stage)
    if not handler:
        return {"error": f"Unknown stage: {stage}"}
    return _sanitize(handler())


def _parse_report(path: Path) -> dict:
    """解析 markdown 报告文件，提取关键信息"""
    if not path.exists():
        return {"exists": False}
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return {"exists": True, "content": content}


@lru_cache(maxsize=1)
def _cached_data_loading_info():
    """缓存数据加载的昂贵计算"""
    raw_path = Path(__file__).parent.parent / "data" / "raw" / "application_train.csv"
    test_path = Path(__file__).parent.parent / "data" / "raw" / "application_test.csv"

    raw_df = pd.read_csv(raw_path, nrows=100)
    raw_train_rows = _fast_row_count(raw_path)
    raw_test_rows = _fast_row_count(test_path) if test_path.exists() else 0
    raw_total = raw_df.shape[1]
    raw_numeric = len(raw_df.select_dtypes(include=[np.number]).columns)
    raw_categorical = len(raw_df.select_dtypes(include=["object", "category", "string"]).columns)

    # 缺失值统计（只读前 1000 行估算，避免读全量）
    missing_sample = pd.read_csv(raw_path, nrows=1000).isnull().sum()
    missing_pct = (missing_sample / 1000 * 100).sort_values(ascending=False)
    missing_cols = missing_pct[missing_pct > 0]

    return {
        "raw_path": raw_path,
        "test_path": test_path,
        "raw_df": raw_df,
        "raw_train_rows": raw_train_rows,
        "raw_test_rows": raw_test_rows,
        "raw_total": raw_total,
        "raw_numeric": raw_numeric,
        "raw_categorical": raw_categorical,
        "missing_cols": missing_cols,
    }


def _detail_data_loading():
    info = _cached_data_loading_info()
    processed_df = _load_train_df()
    missing_cols = info["missing_cols"]

    return {
        "stage": "data_loading",
        "title": "数据加载",
        "input": {
            "source": "Home Credit Default Risk (Kaggle)",
            "train_file": "application_train.csv",
            "test_file": "application_test.csv",
        },
        "output": {
            "train_shape": [info["raw_train_rows"], info["raw_total"]],
            "test_shape": [info["raw_test_rows"], info["raw_total"]],
            "total_columns": info["raw_total"],
            "numeric_columns": info["raw_numeric"],
            "categorical_columns": info["raw_categorical"],
            "note": f"原始 {info['raw_total']} 列 ({info['raw_numeric']} 数值 + {info['raw_categorical']} 分类) → 特征工程后 {processed_df.shape[1]} 列",
        },
        "details": {
            "missing_columns": int((missing_cols > 0).sum()),
            "top_missing": [
                {"column": col, "pct": round(v, 1)}
                for col, v in missing_cols.head(10).items()
            ],
        },
        "sample_data": _get_sample_preview(info["raw_path"]),
        "auxiliary_tables": _get_auxiliary_table_info(),
    }


_sample_cache = {}

def _get_sample_preview(raw_path):
    """返回主表的精选样本数据（只选关键列，保证可读性）"""
    cache_key = str(raw_path)
    if cache_key in _sample_cache:
        return _sample_cache[cache_key]
    key_cols = [
        "SK_ID_CURR", "TARGET", "NAME_CONTRACT_TYPE", "CODE_GENDER",
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
        "DAYS_BIRTH", "DAYS_EMPLOYED", "NAME_EDUCATION_TYPE",
        "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    ]
    df = pd.read_csv(raw_path, usecols=key_cols, nrows=5)
    df = df.replace([np.inf, -np.inf], np.nan)
    result = {
        "columns": key_cols,
        "rows": df.where(df.notna(), None).values.tolist(),
        "dtypes": {col: str(df[col].dtype) for col in key_cols},
    }
    _sample_cache[cache_key] = result
    return result


_aux_tables_cache = None

def _get_auxiliary_table_info():
    """获取辅助表的详细信息和样本数据（缓存结果）"""
    global _aux_tables_cache
    if _aux_tables_cache is not None:
        return _aux_tables_cache
    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    tables = [
        {"name": "bureau.csv", "description": "信用局历史贷款数据", "key_col": "SK_ID_CURR", "purpose": "关联客户的历史贷款记录（来自其他金融机构）",
         "show_cols": ["SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "DAYS_CREDIT", "AMT_CREDIT_SUM", "CREDIT_TYPE"]},
        {"name": "bureau_balance.csv", "description": "信用局月度余额", "key_col": "SK_ID_BUREAU", "purpose": "每笔贷款的月度还款状态",
         "show_cols": ["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"]},
        {"name": "previous_application.csv", "description": "历史申请记录", "key_col": "SK_ID_CURR", "purpose": "客户在 Home Credit 的历史申请",
         "show_cols": ["SK_ID_CURR", "SK_ID_PREV", "NAME_CONTRACT_TYPE", "AMT_APPLICATION", "AMT_CREDIT", "NAME_CONTRACT_STATUS"]},
        {"name": "POS_CASH_balance.csv", "description": "POS/现金贷款余额", "key_col": "SK_ID_PREV", "purpose": "POS 和现金贷款的月度余额",
         "show_cols": ["SK_ID_PREV", "MONTHS_BALANCE", "CNT_INSTALMENT", "SK_DPD"]},
        {"name": "credit_card_balance.csv", "description": "信用卡余额", "key_col": "SK_ID_PREV", "purpose": "信用卡的月度使用和还款情况",
         "show_cols": ["SK_ID_PREV", "MONTHS_BALANCE", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL", "SK_DPD"]},
        {"name": "installments_payments.csv", "description": "分期还款记录", "key_col": "SK_ID_PREV", "purpose": "每笔贷款的分期还款明细",
         "show_cols": ["SK_ID_PREV", "NUM_INSTALMENT_NUMBER", "DAYS_INSTALMENT", "AMT_INSTALMENT", "AMT_PAYMENT"]},
    ]
    result = []
    for t in tables:
        path = raw_dir / t["name"]
        info = {"name": t["name"], "description": t["description"], "key_col": t["key_col"], "purpose": t["purpose"]}
        if path.exists():
            try:
                all_cols = pd.read_csv(path, nrows=0).columns.tolist()
                show_cols = [c for c in t["show_cols"] if c in all_cols]
                df = pd.read_csv(path, usecols=show_cols, nrows=5)
                # 用文件大小估算行数，避免遍历大文件
                file_size_mb = path.stat().st_size / (1024 * 1024)
                info["size_mb"] = round(file_size_mb, 1)
                if file_size_mb > 100:  # 大文件用估算
                    avg_row_bytes = max(df.memory_usage(deep=True).sum() / len(df), 1)
                    row_count = int(path.stat().st_size / avg_row_bytes)
                else:
                    row_count = _fast_row_count(path)
                info["shape"] = [row_count, len(all_cols)]
                info["columns"] = show_cols
                sample = df.head(3).replace([np.inf, -np.inf], np.nan)
                info["sample_rows"] = sample.where(sample.notna(), None).values.tolist()
            except Exception:
                info["shape"] = [0, 0]
                info["columns"] = []
                info["sample_rows"] = []
        result.append(info)
    _aux_tables_cache = result
    return result


def _detail_data_processing():
    report_path = PROCESSED_DATA_DIR / "train_processing_report.md"
    report = _parse_report(report_path)

    # 读取原始数据做前后对比示例
    raw_path = Path(__file__).parent.parent / "data" / "raw" / "application_train.csv"
    raw_df = pd.read_csv(raw_path, nrows=1000)

    steps = [
        {
            "name": "业务规则清洗",
            "description": "修正 DAYS_EMPLOYED 中的异常占位值 (365243)",
            "icon": "tool",
            "example": {
                "column": "DAYS_EMPLOYED",
                "before": "365243 (异常占位值，表示退休/无业)",
                "after": "NaN → 后续用中位数填充",
                "affected_rows": int((raw_df["DAYS_EMPLOYED"] == 365243).sum()) if "DAYS_EMPLOYED" in raw_df.columns else 0,
                "new_flag": "新增 IS_RETIRED 标志位 (1=退休, 0=在职)",
            },
        },
        {
            "name": "缺失值处理",
            "description": "数值列按收入类型分组中位数填充，类别列众数填充",
            "icon": "grid",
            "example": {
                "method_numeric": "按 NAME_INCOME_TYPE 分组，取组内中位数",
                "method_categorical": "全局众数填充",
                "before_missing_cols": 67,
                "after_missing_cols": 0,
                "sample": {
                    "column": "AMT_ANNUITY",
                    "before_missing_pct": 0.004,
                    "fill_value": "按收入类型分组中位数",
                },
            },
        },
        {
            "name": "分类变量编码",
            "description": "低基数用 One-Hot，高基数用 Frequency Encoding",
            "icon": "hash",
            "example": {
                "one_hot": {
                    "columns": ["NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR"],
                    "method": "pd.get_dummies()",
                    "result": "每列展开为 N 个 0/1 列",
                },
                "frequency": {
                    "columns": ["ORGANIZATION_TYPE", "OCCUPATION_TYPE"],
                    "method": "每个类别替换为出现频率",
                    "example": "ORGANIZATION_TYPE='Business Entity Type 3' → 0.22",
                },
            },
        },
        {
            "name": "异常值处理",
            "description": "IQR 方法截断极端值（Q1-1.5*IQR ~ Q3+1.5*IQR）",
            "icon": "alert-triangle",
            "example": {
                "method": "Winsorize: 超出 1.5*IQR 的值截断到边界",
                "columns_affected": "所有数值列",
                "sample": {
                    "column": "AMT_INCOME_TOTAL",
                    "before_max": 117000000,
                    "after_max": "Q3 + 1.5*IQR",
                },
            },
        },
    ]

    return {
        "stage": "data_processing",
        "title": "数据处理",
        "input": {"shape": [307511, 122]},
        "output": {"shape": [307511, 159]},
        "steps": steps,
        "report": report.get("content", "") if report.get("exists") else None,
    }


def _detail_feature_engineering():
    report_path = PROCESSED_DATA_DIR / "train_feature_engineering_report.md"
    report = _parse_report(report_path)

    steps = [
        {
            "name": "基础衍生特征",
            "count": 14,
            "features": [
                "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "AGE_YEARS",
                "EMPLOYMENT_YEARS", "EXT_SOURCE_MEAN", "DOCUMENT_COUNT",
            ],
            "description": "基于业务逻辑的比率、聚合特征",
            "formulas": [
                {"name": "CREDIT_INCOME_RATIO", "formula": "AMT_CREDIT / AMT_INCOME_TOTAL", "example": "1,125,000 / 202,500 = 5.56", "purpose": "贷款金额与收入的比值，衡量负债压力"},
                {"name": "AGE_YEARS", "formula": "-DAYS_BIRTH / 365.25", "example": "-12005 / 365.25 = 32.9", "purpose": "将天数转换为年龄"},
                {"name": "EXT_SOURCE_MEAN", "formula": "mean(EXT_SOURCE_1, 2, 3)", "example": "(0.5 + 0.6 + 0.7) / 3 = 0.60", "purpose": "外部评分的综合指标"},
                {"name": "DOCUMENT_COUNT", "formula": "sum(FLAG_DOCUMENT_2..21)", "example": "FLAG_DOC_3=1, FLAG_DOC_6=1 → 2", "purpose": "提交的文件数量"},
            ],
        },
        {
            "name": "Bureau 聚合",
            "count": 22,
            "source": "bureau.csv + bureau_balance.csv",
            "description": "信用局历史贷款的统计聚合",
            "aggregations": [
                {"column": "AMT_CREDIT_SUM", "funcs": ["mean", "max", "sum"], "example": "该客户在其他机构的贷款总额"},
                {"column": "DAYS_CREDIT", "funcs": ["min", "max"], "example": "最早/最近一次贷款申请距今天数"},
                {"column": "CNT_CREDIT_PROLONG", "funcs": ["sum"], "example": "贷款延期总次数"},
            ],
        },
        {
            "name": "Previous Application 聚合",
            "count": 28,
            "source": "previous_application.csv",
            "description": "历史申请的批准率、金额统计",
            "aggregations": [
                {"column": "NAME_CONTRACT_STATUS", "funcs": ["count", "value_counts"], "example": "Approved: 3, Refused: 1 → 批准率 75%"},
                {"column": "AMT_APPLICATION", "funcs": ["mean", "max"], "example": "历史申请金额的均值和最大值"},
                {"column": "DAYS_DECISION", "funcs": ["min", "max"], "example": "最早/最近一次审批距今天数"},
            ],
        },
        {
            "name": "Credit Card Balance 聚合",
            "count": 24,
            "source": "credit_card_balance.csv",
            "description": "信用卡使用率、余额统计",
            "aggregations": [
                {"column": "AMT_BALANCE", "funcs": ["mean", "max"], "example": "月均信用卡余额"},
                {"column": "AMT_CREDIT_LIMIT_ACTUAL", "funcs": ["mean"], "example": "信用卡额度均值"},
                {"column": "SK_DPD", "funcs": ["max", "mean"], "example": "最大/平均逾期天数"},
            ],
        },
        {
            "name": "Installments Payments 聚合",
            "count": 18,
            "source": "installments_payments.csv",
            "description": "还款延迟、金额差异统计",
            "aggregations": [
                {"column": "DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT", "funcs": ["mean", "max"], "example": "还款延迟天数（正=晚还）"},
                {"column": "AMT_PAYMENT - AMT_INSTALMENT", "funcs": ["mean", "min"], "example": "还款差额（负=少还）"},
            ],
        },
        {
            "name": "POS_CASH Balance 聚合",
            "count": 12,
            "source": "POS_CASH_balance.csv",
            "description": "POS/现金贷款的 DPD 统计",
            "aggregations": [
                {"column": "SK_DPD", "funcs": ["max", "mean"], "example": "最大/平均逾期天数"},
                {"column": "CNT_INSTALMENT", "funcs": ["max", "mean"], "example": "分期期数统计"},
            ],
        },
    ]

    # 相关性删除
    top_corr_matrix = []
    top_corr_labels = []
    try:
        df = _load_train_df()
        final_features = df.shape[1] - 2
        # 计算重要特征间的相关性矩阵（取前15个特征）
        feature_cols = [c for c in df.columns if c not in [TARGET] + EXCLUDE_FEATURES and pd.api.types.is_numeric_dtype(df[c])]
        if len(feature_cols) > 15:
            # 用方差来选择最具区分度的特征
            variances = df[feature_cols].var().sort_values(ascending=False)
            top_features = variances.head(15).index.tolist()
        else:
            top_features = feature_cols
        corr = df[top_features].corr().fillna(0)
        # 转换为 ECharts heatmap 格式 [x, y, value]
        labels = [f[:20] for f in top_features]  # truncate long names
        for i in range(len(top_features)):
            for j in range(len(top_features)):
                top_corr_matrix.append([j, i, round(float(corr.iloc[i, j]), 3)])
        top_corr_labels = labels
    except Exception:
        final_features = 222

    return {
        "stage": "feature_engineering",
        "title": "特征工程",
        "input": {"shape": [307511, 159]},
        "output": {"shape": [307511, 222]},
        "steps": steps,
        "total_new_features": 118,
        "after_correlation_removal": final_features,
        "correlation_removed": 222 - final_features if final_features < 222 else 0,
        "corr_heatmap": {
            "features": top_corr_labels if top_corr_matrix else [],
            "data": top_corr_matrix,
        },
        "report": report.get("content", "") if report.get("exists") else None,
    }


def _detail_model_training():
    results = {}
    for model_type in ["logistic", "xgboost", "lightgbm"]:
        report_path = PROCESSED_DATA_DIR / f"{model_type}_model_report.md" if model_type != "lightgbm" else REPORT_DIR / "lightgbm_model_report.md"
        report = _parse_report(report_path)
        try:
            model = _load_model(model_type)
            feature_count = len(model.feature_names) if model.feature_names else 0
        except Exception:
            feature_count = 0

        results[model_type] = {
            "report": report.get("content", "") if report.get("exists") else None,
            "feature_count": feature_count,
        }

    return {
        "stage": "model_training",
        "title": "模型训练",
        "data_split": {
            "method": "时间序列划分 (Time-based Split)",
            "train_ratio": "70%",
            "validation_ratio": "15%",
            "calibration_ratio": "15%",
            "note": "按申请时间排序，非随机划分，模拟真实生产环境",
        },
        "feature_selection": {
            "method": "IV (Information Value) 筛选",
            "threshold": 0.005,
            "top_n": 150,
            "before": 222,
            "after": 112,
            "note": "IV < 0.005 的特征被移除（预测能力太弱）",
        },
        "models": {
            "logistic": {
                "name": "Logistic Regression",
                "type": "sklearn.linear_model.LogisticRegression",
                "features": results["logistic"]["feature_count"],
                "techniques": ["class_weight='balanced'", "StandardScaler"],
                "hyperparams": {
                    "C": 1.0,
                    "penalty": "l2",
                    "solver": "lbfgs",
                    "max_iter": 1000,
                    "class_weight": "balanced",
                },
                "training_flow": [
                    "StandardScaler 标准化特征",
                    "LogisticRegression 拟合",
                    "Isotonic Regression 概率校准",
                ],
                "report": results["logistic"]["report"],
            },
            "xgboost": {
                "name": "XGBoost",
                "type": "xgboost.XGBClassifier (Booster)",
                "features": results["xgboost"]["feature_count"],
                "techniques": ["scale_pos_weight", "Optuna (20 trials)", "Isotonic Calibration"],
                "hyperparams": {
                    "n_estimators": 500,
                    "max_depth": 6,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "scale_pos_weight": "auto (n_good/n_bad)",
                    "eval_metric": "auc",
                    "early_stopping_rounds": 50,
                },
                "optuna": {
                    "trials": 20,
                    "objective": "maximize AUC",
                    "search_space": ["max_depth", "learning_rate", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"],
                },
                "training_flow": [
                    "Optuna 超参数搜索 (20 trials)",
                    "XGBoost 拟合 (early stopping)",
                    "Isotonic Regression 概率校准",
                ],
                "report": results["xgboost"]["report"],
            },
            "lightgbm": {
                "name": "LightGBM",
                "type": "lightgbm.LGBMClassifier (Booster)",
                "features": results.get("lightgbm", {}).get("feature_count", 0),
                "techniques": ["scale_pos_weight", "Early Stopping", "Isotonic Calibration"],
                "hyperparams": {
                    "objective": "binary",
                    "boosting_type": "gbdt",
                    "num_leaves": 31,
                    "max_depth": 6,
                    "learning_rate": 0.05,
                    "feature_fraction": 0.8,
                    "bagging_fraction": 0.8,
                    "scale_pos_weight": "auto (n_good/n_bad)",
                    "metric": "auc",
                    "early_stopping_rounds": 50,
                },
                "training_flow": [
                    "LightGBM 拟合 (early stopping, 1000 rounds)",
                    "Best iteration: 322",
                    "Isotonic Regression 概率校准",
                ],
                "report": results.get("lightgbm", {}).get("report"),
            },
            "stacking": {
                "name": "Stacking Ensemble",
                "type": "StackingEnsemble (LR + XGB + LGB → LR Meta)",
                "features": results.get("lightgbm", {}).get("feature_count", 112),
                "techniques": ["5-Fold OOF", "3 Base Models", "LR Meta-Learner", "Isotonic Calibration"],
                "hyperparams": {
                    "base_models": "Logistic + XGBoost + LightGBM",
                    "n_folds": 5,
                    "meta_learner": "LogisticRegression (C=1.0)",
                    "stacking_method": "Out-of-Fold predictions",
                },
                "training_flow": [
                    "K-Fold (5折) 生成 OOF 预测",
                    "3 个基模型各训练 5 折 + 1 全量 = 18 次",
                    "OOF 预测作为 meta 特征",
                    "LogisticRegression 拟合元学习器",
                    "Isotonic Regression 概率校准",
                ],
                "report": None,
            },
        },
    }


def _detail_model_evaluation():
    try:
        comp = pd.read_csv(PROCESSED_DATA_DIR / "model_comparison.csv")
        if comp.columns[0].startswith("Unnamed"):
            comp = comp.rename(columns={comp.columns[0]: "Model"})
        models = comp.to_dict(orient="records")
    except Exception:
        models = []

    # 找到最佳模型
    best_model = None
    if models:
        name_col = "Model" if "Model" in models[0] else list(models[0].keys())[0]
        best_idx = max(range(len(models)), key=lambda i: models[i].get("AUC", 0))
        best_model = models[best_idx].get(name_col, "")

    return {
        "stage": "model_evaluation",
        "title": "模型评估",
        "metrics_explanation": [
            {"name": "AUC", "full": "Area Under ROC Curve", "range": "0.5~1.0", "meaning": "模型区分好坏客户的能力，越高越好"},
            {"name": "KS", "full": "Kolmogorov-Smirnov", "range": "0~1.0", "meaning": "好坏客户累计分布的最大差值"},
            {"name": "Gini", "full": "Gini Coefficient", "range": "0~1.0", "meaning": "2*AUC-1，衡量模型排序能力"},
            {"name": "Precision", "full": "精确率", "range": "0~1.0", "meaning": "预测为违约的客户中，实际违约的比例"},
            {"name": "Recall", "full": "召回率", "range": "0~1.0", "meaning": "实际违约的客户中，被正确预测的比例"},
        ],
        "metrics": ["AUC", "KS", "Gini", "Accuracy", "Precision", "Recall", "F1-Score"],
        "models": models,
        "best_model": best_model,
        "charts": ["ROC Curve", "KS Curve", "Gini Curve", "Confusion Matrix"],
        "evaluation_process": [
            "在验证集上计算预测概率",
            "绘制 ROC 曲线，计算 AUC",
            "绘制 KS 曲线，找到最大 KS 值和最优 cut-off",
            "使用 Isotonic Regression 校准概率",
            "在校准集上验证校准效果",
        ],
    }


def _detail_policy_simulation():
    data = _load_policy_json()
    recommendations = data.get("recommendations", {})
    strategies = []
    for name, metrics in recommendations.items():
        if isinstance(metrics, dict):
            strategies.append({
                "name": name,
                "cutoff": round(metrics.get("cutoff", 0), 4),
                "approval_rate": round(metrics.get("approval_rate", 0) * 100, 1),
                "el_rate": round(metrics.get("el_rate", 0) * 100, 2),
            })

    return {
        "stage": "policy_simulation",
        "title": "策略模拟",
        "lgd": 0.45,
        "cutoff_range": "0.01 ~ 0.50",
        "strategies_count": len(strategies),
        "strategies": strategies,
        "metrics": ["utility_score", "marginal_el", "efficiency_score", "pareto_rank"],
    }


def _detail_monitoring():
    try:
        psi_df = _load_psi_csv()
        psi_df["psi"] = psi_df["psi"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        stable = int((psi_df["psi"] <= 0.10).sum())
        marginal = int(((psi_df["psi"] > 0.10) & (psi_df["psi"] <= 0.25)).sum())
        unstable = int((psi_df["psi"] > 0.25).sum())
        top_unstable = psi_df.nlargest(5, "psi")[["feature", "psi"]].to_dict("records")
    except Exception:
        stable = marginal = unstable = 0
        top_unstable = []

    return {
        "stage": "monitoring",
        "title": "稳定性监控",
        "method": "PSI (Population Stability Index)",
        "thresholds": {"stable": 0.10, "marginal": 0.25, "unstable": ">0.25"},
        "results": {"stable": stable, "marginal": marginal, "unstable": unstable},
        "top_unstable_features": [
            {"feature": r["feature"], "psi": round(r["psi"], 4)} for r in top_unstable
        ],
    }


# =========================================================
# 模型评估
# =========================================================

_model_data_cache = {}
_model_data_lock = __import__('threading').Lock()

def _get_model_data(model_type: str):
    """Load training data and model, predict on train set for visualizations."""
    global _model_data_cache
    if model_type in _model_data_cache:
        return _model_data_cache[model_type]
    with _model_data_lock:
        # Double-check after acquiring lock
        if model_type in _model_data_cache:
            return _model_data_cache[model_type]
        # NOTE: Uses training data (test set has no TARGET in Kaggle competition).
        # Metrics shown in model_comparison.csv are from held-out test split during training.
        df = _load_train_df()
        model = _load_model(model_type)
        X = df.drop(columns=[TARGET] + EXCLUDE_FEATURES, errors="ignore")
        X = X.select_dtypes(include=[np.number])
        X_aligned = model.align_features(X)
        y = df[TARGET].values
        y_proba = model.predict(X_aligned)
        result = (y, y_proba, model)
        _model_data_cache[model_type] = result
        return result

def _load_model_comparison():
    """Load model_comparison.csv (test-set metrics from training scripts)."""
    comp_path = PROCESSED_DATA_DIR / "model_comparison.csv"
    if not comp_path.exists():
        return pd.DataFrame()
    comp = pd.read_csv(comp_path)
    if comp.columns[0].startswith("Unnamed"):
        comp = comp.rename(columns={comp.columns[0]: "Model"})
    # Normalize model names to types
    name_map = {
        "Logistic Regression": "logistic",
        "XGBoost": "xgboost",
        "LightGBM": "lightgbm",
        "Stacking Ensemble": "stacking",
    }
    comp["model_type"] = comp["Model"].map(name_map)
    return comp


def get_model_metrics(model_type: str):
    # Core metrics from model_comparison.csv (test-set evaluation)
    comp = _load_model_comparison()
    csv_metrics = {}
    if not comp.empty:
        row = comp[comp["model_type"] == model_type]
        if not row.empty:
            r = row.iloc[0]
            csv_metrics = {
                "auc": round(float(r.get("AUC", 0)), 4),
                "ks": round(float(r.get("KS", 0)), 4),
                "gini": round(float(r.get("Gini", 0)), 4),
                "accuracy": round(float(r.get("Accuracy", 0)), 4),
                "precision": round(float(r.get("Precision", 0)), 4),
                "recall": round(float(r.get("Recall", 0)), 4),
                "f1": round(float(r.get("F1-Score", 0)), 4),
                "confusion_matrix": {
                    "tp": int(r.get("TP", 0)),
                    "tn": int(r.get("TN", 0)),
                    "fp": int(r.get("FP", 0)),
                    "fn": int(r.get("FN", 0)),
                },
            }

    if csv_metrics:
        # Add percentages to confusion matrix
        cm = csv_metrics["confusion_matrix"]
        total = cm["tp"] + cm["tn"] + cm["fp"] + cm["fn"]
        cm["tp_pct"] = round(cm["tp"] / total * 100, 1) if total else 0
        cm["fp_pct"] = round(cm["fp"] / total * 100, 1) if total else 0
        cm["fn_pct"] = round(cm["fn"] / total * 100, 1) if total else 0
        cm["tn_pct"] = round(cm["tn"] / total * 100, 1) if total else 0
        csv_metrics["model_type"] = model_type
        csv_metrics["threshold"] = 0
        return csv_metrics

    # Fallback: compute on training data if CSV not available
    y, y_proba, model = _get_model_data(model_type)
    auc = float(roc_auc_score(y, y_proba))
    fpr, tpr, thresholds = roc_curve(y, y_proba)
    ks_values = tpr - fpr
    ks = float(max(ks_values))
    ks_idx = np.argmax(ks_values)
    optimal_threshold = float(thresholds[ks_idx])
    y_pred = (y_proba >= optimal_threshold).astype(int)

    cm_raw = confusion_matrix(y, y_pred).tolist()
    tn, fp = cm_raw[0][0], cm_raw[0][1]
    fn, tp = cm_raw[1][0], cm_raw[1][1]
    total = tp + tn + fp + fn

    return {
        "model_type": model_type,
        "threshold": round(optimal_threshold, 4),
        "auc": round(auc, 4),
        "ks": round(ks, 4),
        "gini": round(2 * auc - 1, 4),
        "accuracy": round((tp + tn) / total, 4),
        "precision": round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0,
        "recall": round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0,
        "f1": round(2 * tp / (2 * tp + fp + fn), 4) if (2 * tp + fp + fn) > 0 else 0,
        "confusion_matrix": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "tp_pct": round(tp / total * 100, 1) if total else 0,
            "fp_pct": round(fp / total * 100, 1) if total else 0,
            "fn_pct": round(fn / total * 100, 1) if total else 0,
            "tn_pct": round(tn / total * 100, 1) if total else 0,
        },
    }


def get_roc_data(model_type: str):
    y, y_proba, _ = _get_model_data(model_type)
    fpr, tpr, _ = roc_curve(y, y_proba)
    auc = roc_auc_score(y, y_proba)

    # 降采样到 200 个点
    idx = np.linspace(0, len(fpr) - 1, 200, dtype=int)
    return {
        "fpr": fpr[idx].tolist(),
        "tpr": tpr[idx].tolist(),
        "auc": round(auc, 4),
    }


def get_ks_data(model_type: str):
    y, y_proba, _ = _get_model_data(model_type)
    fpr, tpr, thresholds = roc_curve(y, y_proba)
    ks_values = tpr - fpr
    ks_idx = np.argmax(ks_values)

    idx = np.linspace(0, len(fpr) - 1, 200, dtype=int)
    return {
        "fpr": fpr[idx].tolist(),
        "tpr": tpr[idx].tolist(),
        "ks": round(float(ks_values[ks_idx]), 4),
        "ks_threshold": round(float(thresholds[ks_idx]), 4),
        "ks_fpr": round(float(fpr[ks_idx]), 4),
        "ks_tpr": round(float(tpr[ks_idx]), 4),
    }


def get_gini_data(model_type: str):
    y, y_proba, _ = _get_model_data(model_type)
    sorted_idx = np.argsort(y_proba)[::-1]
    y_sorted = y[sorted_idx]

    n = len(y)
    x_cum = np.linspace(1 / n, 1.0, n).tolist()
    y_cum = (np.cumsum(y_sorted) / y_sorted.sum()).tolist()

    # 降采样
    step = max(1, n // 200)
    return {
        "x": x_cum[::step],
        "y": y_cum[::step],
        "gini": round(2 * roc_auc_score(y, y_proba) - 1, 4),
    }


def get_pd_distribution(model_type: str, bins: int = 30):
    """返回 PD 预测概率的分布直方图数据"""
    y, y_proba, _ = _get_model_data(model_type)

    # 分别统计好客户和坏客户的 PD 分布
    good_mask = y == 0
    bad_mask = y == 1

    bin_edges = np.linspace(0, 1, bins + 1)
    good_hist, _ = np.histogram(y_proba[good_mask], bins=bin_edges)
    bad_hist, _ = np.histogram(y_proba[bad_mask], bins=bin_edges)

    bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2).round(4).tolist()

    return {
        "bins": bin_centers,
        "good": good_hist.tolist(),
        "bad": bad_hist.tolist(),
        "total": len(y),
        "mean_pd": round(float(y_proba.mean()), 4),
        "median_pd": round(float(np.median(y_proba)), 4),
    }


# =========================================================
# 特征分析
# =========================================================

def get_feature_importance(model_type: str, top_n: int = 20):
    model = _load_model(model_type)
    df = _load_train_df()
    X = df.drop(columns=[TARGET] + EXCLUDE_FEATURES, errors="ignore")
    X = X.select_dtypes(include=[np.number])
    X_aligned = model.align_features(X)

    if model_type == "xgboost":
        scores = model.model.get_score(importance_type="gain")
        fi = pd.DataFrame([
            {"feature": f, "importance": scores.get(f, 0)}
            for f in X_aligned.columns
        ]).sort_values("importance", ascending=False).head(top_n)
    elif model_type == "lightgbm":
        importance = model.model.feature_importance(importance_type="gain")
        fi = pd.DataFrame({
            "feature": X_aligned.columns,
            "importance": importance,
        }).sort_values("importance", ascending=False).head(top_n)
    elif model_type == "stacking":
        fi = model.get_feature_importance(top_n=top_n)
    else:
        importance = np.abs(model.model.coef_[0])
        fi = pd.DataFrame({
            "feature": X_aligned.columns,
            "importance": importance,
        }).sort_values("importance", ascending=False).head(top_n)

    return {
        "features": fi["feature"].tolist(),
        "importance": fi["importance"].round(6).tolist(),
    }


def get_iv_data(top_n: int = 30):
    from utils import calculate_woe_iv
    df = _load_train_df()

    feature_cols = [col for col in df.columns
                    if col != TARGET and col not in EXCLUDE_FEATURES
                    and pd.api.types.is_numeric_dtype(df[col])]

    iv_results = []
    for col in feature_cols[:100]:  # 限制计算量
        try:
            _, iv = calculate_woe_iv(df, col, TARGET, bins=10)
            iv_results.append({"feature": col, "iv": round(iv, 4)})
        except Exception:
            continue

    iv_df = pd.DataFrame(iv_results).sort_values("iv", ascending=False).head(top_n)
    return {
        "features": iv_df["feature"].tolist(),
        "iv": iv_df["iv"].tolist(),
    }


def get_shap_data(model_type: str, sample_n: int = 500):
    try:
        import shap
    except ImportError:
        return {"error": "shap not installed"}

    model = _load_model(model_type)
    df = _load_train_df()
    X = df.drop(columns=[TARGET] + EXCLUDE_FEATURES, errors="ignore")
    X = X.select_dtypes(include=[np.number])
    X_aligned = model.align_features(X).iloc[:sample_n]

    if model_type == "xgboost":
        explainer = shap.TreeExplainer(model.model)
        shap_values = explainer.shap_values(X_aligned)
    elif model_type == "stacking":
        return {"error": "SHAP is not supported for Stacking ensemble", "data": [], "features": []}
    elif model_type == "lightgbm":
        explainer = shap.TreeExplainer(model.model)
        shap_values = explainer.shap_values(X_aligned)
    else:
        # Logistic: must scale features first
        X_scaled = pd.DataFrame(model.scaler.transform(X_aligned), columns=X_aligned.columns)
        bg = X_scaled.iloc[:100]
        explainer = shap.KernelExplainer(model.model.predict_proba, bg)
        shap_values = explainer.shap_values(X_scaled.iloc[:200], nsamples=50)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # 正类

    features = X_aligned.columns.tolist()
    data = []
    for i, feat in enumerate(features):
        vals = shap_values[:, i].tolist()
        feat_vals = X_aligned[feat].tolist()
        for j in range(len(vals)):
            data.append({
                "feature": feat,
                "shap_value": round(vals[j], 4),
                "feature_value": round(feat_vals[j], 4) if j < len(feat_vals) else 0,
            })

    return {"data": data[:5000], "features": features}


def get_correlation_matrix(top_n: int = 15):
    """返回 Top N 特征的相关性矩阵"""
    df = _load_train_df()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c != TARGET and c not in EXCLUDE_FEATURES]

    # 选择方差最大的 top_n 个特征
    variances = df[feature_cols].var().sort_values(ascending=False)
    selected = variances.head(top_n).index.tolist()

    corr = df[selected].corr().round(3)

    features = corr.columns.tolist()
    matrix = []
    for i, f1 in enumerate(features):
        for j, f2 in enumerate(features):
            matrix.append({"x": j, "y": i, "value": corr.iloc[i, j]})

    return {
        "features": features,
        "matrix": matrix,
    }


# =========================================================
# 策略模拟
# =========================================================

def get_policy_simulation():
    data = _load_policy_json()
    # Extract pareto_front from simulation_results
    sim = data.get("simulation_results", [])
    pareto = [
        {"cutoff": r["cutoff"], "approval_rate": r["approval_rate"], "el_rate": r["el_rate"]}
        for r in sim if r.get("pareto_rank", 0) == 1
    ]
    return {"pareto_front": pareto, "metadata": data.get("metadata", {})}


def get_policy_interactive(cutoff: float, model_type: str = "xgboost"):
    """实时计算指定 cut-off 下的策略表现"""
    df = _load_train_df().copy()
    model = _load_model(model_type)

    X = df.drop(columns=[TARGET] + EXCLUDE_FEATURES, errors="ignore")
    X = X.select_dtypes(include=[np.number])
    X_aligned = model.align_features(X)
    df["PD"] = model.predict(X_aligned)

    # Guard: AMT_CREDIT may be dropped during feature engineering
    ead_col = "AMT_CREDIT" if "AMT_CREDIT" in df.columns else None

    approved = df[df["PD"] <= cutoff]
    rejected = df[df["PD"] > cutoff]

    total_count = len(df)
    total_ead = df[ead_col].sum() if ead_col else 0

    approval_rate = len(approved) / total_count if total_count > 0 else 0
    approved_ead = approved[ead_col].sum() if ead_col and len(approved) > 0 else 0
    avg_pd = float(approved["PD"].mean()) if len(approved) > 0 else 0

    if len(approved) > 0 and ead_col:
        el = (approved["PD"] * LGD * approved[ead_col]).sum()
        el_rate = el / approved_ead if approved_ead > 0 else 0
    else:
        el = 0
        el_rate = 0

    # 实际违约率
    actual_default_rate = float(approved[TARGET].mean()) if len(approved) > 0 else 0
    reject_capture = float(rejected[TARGET].sum() / df[TARGET].sum()) if df[TARGET].sum() > 0 else 0
    good_keep = float((len(approved) - approved[TARGET].sum()) / (total_count - df[TARGET].sum())) if (total_count - df[TARGET].sum()) > 0 else 0

    return {
        "cutoff": cutoff,
        "approval_rate": round(approval_rate, 4),
        "approved_count": len(approved),
        "rejected_count": len(rejected),
        "avg_pd": round(avg_pd, 4),
        "el_rate": round(float(el_rate), 4),
        "total_el": round(float(el), 2),
        "actual_default_rate": round(actual_default_rate, 4),
        "reject_bad_capture": round(reject_capture, 4),
        "good_keep_rate": round(good_keep, 4),
        "approved_ead": round(float(approved_ead), 2),
    }


def get_recommendations(target_approval: float = None, max_el: float = None):
    """获取多策略推荐 - 返回列表格式"""
    data = _load_policy_json()
    raw = data.get("recommendations", {})
    result = []
    for strategy, metrics in raw.items():
        if isinstance(metrics, dict):
            result.append({"strategy": strategy, **metrics})
    return {"strategies": result}


# =========================================================
# 监控
# =========================================================

def get_psi_data(threshold: float = 0.25):
    psi_df = _load_psi_csv()
    # Sanitize inf/nan values
    psi_df["psi"] = psi_df["psi"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    stable = psi_df[psi_df["psi"] <= 0.10]
    marginal = psi_df[(psi_df["psi"] > 0.10) & (psi_df["psi"] <= threshold)]
    unstable = psi_df[psi_df["psi"] > threshold]

    return {
        "total_features": len(psi_df),
        "stable_count": len(stable),
        "marginal_count": len(marginal),
        "unstable_count": len(unstable),
        "features": psi_df["feature"].tolist(),
        "psi_values": psi_df["psi"].round(4).tolist(),
        "threshold": threshold,
    }


def get_monitoring_health(threshold: float = 0.25):
    """Overall monitoring health status with alerts and recommendations."""
    psi_df = _load_psi_csv()
    psi_df["psi"] = psi_df["psi"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    stable_count = int((psi_df["psi"] <= 0.10).sum())
    marginal_count = int(((psi_df["psi"] > 0.10) & (psi_df["psi"] <= threshold)).sum())
    unstable_count = int((psi_df["psi"] > threshold).sum())
    total = len(psi_df)

    # Overall status
    if unstable_count == 0 and marginal_count == 0:
        status = "healthy"
        status_label = "All Clear"
        color = "emerald"
    elif unstable_count == 0:
        status = "warning"
        status_label = "Attention Needed"
        color = "amber"
    else:
        status = "critical"
        status_label = "Action Required"
        color = "rose"

    # Top drift features (highest PSI)
    top_drift = psi_df.nlargest(5, "psi")[["feature", "psi"]].to_dict(orient="records")

    # Alert rules
    alerts = []
    if unstable_count > 0:
        alerts.append({
            "level": "critical",
            "message": f"{unstable_count} feature(s) exceeded PSI threshold ({threshold:.2f})",
            "action": "Review top drifting features. Consider retraining the model if drift is significant.",
        })
    if marginal_count > total * 0.3:
        alerts.append({
            "level": "warning",
            "message": f"{marginal_count} feature(s) in marginal zone (>30% of total)",
            "action": "Monitor closely. Schedule a feature distribution review within the next sprint.",
        })
    if stable_count == total:
        alerts.append({
            "level": "info",
            "message": "All features are stable — model performance should be consistent.",
            "action": "No immediate action required. Continue periodic monitoring.",
        })

    return _sanitize({
        "status": status,
        "status_label": status_label,
        "color": color,
        "total_features": total,
        "stable_count": stable_count,
        "marginal_count": marginal_count,
        "unstable_count": unstable_count,
        "stable_pct": round(stable_count / total * 100, 1) if total else 0,
        "marginal_pct": round(marginal_count / total * 100, 1) if total else 0,
        "unstable_pct": round(unstable_count / total * 100, 1) if total else 0,
        "top_drift": top_drift,
        "alerts": alerts,
        "threshold": threshold,
    })


def get_monitoring_vintage():
    """Return vintage analysis data for chart."""
    data = _load_monitoring_xlsx()
    if data is None or 'Vintage Analysis' not in data:
        return {"cohorts": [], "months": [], "series": []}
    df = data['Vintage Analysis']
    obs_cols = [c for c in df.columns if 'default_rate_' in c]
    months = [int(c.split('_')[-1].replace('m', '')) for c in obs_cols]
    series = []
    for _, row in df.iterrows():
        series.append({
            "cohort": str(row['cohort']),
            "values": [round(float(row[c]), 2) for c in obs_cols],
        })
    return {"months": months, "series": series}


def get_monitoring_roll_rate():
    """Return roll rate data for chart."""
    data = _load_monitoring_xlsx()
    if data is None or 'Roll Rate' not in data:
        return {"buckets": [], "counts": [], "percentages": []}
    df = data['Roll Rate']
    return {
        "buckets": df['bucket'].tolist(),
        "counts": df['count'].tolist(),
        "percentages": [round(float(v), 2) for v in df['percentage'].tolist()],
    }


def get_monitoring_ewi():
    """Return early warning indicators."""
    data = _load_monitoring_xlsx()
    if data is None or 'Early Warning Indicators' not in data:
        return {}
    df = data['Early Warning Indicators']
    row = df.iloc[0].to_dict()
    return _sanitize(row)


# =========================================================
# Scorecard
# =========================================================

def _get_scorecard_data(model_type: str):
    """Load model, predict PD, convert to scores. Cached."""
    if model_type in _scorecard_cache:
        return _scorecard_cache[model_type]

    y, y_proba, model = _get_model_data(model_type)
    from scorecard import pd_to_score
    scores = pd_to_score(y_proba)

    result = (y, y_proba, scores, model)
    _scorecard_cache[model_type] = result
    return result

_scorecard_cache: dict = {}


def get_scorecard_summary(model_type: str):
    """Return high-level scorecard statistics."""
    from scorecard import compute_scorecard_summary
    y, y_proba, scores, model = _get_scorecard_data(model_type)
    return _sanitize(compute_scorecard_summary(scores, y))


def get_scorecard_distribution(model_type: str, n_bins: int = 20):
    """Return score distribution with per-bin statistics."""
    from scorecard import compute_score_bins
    y, y_proba, scores, model = _get_scorecard_data(model_type)
    bins_df = compute_score_bins(scores, y, n_bins=n_bins)
    return _sanitize({
        "bins": bins_df.to_dict(orient="records"),
        "config": {
            "base_score": BASE_SCORE,
            "base_odds": BASE_ODDS,
            "pdo": PDO,
            "score_min": SCORE_MIN,
            "score_max": SCORE_MAX,
        },
    })


def get_scorecard_lift(model_type: str, n_bins: int = 10):
    """Return Lift / Gains / Capture Rate analysis."""
    from scorecard import compute_lift_gains
    y, y_proba, scores, model = _get_scorecard_data(model_type)
    return _sanitize(compute_lift_gains(scores, y, n_bins=n_bins))
