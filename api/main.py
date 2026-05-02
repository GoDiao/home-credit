"""
FastAPI 后端 - Home Credit Dashboard
"""
import sys
from pathlib import Path
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import OUTPUT_DIR, VIZ_DIR
from api.services import (
    get_overview, get_model_metrics, get_roc_data, get_ks_data,
    get_gini_data, get_feature_importance, get_iv_data, get_shap_data,
    get_policy_simulation, get_policy_interactive, get_recommendations,
    get_psi_data, get_pipeline_detail, get_pd_distribution, get_correlation_matrix,
    get_model_registry,
    get_scorecard_summary, get_scorecard_distribution, get_scorecard_lift,
    get_monitoring_health, get_monitoring_vintage, get_monitoring_roll_rate,
    get_monitoring_ewi,
    get_overview_briefing,
)

app = FastAPI(title="Home Credit Dashboard API", version="1.0.0")


@app.on_event("startup")
async def warmup_cache():
    """启动时预热缓存，避免首次请求卡顿"""
    import threading
    def _warmup():
        try:
            get_pipeline_detail("data_loading")
        except Exception:
            pass
        try:
            # Pre-warm model data cache for all models
            from api.services import _get_model_data
            for mt in ["stacking", "xgboost", "lightgbm", "logistic"]:
                try:
                    _get_model_data(mt)
                except Exception:
                    pass
        except Exception:
            pass
    threading.Thread(target=_warmup, daemon=True).start()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件：可视化图片
app.mount("/images", StaticFiles(directory=str(VIZ_DIR)), name="images")


# =========================================================
# 概览
# =========================================================
@app.get("/api/overview")
def overview():
    return get_overview()


@app.get("/api/overview/briefing")
def overview_briefing():
    return get_overview_briefing()


@app.get("/api/pipeline/{stage}")
def pipeline_detail(stage: str):
    return get_pipeline_detail(stage)


# =========================================================
# 模型评估
# =========================================================
@app.get("/api/model/{model_type}/metrics")
def model_metrics(model_type: str):
    return get_model_metrics(model_type)


@app.get("/api/models/registry")
def models_registry():
    return get_model_registry()


@app.get("/api/model/{model_type}/roc")
def model_roc(model_type: str):
    return get_roc_data(model_type)


@app.get("/api/model/{model_type}/ks")
def model_ks(model_type: str):
    return get_ks_data(model_type)


@app.get("/api/model/{model_type}/gini")
def model_gini(model_type: str):
    return get_gini_data(model_type)


@app.get("/api/model/{model_type}/pd_distribution")
def pd_distribution(model_type: str, bins: int = Query(30, ge=10, le=50)):
    return get_pd_distribution(model_type, bins)


# =========================================================
# 特征分析
# =========================================================
@app.get("/api/features/importance")
def feature_importance(
    model_type: str = Query("xgboost"),
    top_n: int = Query(20, ge=5, le=50),
):
    return get_feature_importance(model_type, top_n)


@app.get("/api/features/iv")
def feature_iv(top_n: int = Query(30, ge=5, le=50)):
    return get_iv_data(top_n)


@app.get("/api/features/shap")
def feature_shap(
    model_type: str = Query("xgboost"),
    sample_n: int = Query(500, ge=100, le=2000),
):
    return get_shap_data(model_type, sample_n)


@app.get("/api/features/correlation")
def feature_correlation(top_n: int = Query(15, ge=5, le=30)):
    return get_correlation_matrix(top_n)


# =========================================================
# 策略模拟
# =========================================================
@app.get("/api/policy/simulation")
def policy_simulation():
    return get_policy_simulation()


@app.get("/api/policy/interactive")
def policy_interactive(
    cutoff: float = Query(..., ge=0.001, le=0.99),
    model_type: str = Query("xgboost"),
):
    """动态 cut-off 查询"""
    return get_policy_interactive(cutoff, model_type)


@app.get("/api/policy/recommend")
def policy_recommend(
    target_approval: float = Query(None, ge=0, le=1),
    max_el: float = Query(None, ge=0, le=1),
):
    return get_recommendations(target_approval, max_el)


# =========================================================
# 监控
# =========================================================
@app.get("/api/monitoring/psi")
def psi_monitoring(threshold: float = Query(0.25, ge=0.05, le=1.0)):
    return get_psi_data(threshold)


@app.get("/api/monitoring/health")
def monitoring_health(threshold: float = Query(0.25, ge=0.05, le=1.0)):
    return get_monitoring_health(threshold)


@app.get("/api/monitoring/vintage")
def monitoring_vintage():
    return get_monitoring_vintage()


@app.get("/api/monitoring/roll_rate")
def monitoring_roll_rate():
    return get_monitoring_roll_rate()


@app.get("/api/monitoring/ewi")
def monitoring_ewi():
    return get_monitoring_ewi()


# =========================================================
# 评分卡
# =========================================================
@app.get("/api/scorecard/summary")
def scorecard_summary(model_type: str = Query("xgboost")):
    return get_scorecard_summary(model_type)


@app.get("/api/scorecard/distribution")
def scorecard_distribution(
    model_type: str = Query("xgboost"),
    n_bins: int = Query(20, ge=5, le=50),
):
    return get_scorecard_distribution(model_type, n_bins)


@app.get("/api/scorecard/lift")
def scorecard_lift(
    model_type: str = Query("xgboost"),
    n_bins: int = Query(10, ge=5, le=20),
):
    return get_scorecard_lift(model_type, n_bins)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
