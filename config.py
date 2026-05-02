"""
配置文件 - 信用风险项目
"""
import os
from pathlib import Path

# ==================== 路径配置 ====================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = OUTPUT_DIR / "reports"
VIZ_DIR = OUTPUT_DIR / "visualizations"

# 创建必要的目录
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, REPORT_DIR, VIZ_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ==================== 数据配置 ====================
# Kaggle 竞赛名称
COMPETITION_NAME = "home-credit-default-risk"

# 主要数据文件
DATA_FILES = {
    "application_train": "application_train.csv",
    "application_test": "application_test.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "credit_card_balance": "credit_card_balance.csv",
    "installments_payments": "installments_payments.csv",
    "POS_CASH_balance": "POS_CASH_balance.csv",
    "previous_application": "previous_application.csv",
}

# ==================== 模型配置 ====================
# 目标变量
TARGET = "TARGET"

# 随机种子
RANDOM_STATE = 42

# 数据划分比例
TRAIN_SIZE = 0.7
VALID_SIZE = 0.15
TEST_SIZE = 0.15

# 模型参数
LOGISTIC_PARAMS = {
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
    "class_weight": "balanced",
    "solver": "lbfgs",
}

XGBOOST_PARAMS = {
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "random_state": RANDOM_STATE,
    "scale_pos_weight": 10,  # 处理不平衡数据
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

# ==================== 业务配置 ====================
# 违约定义
DEFAULT_DEFINITION = "TARGET=1 (Home Credit 综合违约定义)"

# LGD 假设（违约损失率）
LGD = 0.45  # 45%

# 评分卡参数
BASE_SCORE = 600       # 基准分数（对应 BASE_ODDS 时的分数）
BASE_ODDS = 50         # 基准好坏比 (Good:Bad = 50:1)
PDO = 20               # 每增加 PDO 分，好坏比翻倍
SCORE_MIN = 300        # 评分卡最低分
SCORE_MAX = 850        # 评分卡最高分

# IFRS9 分层阈值（示例）
IFRS9_THRESHOLDS = {
    "stage_1_to_2": 0.05,  # PD 增长 5% 触发 Stage 2
    "stage_2_to_3": 0.90,  # 90+ DPD 触发 Stage 3
}

# 策略模拟 - Cut-off 范围
CUTOFF_RANGE = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
                0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

# ==================== 可视化配置 ====================
# 图表风格
PLOT_STYLE = "seaborn-v0_8-darkgrid"
FIGURE_SIZE = (12, 6)
FIGURE_DPI = 100

# 颜色方案
COLOR_PALETTE = {
    "primary": "#2E86AB",      # 蓝色
    "secondary": "#A23B72",    # 紫色
    "success": "#06A77D",      # 绿色
    "warning": "#F18F01",      # 橙色
    "danger": "#C73E1D",       # 红色
    "default": "#6C757D",      # 灰色
}

# ==================== 特征工程配置 ====================
# 需要排除的特征
EXCLUDE_FEATURES = [
    "SK_ID_CURR",  # ID 列
    "SK_ID_BUREAU",
    "SK_ID_PREV",
    "TARGET",  # 目标变量
]

# 缺失值处理阈值
MISSING_THRESHOLD = 0.8  # 缺失率超过 80% 的特征将被删除

# 相关性阈值
CORRELATION_THRESHOLD = 0.95  # 相关性超过 95% 的特征将被删除
REMOVE_CORRELATED = True  # 是否执行高相关特征删除

# 编码策略
USE_TARGET_ENCODING = False  # 是否使用 Target Encoding 替代 Frequency Encoding

# WOE/IV 特征筛选
USE_IV_SELECTION = True   # 是否启用 IV 特征筛选
IV_THRESHOLD = 0.005      # IV 最低阈值（0.02=有用, 0.1=强, 0.5=非常强）
IV_TOP_N = 150            # 最多保留特征数

# SHAP 特征筛选（替代 IV，更精准但更慢）
USE_SHAP_SELECTION = False  # 是否启用 SHAP 特征筛选
SHAP_TOP_N = 100            # SHAP 筛选最多保留特征数

# ==================== 组合监控配置 ====================
# Vintage 分析 - 观察期（月）
VINTAGE_OBSERVATION_MONTHS = [3, 6, 9, 12, 18, 24]

# Roll Rate 分析 - DPD 分桶
DPD_BUCKETS = {
    "Current": (0, 0),
    "1-30": (1, 30),
    "31-60": (31, 60),
    "61-90": (61, 90),
    "91-120": (91, 120),
    "121-150": (121, 150),
    "151-180": (151, 180),
    "180+": (181, 999),
}

# Early Warning Indicators
EWI_METRICS = [
    "approval_rate",           # 批准率
    "average_loan_amount",     # 平均贷款金额
    "average_income",          # 平均收入
    "debt_to_income_ratio",    # 负债收入比
    "fpd_rate",                # First Payment Default 率
]

# ==================== 报告配置 ====================
# 模型报告包含的指标
MODEL_METRICS = [
    "AUC",
    "KS",
    "Gini",
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
]

# ==================== 日志配置 ====================
import logging

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"

def setup_logging():
    """配置全局 logging：控制台 + 文件"""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(OUTPUT_DIR / "pipeline.log", encoding="utf-8"),
        ]
    )
