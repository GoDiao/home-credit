# Home Credit 风控建模项目

基于 Kaggle **Home Credit Default Risk** 数据集的信用风险量化分析项目，覆盖数据处理、特征工程、PD 模型训练、策略模拟、组合监控，以及 FastAPI + Vue Dashboard 可视化展示。

> **想理解这个项目做了什么？** 看 [GUIDE.md](GUIDE.md) — 从业务背景到每一步的技术细节，面向零基础读者。

---

## 一键运行

### 前置条件

| 依赖 | 版本 | 安装 |
|------|------|------|
| Python | >= 3.11 | [python.org](https://www.python.org/downloads/) |
| Node.js | >= 18 | [nodejs.org](https://nodejs.org/) |
| uv | 最新 | `pip install uv` 或 [uv 官网](https://docs.astral.sh/uv/) |

### 运行

```bash
# 克隆仓库
git clone <repo-url>
cd home_credit

# 一键运行（安装依赖 → 下载数据 → 跑 Pipeline → 启动 Dashboard）
python run.py
```

脚本会自动完成所有步骤，最后打开浏览器显示 Dashboard。

如果数据和依赖已经装好，只想启动 Dashboard：

```bash
python run.py --skip
```

> **数据来源**：需要从 Kaggle 下载 [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data) 数据集，脚本会引导你完成。也可以安装 `kaggle` CLI 自动下载。

---

## 项目概览

- **数据集**：Home Credit Default Risk（约 307K 训练样本，主表 122 个基础字段）
- **目标变量**：`TARGET`，1 表示违约，0 表示正常
- **模型体系**：Logistic Regression、XGBoost、LightGBM、Stacking Ensemble
- **监控分析**：PSI、Vintage Analysis、Roll Rate、策略 Cut-off 模拟
- **服务与展示**：FastAPI 后端 + Vue 3 / Vite / ECharts Dashboard

## 目录结构

```text
home_credit/
├── run.py                          # 一键运行脚本（安装依赖 → 下载数据 → 跑 Pipeline → 启动 Dashboard）
├── config.py                       # 全局配置、路径、模型参数
├── main.py                         # Pipeline CLI 总入口
├── data_processing.py              # 数据清洗、缺失值、编码、异常值处理
├── feature_engineering.py          # 基础衍生特征与辅助表聚合
├── pd_model.py                     # PD 模型训练、评估、保存
├── train_lightgbm.py               # LightGBM 单独训练脚本
├── train_stacking.py               # Stacking 单独训练脚本
├── policy_simulation.py            # Cut-off 策略模拟与预期损失分析
├── portfolio_monitoring.py         # Vintage / Roll Rate / EWI 监控
├── scorecard.py                    # 评分卡：PD ↔ 分数转换
├── model_registry.py               # 模型元数据管理
├── utils.py                        # 指标、绘图、WOE/IV、PSI 工具函数
├── api/                            # FastAPI 后端
│   ├── main.py                     # API 路由
│   └── services.py                 # API 数据服务层
├── dashboard/                      # Vue 3 + Vite 前端 Dashboard
├── data/                           # 数据（运行 run.py 后自动生成，不入 git）
│   ├── raw/                        # Kaggle 原始数据（10 个 CSV，约 2.5 GB）
│   └── processed/                  # 处理后数据、模型对比 CSV、预测结果
├── outputs/                        # 产物（运行 pipeline 后自动生成，不入 git）
│   ├── models/                     # 训练后的模型文件（.pkl）
│   ├── reports/                    # 分析报告、策略结果、监控结果
│   └── visualizations/             # ROC/KS/特征重要性等图表
├── tests/                          # pytest 单元测试
├── GUIDE.md                        # 学习指南（业务背景 + 技术详解）
├── pyproject.toml                  # Python 依赖配置（uv 管理）
└── uv.lock                         # uv 锁文件
```

## Pipeline 命令

根目录 `main.py` 提供统一 CLI 入口：

```bash
# 完整流水线
uv run python main.py --stage all

# 单独阶段
uv run python main.py --stage process      # 数据处理
uv run python main.py --stage features     # 特征工程
uv run python main.py --stage train        # 模型训练（Logistic + XGBoost）
uv run python main.py --stage lightgbm     # LightGBM 单独训练
uv run python main.py --stage stacking     # Stacking 集成训练
uv run python main.py --stage policy       # 策略模拟
uv run python main.py --stage monitoring   # 组合监控

# 查看帮助
uv run python main.py --help
```

> 完整训练和 Stacking 可能耗时较长；如果只想查看 Dashboard，直接用仓库已有的产物即可。

## 运行测试

```bash
uv run python -m pytest -q
```

## Pipeline 阶段说明

| Stage | 命令 | 输入 | 输出 |
|---|---|---|---|
| 数据处理 | `--stage process` | `data/raw/application_train.csv` 等 | `data/processed/*_processed.csv` |
| 特征工程 | `--stage features` | processed 数据 + 辅助表 | `data/processed/*_with_features.csv` |
| 模型训练 | `--stage train` | `train_with_features.csv` | 模型文件、模型报告、预测结果 |
| LightGBM | `--stage lightgbm` | `train_with_features.csv` | `pd_model_lightgbm.pkl` 等 |
| Stacking | `--stage stacking` | 已训练 LightGBM + 特征数据 | `pd_model_stacking.pkl` |
| 策略 | `--stage policy` | 模型预测 / 特征数据 | 策略报告、Cut-off 模拟结果 |
| 组合监控 | `--stage monitoring` | 处理后数据 / 预测结果 | Vintage、Roll Rate、PSI 等报告 |

## 核心指标与公式

### 预期损失（Expected Loss）

```text
EL = PD × LGD × EAD
```

- **PD**：Probability of Default，违约概率
- **LGD**：Loss Given Default，违约损失率，本项目默认 45%
- **EAD**：Exposure at Default，违约风险敞口

### 模型评估指标

- **AUC**：模型区分能力（0.5=随机，1=完美）
- **KS**：最大区分度
- **Gini**：`2 × AUC - 1`
- **Precision / Recall / F1**：分类效果
- **PSI**：特征稳定性 / 分布漂移监控

## 主要产物

运行 Pipeline 后自动生成（不入 git）：

```text
data/processed/model_comparison.csv          # 多模型指标对比
data/processed/train_with_features.csv       # 训练集特征工程结果
outputs/models/*.pkl                         # 模型文件
outputs/reports/*.md / *.json / *.csv        # 模型、策略、监控报告
outputs/visualizations/*.png                 # ROC、KS、特征重要性等图表
```

## Dashboard 页面

- **Overview**：项目总览、Pipeline 状态、模型对比、Executive Summary
- **Model Evaluation**：AUC / KS / Gini、ROC、KS、PD 分布、混淆矩阵
- **Feature Analysis**：特征重要性、IV、SHAP、相关性热力图
- **Scorecard**：评分卡分布、Lift 分析、分箱详情
- **Policy Simulation**：Cut-off 策略、通过率、预期损失、推荐策略
- **Monitoring**：PSI 稳定性监控、Vintage / Roll Rate、早期预警

## 参考资料

- [Home Credit Default Risk - Kaggle](https://www.kaggle.com/c/home-credit-default-risk)
- [IFRS 9 - Impairment](https://www.ifrs.org/issued-standards/list-of-standards/ifrs-9-financial-instruments/)
- [CECL - Current Expected Credit Loss](https://www.fasb.org/cecl)
