"""
策略模拟模块 - Cut-off 优化与预期损失分析
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import warnings
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import *


class PolicySimulator:
    """策略模拟类"""

    def __init__(self, lgd: float = LGD,
                 utility_weight_approval: float = 1.0,
                 utility_weight_el: float = 3.0):
        """
        初始化策略模拟器

        Parameters:
        -----------
        lgd : float
            违约损失率 (Loss Given Default)
        utility_weight_approval : float
            通过率权重（越大越重视业务规模）
        utility_weight_el : float
            EL 率权重（越大越重视风险控制）
        """
        self.lgd = lgd
        self.utility_weight_approval = utility_weight_approval
        self.utility_weight_el = utility_weight_el
        self.simulation_results: Optional[pd.DataFrame] = None
        self.scenario_results: Optional[pd.DataFrame] = None
        self.recommendations: Optional[Dict] = None
        self.chart_paths: Dict[str, str] = {}
        self.meta_info: Dict = {}

        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (12, 6)
        plt.rcParams["axes.unicode_minus"] = False

    # =========================================================
    # 1. Utility Functions
    # =========================================================
    def calculate_expected_loss(
        self,
        pd_values: Union[float, pd.Series, np.ndarray],
        lgd: float,
        ead_values: Union[float, pd.Series, np.ndarray]
    ) -> Union[float, pd.Series, np.ndarray]:
        """
        计算预期损失

        EL = PD × LGD × EAD
        """
        return pd_values * lgd * ead_values

    def _validate_input_data(self, df: pd.DataFrame, pd_col: str, ead_col: str) -> pd.DataFrame:
        """
        校验并清洗策略模拟输入数据
        """
        if df is None or df.empty:
            raise ValueError("输入数据为空，无法进行策略模拟。")

        if pd_col not in df.columns:
            raise ValueError(f"未找到 PD 列: {pd_col}")

        if ead_col not in df.columns:
            raise ValueError(f"未找到 EAD 列: {ead_col}")

        sim_df = df.copy()

        sim_df[pd_col] = pd.to_numeric(sim_df[pd_col], errors="coerce")
        sim_df[ead_col] = pd.to_numeric(sim_df[ead_col], errors="coerce")

        before_rows = len(sim_df)
        sim_df = sim_df.dropna(subset=[pd_col, ead_col]).copy()

        sim_df[pd_col] = sim_df[pd_col].clip(lower=0, upper=1)
        sim_df = sim_df[sim_df[ead_col] >= 0].copy()

        after_rows = len(sim_df)

        print("\n📋 输入数据检查完成")
        print(f"原始样本数: {before_rows:,}")
        print(f"清洗后样本数: {after_rows:,}")
        print(f"PD 最小值: {sim_df[pd_col].min():.6f}")
        print(f"PD 最大值: {sim_df[pd_col].max():.6f}")
        print(f"EAD 总额: {sim_df[ead_col].sum():,.2f}")

        return sim_df

    def _evaluate_single_cutoff(
        self,
        df: pd.DataFrame,
        pd_col: str,
        cutoff: float,
        ead_col: str,
        target_col: Optional[str] = None
    ) -> Dict:
        """
        评估单个 cut-off 下的策略表现
        """
        approved = df[df[pd_col] <= cutoff].copy()
        rejected = df[df[pd_col] > cutoff].copy()

        total_count = len(df)
        total_ead_all = df[ead_col].sum()

        approval_rate = len(approved) / total_count if total_count > 0 else 0
        reject_rate = len(rejected) / total_count if total_count > 0 else 0

        approved_count = len(approved)
        rejected_count = len(rejected)

        approved_ead = approved[ead_col].sum() if approved_count > 0 else 0
        rejected_ead = rejected[ead_col].sum() if rejected_count > 0 else 0

        approval_ead_share = approved_ead / total_ead_all if total_ead_all > 0 else 0
        reject_ead_share = rejected_ead / total_ead_all if total_ead_all > 0 else 0

        avg_pd = approved[pd_col].mean() if approved_count > 0 else 0
        max_pd_approved = approved[pd_col].max() if approved_count > 0 else 0
        min_pd_approved = approved[pd_col].min() if approved_count > 0 else 0

        if approved_count > 0:
            approved["EL"] = self.calculate_expected_loss(
                approved[pd_col],
                self.lgd,
                approved[ead_col]
            )
            total_el = approved["EL"].sum()
            avg_el_per_loan = approved["EL"].mean()
            el_rate = total_el / approved_ead if approved_ead > 0 else 0
        else:
            total_el = 0
            avg_el_per_loan = 0
            el_rate = 0

        result = {
            "cutoff": float(cutoff),
            "approval_rate": float(approval_rate),
            "reject_rate": float(reject_rate),
            "approved_count": int(approved_count),
            "rejected_count": int(rejected_count),
            "approved_ead": float(approved_ead),
            "rejected_ead": float(rejected_ead),
            "approval_ead_share": float(approval_ead_share),
            "reject_ead_share": float(reject_ead_share),
            "avg_pd": float(avg_pd),
            "min_pd_approved": float(min_pd_approved),
            "max_pd_approved": float(max_pd_approved),
            "total_el": float(total_el),
            "avg_el_per_loan": float(avg_el_per_loan),
            "el_rate": float(el_rate),
        }

        if target_col is not None and target_col in df.columns:
            approved_bad = approved[target_col].sum() if approved_count > 0 else 0
            rejected_bad = rejected[target_col].sum() if rejected_count > 0 else 0
            total_bad = df[target_col].sum()

            approved_bad_rate = approved[target_col].mean() if approved_count > 0 else 0
            rejected_bad_rate = rejected[target_col].mean() if rejected_count > 0 else 0

            reject_bad_capture_rate = rejected_bad / total_bad if total_bad > 0 else 0
            approve_good_keep_rate = (
                (approved_count - approved_bad) / (total_count - total_bad)
                if (total_count - total_bad) > 0 else 0
            )

            result.update({
                "actual_defaults_approved": int(approved_bad),
                "actual_defaults_rejected": int(rejected_bad),
                "actual_default_rate_approved": float(approved_bad_rate),
                "actual_default_rate_rejected": float(rejected_bad_rate),
                "total_bad_accounts": int(total_bad),
                "reject_bad_capture_rate": float(reject_bad_capture_rate),
                "approve_good_keep_rate": float(approve_good_keep_rate),
            })
        else:
            result.update({
                "actual_defaults_approved": None,
                "actual_defaults_rejected": None,
                "actual_default_rate_approved": None,
                "actual_default_rate_rejected": None,
                "total_bad_accounts": None,
                "reject_bad_capture_rate": None,
                "approve_good_keep_rate": None,
            })

        return result

    def _format_pct(self, value) -> str:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{value:.2%}"

    def _format_num(self, value, digits: int = 2) -> str:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{value:,.{digits}f}"

    def _clean_value(self, x):
        if pd.isna(x):
            return None
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        return x

    def _df_to_records(self, df: Optional[pd.DataFrame]):
        if df is None or df.empty:
            return []
        records = df.to_dict(orient="records")
        cleaned = []
        for row in records:
            cleaned.append({k: self._clean_value(v) for k, v in row.items()})
        return cleaned

    # =========================================================
    # 2. Cut-off Simulation
    # =========================================================
    def simulate_cutoff_policy(
        self,
        df: pd.DataFrame,
        pd_col: str,
        ead_col: str = "AMT_CREDIT",
        cutoff_range: Optional[List[float]] = None,
        target_col: Optional[str] = TARGET
    ) -> pd.DataFrame:
        """
        模拟不同 Cut-off 阈值下的策略表现
        """
        print("\n模拟 Cut-off 策略...")

        if cutoff_range is None:
            cutoff_range = CUTOFF_RANGE

        sim_df = self._validate_input_data(df, pd_col=pd_col, ead_col=ead_col)

        cutoff_range = sorted(set([float(x) for x in cutoff_range if x is not None]))
        if len(cutoff_range) == 0:
            raise ValueError("cutoff_range 为空，无法进行模拟。")

        results = []
        for cutoff in cutoff_range:
            result = self._evaluate_single_cutoff(
                df=sim_df,
                pd_col=pd_col,
                cutoff=cutoff,
                ead_col=ead_col,
                target_col=target_col
            )
            results.append(result)

        result_df = pd.DataFrame(results).sort_values("cutoff").reset_index(drop=True)

        # === 多维度评分 ===
        # 1. Utility Score: w1 * approval_rate - w2 * el_rate (可配置权重)
        result_df["utility_score"] = (
            self.utility_weight_approval * result_df["approval_rate"] -
            self.utility_weight_el * result_df["el_rate"]
        )

        # 2. Marginal EL: 每增加 1% 通过率带来的 EL 增量
        result_df["marginal_el"] = result_df["el_rate"].diff() / (result_df["approval_rate"].diff() + 1e-12)

        # 3. Efficiency Score: 通过率 / (EL率 + epsilon)，越高越好
        result_df["efficiency_score"] = result_df["approval_rate"] / (result_df["el_rate"] + 1e-6)

        # 4. Pareto Rank: 非支配排序
        result_df["pareto_rank"] = self._compute_pareto_rank(result_df)

        # 保留旧版 balance_score 兼容（但标记为 deprecated）
        approval_norm = (
            (result_df["approval_rate"] - result_df["approval_rate"].min()) /
            (result_df["approval_rate"].max() - result_df["approval_rate"].min() + 1e-12)
        )
        el_norm = (
            (result_df["el_rate"] - result_df["el_rate"].min()) /
            (result_df["el_rate"].max() - result_df["el_rate"].min() + 1e-12)
        )
        result_df["balance_score"] = approval_norm - el_norm

        self.simulation_results = result_df

        print(f"模拟完成: {len(cutoff_range)} 个 cut-off 阈值")
        print("\n模拟结果预览:")
        print(
            result_df[
                [
                    "cutoff", "approval_rate", "approved_count",
                    "avg_pd", "el_rate", "utility_score", "efficiency_score"
                ]
            ].head(10).to_string()
        )

        return self.simulation_results

    def _compute_pareto_rank(self, df: pd.DataFrame) -> pd.Series:
        """
        计算 Pareto 非支配排序
        目标: 最大化 approval_rate, 最小化 el_rate
        """
        ranks = pd.Series(0, index=df.index)
        remaining = set(df.index)

        for rank in range(1, len(df) + 1):
            if not remaining:
                break
            current_front = []
            for i in remaining:
                dominated = False
                for j in remaining:
                    if i == j:
                        continue
                    # j 支配 i: j 的 approval >= i 且 el <= i, 至少一个严格
                    if (df.loc[j, "approval_rate"] >= df.loc[i, "approval_rate"] and
                        df.loc[j, "el_rate"] <= df.loc[i, "el_rate"] and
                        (df.loc[j, "approval_rate"] > df.loc[i, "approval_rate"] or
                         df.loc[j, "el_rate"] < df.loc[i, "el_rate"])):
                        dominated = True
                        break
                if not dominated:
                    current_front.append(i)
            for i in current_front:
                ranks.loc[i] = rank
                remaining.discard(i)

        return ranks

    # =========================================================
    # 3. Plotting
    # =========================================================
    def plot_approval_vs_loss(self, save_path: Optional[Path] = None):
        """绘制通过率 vs 预期损失率曲线"""
        if self.simulation_results is None:
            print("❌ 请先运行 simulate_cutoff_policy()")
            return

        fig, ax1 = plt.subplots(figsize=(12, 6))

        color1 = "tab:blue"
        ax1.set_xlabel("Cut-off Threshold", fontsize=12)
        ax1.set_ylabel("Approval Rate (%)", color=color1, fontsize=12)
        ax1.plot(
            self.simulation_results["cutoff"],
            self.simulation_results["approval_rate"] * 100,
            color=color1,
            marker="o",
            linewidth=2,
            label="Approval Rate"
        )
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.grid(alpha=0.3)

        ax2 = ax1.twinx()
        color2 = "tab:red"
        ax2.set_ylabel("Expected Loss Rate (%)", color=color2, fontsize=12)
        ax2.plot(
            self.simulation_results["cutoff"],
            self.simulation_results["el_rate"] * 100,
            color=color2,
            marker="s",
            linewidth=2,
            label="EL Rate"
        )
        ax2.tick_params(axis="y", labelcolor=color2)

        plt.title("Approval Rate vs Expected Loss Rate by Cut-off", fontsize=14, fontweight="bold")
        fig.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self.chart_paths["approval_vs_loss"] = str(save_path)
        plt.show()

    def plot_policy_comparison(self, save_path: Optional[Path] = None):
        """绘制策略对比图"""
        if self.simulation_results is None:
            print("❌ 请先运行 simulate_cutoff_policy()")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        axes[0, 0].plot(
            self.simulation_results["cutoff"],
            self.simulation_results["approval_rate"] * 100,
            marker="o", linewidth=2, color="steelblue"
        )
        axes[0, 0].set_xlabel("Cut-off Threshold")
        axes[0, 0].set_ylabel("Approval Rate (%)")
        axes[0, 0].set_title("Approval Rate by Cut-off")
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].plot(
            self.simulation_results["cutoff"],
            self.simulation_results["avg_pd"] * 100,
            marker="s", linewidth=2, color="coral"
        )
        axes[0, 1].set_xlabel("Cut-off Threshold")
        axes[0, 1].set_ylabel("Average PD (%)")
        axes[0, 1].set_title("Average PD of Approved Loans")
        axes[0, 1].grid(alpha=0.3)

        axes[1, 0].plot(
            self.simulation_results["cutoff"],
            self.simulation_results["total_el"] / 1e6,
            marker="^", linewidth=2, color="darkgreen"
        )
        axes[1, 0].set_xlabel("Cut-off Threshold")
        axes[1, 0].set_ylabel("Total Expected Loss (Million)")
        axes[1, 0].set_title("Total Expected Loss")
        axes[1, 0].grid(alpha=0.3)

        axes[1, 1].plot(
            self.simulation_results["cutoff"],
            self.simulation_results["el_rate"] * 100,
            marker="D", linewidth=2, color="purple"
        )
        axes[1, 1].set_xlabel("Cut-off Threshold")
        axes[1, 1].set_ylabel("EL Rate (%)")
        axes[1, 1].set_title("Expected Loss Rate (EL / EAD)")
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self.chart_paths["policy_comparison"] = str(save_path)
        plt.show()

    def plot_balance_score(self, save_path: Optional[Path] = None):
        """绘制平衡评分曲线"""
        if self.simulation_results is None:
            print("❌ 请先运行 simulate_cutoff_policy()")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(
            self.simulation_results["cutoff"],
            self.simulation_results["balance_score"],
            marker="o",
            linewidth=2,
            color="teal"
        )
        plt.title("Balance Score by Cut-off", fontsize=14, fontweight="bold")
        plt.xlabel("Cut-off Threshold")
        plt.ylabel("Balance Score")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self.chart_paths["balance_score"] = str(save_path)
        plt.show()

    # =========================================================
    # 4. Recommendation
    # =========================================================
    def recommend_cutoff(
        self,
        target_approval_rate: Optional[float] = None,
        max_el_rate: Optional[float] = None
    ) -> Dict:
        """
        推荐最优 Cut-off — 多策略推荐
        """
        if self.simulation_results is None:
            print("请先运行 simulate_cutoff_policy()")
            return {}

        df = self.simulation_results.copy()

        print("\n" + "=" * 60)
        print("Cut-off 推荐（多策略）")
        print("=" * 60)

        recommendations = {}

        # === 策略 1: 目标通过率 ===
        if target_approval_rate is not None:
            diff = abs(df["approval_rate"] - target_approval_rate)
            idx = diff.idxmin()
            rec = df.loc[idx]
            recommendations["target_approval"] = rec.to_dict()
            print(f"\n[目标通过率 {target_approval_rate:.1%}]")
            print(f"  Cut-off: {rec['cutoff']:.4f}  通过率: {rec['approval_rate']:.2%}  EL率: {rec['el_rate']:.2%}")

        # === 策略 2: EL 约束下最大化通过率 ===
        if max_el_rate is not None:
            eligible = df[df["el_rate"] <= max_el_rate]
            if len(eligible) > 0:
                idx = eligible["approval_rate"].idxmax()
                rec = df.loc[idx]
                recommendations["max_el"] = rec.to_dict()
                print(f"\n[EL <= {max_el_rate:.2%} 约束下最大通过率]")
                print(f"  Cut-off: {rec['cutoff']:.4f}  通过率: {rec['approval_rate']:.2%}  EL率: {rec['el_rate']:.2%}")
            else:
                print(f"\n[EL <= {max_el_rate:.2%}] 无满足条件的 cut-off")

        # === 策略 3: 最大化效用得分（推荐） ===
        idx = df["utility_score"].idxmax()
        rec = df.loc[idx]
        recommendations["max_utility"] = rec.to_dict()
        print(f"\n[最大化效用得分] (权重: approval={self.utility_weight_approval}, el={self.utility_weight_el})")
        print(f"  Cut-off: {rec['cutoff']:.4f}  通过率: {rec['approval_rate']:.2%}  EL率: {rec['el_rate']:.2%}")
        print(f"  效用得分: {rec['utility_score']:.4f}")

        # === 策略 4: 最高效率（通过率/EL率） ===
        idx = df["efficiency_score"].idxmax()
        rec = df.loc[idx]
        recommendations["max_efficiency"] = rec.to_dict()
        print(f"\n[最高效率] (通过率/EL率)")
        print(f"  Cut-off: {rec['cutoff']:.4f}  通过率: {rec['approval_rate']:.2%}  EL率: {rec['el_rate']:.2%}")
        print(f"  效率得分: {rec['efficiency_score']:.2f}")

        # === 策略 5: Pareto 最优（第一梯队） ===
        pareto_front = df[df["pareto_rank"] == 1]
        if len(pareto_front) > 0:
            # 选 Pareto 前沿中 utility_score 最高的
            idx = pareto_front["utility_score"].idxmax()
            rec = df.loc[idx]
            recommendations["pareto_optimal"] = rec.to_dict()
            print(f"\n[Pareto 最优] (非支配解中效用最高)")
            print(f"  Cut-off: {rec['cutoff']:.4f}  通过率: {rec['approval_rate']:.2%}  EL率: {rec['el_rate']:.2%}")
            print(f"  Pareto 前沿共 {len(pareto_front)} 个解")

        # === 策略 6: 肘部法则（边际 EL 突变点） ===
        if "marginal_el" in df.columns:
            # 找边际 EL 从低到高突变的点（二阶差分最大）
            marginal = df["marginal_el"].fillna(0).values
            if len(marginal) > 2:
                second_diff = np.diff(marginal)
                elbow_idx = np.argmax(np.abs(second_diff)) + 1
                rec = df.iloc[elbow_idx]
                recommendations["elbow"] = rec.to_dict()
                print(f"\n[肘部法则] (边际EL突变点)")
                print(f"  Cut-off: {rec['cutoff']:.4f}  通过率: {rec['approval_rate']:.2%}  EL率: {rec['el_rate']:.2%}")

        # === 策略 7: 保守/中性/激进 ===
        scenarios = {
            "conservative": {"max_el_rate": 0.02},  # EL <= 2%
            "moderate": {"max_el_rate": 0.05},       # EL <= 5%
            "aggressive": {"max_el_rate": 0.10},     # EL <= 10%
        }
        for name, params in scenarios.items():
            eligible = df[df["el_rate"] <= params["max_el_rate"]]
            if len(eligible) > 0:
                idx = eligible["approval_rate"].idxmax()
                rec = df.loc[idx]
                recommendations[name] = rec.to_dict()
                print(f"\n[{name}] (EL <= {params['max_el_rate']:.0%})")
                print(f"  Cut-off: {rec['cutoff']:.4f}  通过率: {rec['approval_rate']:.2%}  EL率: {rec['el_rate']:.2%}")

        print("\n" + "=" * 60)

        self.recommendations = recommendations
        return recommendations

    # =========================================================
    # 5. Scenario Analysis
    # =========================================================
    def scenario_analysis(
        self,
        df: pd.DataFrame,
        pd_col: str,
        scenarios: Dict[str, float],
        ead_col: str = "AMT_CREDIT",
        target_col: Optional[str] = TARGET
    ) -> pd.DataFrame:
        """
        场景分析
        """
        print("\n🎭 场景分析...")

        sim_df = self._validate_input_data(df, pd_col=pd_col, ead_col=ead_col)

        results = []
        for scenario_name, cutoff in scenarios.items():
            row = self._evaluate_single_cutoff(
                df=sim_df,
                pd_col=pd_col,
                cutoff=cutoff,
                ead_col=ead_col,
                target_col=target_col
            )
            row["scenario"] = scenario_name
            results.append(row)

        scenario_df = pd.DataFrame(results)
        scenario_df = scenario_df[
            [
                "scenario", "cutoff", "approval_rate", "approved_count", "approved_ead",
                "avg_pd", "total_el", "el_rate",
                "actual_default_rate_approved", "reject_bad_capture_rate", "approve_good_keep_rate"
            ]
        ]

        self.scenario_results = scenario_df

        print("\n场景分析结果:")
        print(scenario_df)

        return scenario_df

    # =========================================================
    # 6. JSON Export
    # =========================================================
    def save_json_summary(
        self,
        output_path: Path,
        recommendations: Optional[Dict] = None,
        chart_paths: Optional[Dict[str, str]] = None,
        meta_info: Optional[Dict] = None
    ) -> None:
        """
        保存策略模拟结果汇总到 JSON 文件
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary = {
            "metadata": {
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "lgd": self.lgd,
                **(meta_info or {})
            },
            "simulation_results": self._df_to_records(self.simulation_results),
            "scenario_results": self._df_to_records(self.scenario_results),
            "recommendations": recommendations or {},
            "charts": chart_paths or self.chart_paths,
            "chart_data_summary": {
                "approval_vs_loss": {
                    "x": "cutoff",
                    "y_left": "approval_rate",
                    "y_right": "el_rate",
                    "description": "不同 cut-off 下的通过率与预期损失率变化"
                },
                "policy_comparison": {
                    "x": "cutoff",
                    "series": ["approval_rate", "avg_pd", "total_el", "el_rate"],
                    "description": "不同 cut-off 下的策略表现对比"
                },
                "balance_score": {
                    "x": "cutoff",
                    "y": "balance_score",
                    "description": "标准化后的通过率与损失率平衡评分"
                },
                "scenario_analysis": {
                    "dimensions": [
                        "scenario", "cutoff", "approval_rate", "approved_count",
                        "approved_ead", "avg_pd", "total_el", "el_rate"
                    ],
                    "description": "保守/中性/激进策略对比"
                }
            }
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\nJSON 汇总文件已保存: {output_path}")

    # =========================================================
    # 7. Markdown Export
    # =========================================================
    def save_markdown_report(
        self,
        output_path: Path,
        recommendations: Optional[Dict] = None,
        meta_info: Optional[Dict] = None
    ) -> None:
        """
        保存 markdown 报告
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        recommendations = recommendations or self.recommendations or {}
        meta_info = meta_info or self.meta_info or {}

        lines = [
            "# 策略模拟报告 - Cut-off 优化与预期损失分析",
            "",
            "## 1. 报告概览",
            f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"- LGD: {self.lgd}",
        ]

        for k, v in meta_info.items():
            lines.append(f"- {k}: {v}")

        lines.extend([
            "",
            "## 2. Cut-off 模拟结果概览",
        ])

        if self.simulation_results is not None and not self.simulation_results.empty:
            best_balance_idx = self.simulation_results["balance_score"].idxmax()
            best_balance = self.simulation_results.loc[best_balance_idx]

            min_el_idx = self.simulation_results["el_rate"].idxmin()
            min_el = self.simulation_results.loc[min_el_idx]

            max_approval_idx = self.simulation_results["approval_rate"].idxmax()
            max_approval = self.simulation_results.loc[max_approval_idx]

            lines.extend([
                f"- 共模拟 cut-off 数量: {len(self.simulation_results)}",
                f"- 平衡策略最优 cut-off: {best_balance['cutoff']:.4f}",
                f"- 平衡策略通过率: {self._format_pct(best_balance['approval_rate'])}",
                f"- 平衡策略 EL 率: {self._format_pct(best_balance['el_rate'])}",
                f"- 最低 EL 率 cut-off: {min_el['cutoff']:.4f}，EL率 = {self._format_pct(min_el['el_rate'])}",
                f"- 最高通过率 cut-off: {max_approval['cutoff']:.4f}，通过率 = {self._format_pct(max_approval['approval_rate'])}",
                "",
                "### 2.1 Cut-off 模拟结果表（前10行）",
                "",
                self.simulation_results.head(10).to_markdown(index=False),
            ])
        else:
            lines.append("- 暂无模拟结果")

        lines.extend([
            "",
            "## 3. 推荐 Cut-off",
        ])

        if recommendations:
            for rec_type, rec in recommendations.items():
                lines.extend([
                    f"### 3.{list(recommendations.keys()).index(rec_type)+1} {rec_type}",
                    f"- 推荐 Cut-off: {self._format_num(rec.get('cutoff'), 4)}",
                    f"- 通过率: {self._format_pct(rec.get('approval_rate'))}",
                    f"- EL 率: {self._format_pct(rec.get('el_rate'))}",
                    f"- 平均 PD: {self._format_pct(rec.get('avg_pd'))}",
                    ""
                ])
        else:
            lines.append("- 暂无推荐结果")
            lines.append("")

        lines.append("## 4. 场景分析")

        if self.scenario_results is not None and not self.scenario_results.empty:
            lines.extend([
                "",
                self.scenario_results.to_markdown(index=False),
                ""
            ])
        else:
            lines.extend([
                "- 暂无场景分析结果",
                ""
            ])

        lines.append("## 5. 图表输出")

        if self.chart_paths:
            for chart_name, chart_path in self.chart_paths.items():
                lines.append(f"- {chart_name}: {chart_path}")
        else:
            lines.append("- 暂无图表输出")

        lines.extend([
            "",
            "## 6. 结论总结",
        ])

        if self.simulation_results is not None and not self.simulation_results.empty:
            best_balance_idx = self.simulation_results["balance_score"].idxmax()
            best_balance = self.simulation_results.loc[best_balance_idx]

            lines.extend([
                f"- 从平衡策略看，推荐 cut-off 为 {best_balance['cutoff']:.4f}。",
                f"- 在该阈值下，通过率约为 {self._format_pct(best_balance['approval_rate'])}，EL 率约为 {self._format_pct(best_balance['el_rate'])}。",
                "- 随着 cut-off 提高，通过率通常上升，但组合预期损失也会同步上升。",
                "- 因此，审批策略需要在业务规模与风险控制之间进行权衡。"
            ])
        else:
            lines.append("- 当前无足够结果生成结论。")

        output_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"\nMarkdown 报告已保存: {output_path}")


def main():
    """主函数 - 策略模拟示例"""
    from pd_model import PDModel

    print("=" * 70)
    print("🏦 Cut-off 策略模拟与预期损失分析")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. 加载数据与模型
    # ---------------------------------------------------------
    print("\n📂 加载数据和模型...")
    df = pd.read_csv(PROCESSED_DATA_DIR / "train_with_features.csv")
    model = PDModel.load_model(MODEL_DIR / "pd_model_xgboost.pkl")

    print(f"输入数据: {df.shape[0]:,} 行 × {df.shape[1]:,} 列")

    # ---------------------------------------------------------
    # 2. 准备特征并预测 PD
    # ---------------------------------------------------------
    X = df.drop(columns=[TARGET] + EXCLUDE_FEATURES, errors="ignore")
    X = X.select_dtypes(include=[np.number])

    X_aligned = model.align_features(X)
    df["PD"] = model.predict(X_aligned)

    print("\nPD 预测完成")
    print(df["PD"].describe())

    # ---------------------------------------------------------
    # 3. 策略模拟
    # ---------------------------------------------------------
    simulator = PolicySimulator(lgd=LGD)

    simulator.meta_info = {
        "model_name": "pd_model_xgboost.pkl",
        "pd_column": "PD",
        "ead_column": "AMT_CREDIT",
        "target_column": TARGET,
        "input_rows": int(df.shape[0]),
        "input_columns": int(df.shape[1]),
        "cutoff_range": list(CUTOFF_RANGE)
    }

    results = simulator.simulate_cutoff_policy(
        df=df,
        pd_col="PD",
        ead_col="AMT_CREDIT",
        cutoff_range=CUTOFF_RANGE,
        target_col=TARGET
    )

    print("\n模拟结果（前5行）:")
    print(results.head())

    # ---------------------------------------------------------
    # 4. 可视化
    # ---------------------------------------------------------
    approval_vs_loss_path = VIZ_DIR / "approval_vs_loss.png"
    policy_comparison_path = VIZ_DIR / "policy_comparison.png"
    balance_score_path = VIZ_DIR / "balance_score.png"

    simulator.plot_approval_vs_loss(save_path=approval_vs_loss_path)
    simulator.plot_policy_comparison(save_path=policy_comparison_path)
    simulator.plot_balance_score(save_path=balance_score_path)

    # ---------------------------------------------------------
    # 5. Cut-off 推荐
    # ---------------------------------------------------------
    recommendations = simulator.recommend_cutoff(
        target_approval_rate=0.80,
        max_el_rate=0.05
    )

    # ---------------------------------------------------------
    # 6. 场景分析
    # ---------------------------------------------------------
    scenarios = {
        "保守策略": 0.03,
        "中性策略": 0.05,
        "激进策略": 0.08,
    }

    scenario_df = simulator.scenario_analysis(
        df=df,
        pd_col="PD",
        scenarios=scenarios,
        ead_col="AMT_CREDIT",
        target_col=TARGET
    )

    # 避免未使用变量告警
    _ = scenario_df

    # ---------------------------------------------------------
    # 7. 输出 JSON 与 Markdown
    # ---------------------------------------------------------
    simulator.save_json_summary(
        output_path=REPORT_DIR / "policy_simulation_summary.json",
        recommendations=recommendations,
        chart_paths=simulator.chart_paths,
        meta_info=simulator.meta_info
    )

    simulator.save_markdown_report(
        output_path=REPORT_DIR / "policy_simulation_report.md",
        recommendations=recommendations,
        meta_info=simulator.meta_info
    )

    print("\n" + "=" * 70)
    print("策略模拟全部完成")
    print("=" * 70)


if __name__ == "__main__":
    setup_logging()
    main()
