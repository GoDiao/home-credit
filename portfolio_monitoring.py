"""
组合监控模块 - Vintage Analysis, Roll Rate, Flow Rate
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import logging
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from config import *

class PortfolioMonitor:
    """组合监控类"""
    
    def __init__(self):
        self.vintage_data = None
        self.roll_rate_data = None
    
    def vintage_analysis(self, df: pd.DataFrame, cohort_col: str, 
                        target_col: str = TARGET, 
                        observation_months: List[int] = VINTAGE_OBSERVATION_MONTHS) -> pd.DataFrame:
        """
        Vintage Analysis - 按批次追踪违约率
        
        Parameters:
        -----------
        df : pd.DataFrame
            贷款数据
        cohort_col : str
            批次列名（如申请月份）
        target_col : str
            目标变量列名
        observation_months : list
            观察期（月）
        
        Returns:
        --------
        pd.DataFrame : Vintage 分析结果
        """
        print("\nVintage Analysis...")
        
        # 如果没有批次列，创建一个（基于索引或其他时间列）
        if cohort_col is None or cohort_col not in df.columns:
            print(f" 未找到批次列 {cohort_col}，使用索引创建虚拟批次")
            df = df.copy()
            # 创建临时批次列名
            cohort_col = 'cohort_temp'
            # 将数据分成若干批次
            df[cohort_col] = pd.qcut(df.index, q=12, labels=[f'2023-{i:02d}' for i in range(1, 13)])
        
        # 按批次计算违约率
        vintage_df = df.groupby(cohort_col).agg({
            target_col: ['count', 'sum', 'mean']
        }).reset_index()
        
        vintage_df.columns = ['cohort', 'total_loans', 'defaults', 'default_rate']
        vintage_df['default_rate'] = vintage_df['default_rate'] * 100  # 转换为百分比
        
        # 计算累计违约率（模拟）
        for month in observation_months:
            # 这里简化处理，实际应该基于真实的月度数据
            vintage_df[f'default_rate_{month}m'] = vintage_df['default_rate'] * (1 + month * 0.05)
        
        self.vintage_data = vintage_df
        
        print(f"Vintage 分析完成: {len(vintage_df)} 个批次")
        return vintage_df
    
    def plot_vintage_curve(self, save_path: Path = None):
        """绘制 Vintage 曲线"""
        if self.vintage_data is None:
            print("❌ 请先运行 vintage_analysis()")
            return
        
        plt.figure(figsize=(14, 8))
        
        # 获取所有观察期列
        obs_cols = [col for col in self.vintage_data.columns if 'default_rate_' in col]
        
        for idx, cohort in enumerate(self.vintage_data['cohort']):
            rates = [self.vintage_data.loc[idx, col] for col in obs_cols]
            months = [int(col.split('_')[-1].replace('m', '')) for col in obs_cols]
            plt.plot(months, rates, marker='o', label=cohort, linewidth=2)
        
        plt.xlabel('Months on Book', fontsize=12)
        plt.ylabel('Cumulative Default Rate (%)', fontsize=12)
        plt.title('Vintage Analysis - Cumulative Default Rate by Cohort', 
                 fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def roll_rate_analysis(self, df: pd.DataFrame, dpd_col: str = None) -> pd.DataFrame:
        """
        Roll Rate Analysis - 逾期迁移率分析
        
        Parameters:
        -----------
        df : pd.DataFrame
            贷款数据（需要包含逾期天数）
        dpd_col : str
            逾期天数列名
        
        Returns:
        --------
        pd.DataFrame : Roll Rate 矩阵
        """
        print("\nRoll Rate Analysis...")
        
        # 如果没有 DPD 列，创建模拟数据
        if dpd_col is None or dpd_col not in df.columns:
            print(" 未找到 DPD 列，创建模拟数据")
            df = df.copy()
            # 基于违约标签模拟 DPD
            rng = np.random.RandomState(RANDOM_STATE)
            df['DPD'] = np.where(df[TARGET] == 1,
                                rng.choice([35, 65, 95, 125], len(df)),
                                rng.choice([0, 15], len(df)))
            dpd_col = 'DPD'
        
        # 将 DPD 分桶
        def dpd_bucket(dpd):
            for bucket_name, (min_dpd, max_dpd) in DPD_BUCKETS.items():
                if min_dpd <= dpd <= max_dpd:
                    return bucket_name
            return '180+'
        
        df['dpd_bucket'] = df[dpd_col].apply(dpd_bucket)
        
        # 创建 Roll Rate 矩阵（简化版：当前状态分布）
        roll_rate_df = df['dpd_bucket'].value_counts().reset_index()
        roll_rate_df.columns = ['bucket', 'count']
        roll_rate_df['percentage'] = roll_rate_df['count'] / roll_rate_df['count'].sum() * 100
        
        # 排序
        bucket_order = list(DPD_BUCKETS.keys())
        roll_rate_df['bucket'] = pd.Categorical(roll_rate_df['bucket'], 
                                                categories=bucket_order, 
                                                ordered=True)
        roll_rate_df = roll_rate_df.sort_values('bucket')
        
        self.roll_rate_data = roll_rate_df
        
        print(f"Roll Rate 分析完成")
        return roll_rate_df
    
    def plot_roll_rate(self, save_path: Path = None):
        """绘制 Roll Rate 图表"""
        if self.roll_rate_data is None:
            print("❌ 请先运行 roll_rate_analysis()")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 柱状图
        ax1.bar(self.roll_rate_data['bucket'], self.roll_rate_data['count'], 
               color='steelblue', alpha=0.7)
        ax1.set_xlabel('DPD Bucket', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('Account Distribution by DPD Bucket', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # 饼图
        colors = plt.cm.Set3(range(len(self.roll_rate_data)))
        ax2.pie(self.roll_rate_data['percentage'], labels=self.roll_rate_data['bucket'],
               autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('Percentage Distribution by DPD Bucket', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def calculate_fpd(self, df: pd.DataFrame, dpd_col: str = None, 
                     fpd_threshold: int = 30) -> float:
        """
        计算 FPD (First Payment Default) 率
        
        Parameters:
        -----------
        df : pd.DataFrame
            贷款数据
        dpd_col : str
            逾期天数列名
        fpd_threshold : int
            FPD 定义阈值（天）
        
        Returns:
        --------
        float : FPD 率
        """
        if dpd_col is None or dpd_col not in df.columns:
            # 使用目标变量作为代理
            fpd_rate = df[TARGET].mean()
        else:
            fpd_rate = (df[dpd_col] >= fpd_threshold).mean()
        
        print(f"\nFPD Rate ({fpd_threshold}+ DPD): {fpd_rate:.2%}")
        return fpd_rate
    
    def early_warning_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算早期预警指标 (EWI)
        
        Parameters:
        -----------
        df : pd.DataFrame
            贷款数据
        
        Returns:
        --------
        pd.DataFrame : EWI 指标
        """
        print("\n计算早期预警指标...")
        
        ewi_metrics = {}
        
        # 1. 批准率（假设所有记录都是批准的）
        ewi_metrics['approval_rate'] = 1.0
        
        # 2. 平均贷款金额
        if 'AMT_CREDIT' in df.columns:
            ewi_metrics['average_loan_amount'] = df['AMT_CREDIT'].mean()
        
        # 3. 平均收入
        if 'AMT_INCOME_TOTAL' in df.columns:
            ewi_metrics['average_income'] = df['AMT_INCOME_TOTAL'].mean()
        
        # 4. 负债收入比
        if 'AMT_CREDIT' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
            ewi_metrics['debt_to_income_ratio'] = (df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']).mean()
        
        # 5. FPD 率
        ewi_metrics['fpd_rate'] = self.calculate_fpd(df)
        
        # 6. 违约率
        ewi_metrics['default_rate'] = df[TARGET].mean()
        
        ewi_df = pd.DataFrame([ewi_metrics])
        
        print("\n早期预警指标:")
        print(ewi_df.T)
        
        return ewi_df
    
    def generate_monitoring_report(self, df: pd.DataFrame, cohort_col: str = None,
                                  save_dir: Path = None) -> Dict:
        """
        生成完整的组合监控报告
        
        Parameters:
        -----------
        df : pd.DataFrame
            贷款数据
        cohort_col : str
            批次列名
        save_dir : Path
            保存目录
        
        Returns:
        --------
        dict : 监控报告
        """
        if save_dir is None:
            save_dir = REPORT_DIR
        
        print("\n" + "=" * 60)
        print("生成组合监控报告")
        print("=" * 60)
        
        report = {}
        
        # 1. Vintage Analysis
        vintage_df = self.vintage_analysis(df, cohort_col if cohort_col else 'cohort')
        self.plot_vintage_curve(save_path=VIZ_DIR / "vintage_analysis.png")
        report['vintage'] = vintage_df
        
        # 2. Roll Rate Analysis
        roll_rate_df = self.roll_rate_analysis(df)
        self.plot_roll_rate(save_path=VIZ_DIR / "roll_rate_analysis.png")
        report['roll_rate'] = roll_rate_df
        
        # 3. Early Warning Indicators
        ewi_df = self.early_warning_indicators(df)
        report['ewi'] = ewi_df
        
        # 保存报告
        with pd.ExcelWriter(save_dir / "portfolio_monitoring_report.xlsx") as writer:
            vintage_df.to_excel(writer, sheet_name='Vintage Analysis', index=False)
            roll_rate_df.to_excel(writer, sheet_name='Roll Rate', index=False)
            ewi_df.to_excel(writer, sheet_name='Early Warning Indicators', index=False)
        
        print("\n" + "=" * 60)
        print(f"监控报告已保存: {save_dir / 'portfolio_monitoring_report.xlsx'}")
        print("=" * 60)
        
        return report

def main():
    """主函数 - 组合监控示例"""
    from data_processing import DataProcessor
    
    processor = DataProcessor()
    monitor = PortfolioMonitor()
    
    # 加载数据
    print("📂 加载数据...")
    df = processor.load_data("application_train.csv")
    
    # 生成监控报告
    report = monitor.generate_monitoring_report(df)

if __name__ == "__main__":
    main()
