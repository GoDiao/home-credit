"""
特征工程模块
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import logging
import warnings

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

from config import *


class FeatureEngineer:
    """特征工程类"""

    def __init__(self):
        self.bureau_agg = None
        self.prev_app_agg = None

        # 新增：记录过程信息
        self.last_summary: Dict[str, Any] = {}
        self.created_basic_features: List[str] = []
        self.merged_missing_filled_cols: List[str] = []

    def create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建基础衍生特征

        Parameters:
        -----------
        df : pd.DataFrame
            输入数据（已处理后的 application 主表）

        Returns:
        --------
        pd.DataFrame : 添加特征后的数据
        """
        print("\n创建基础衍生特征...")

        df = df.copy()
        original_columns = df.columns.tolist()

        # 1. 收入相关特征
        if 'AMT_INCOME_TOTAL' in df.columns and 'AMT_CREDIT' in df.columns:
            df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1)

            if 'AMT_ANNUITY' in df.columns:
                df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1)

        # 2. 年龄相关特征
        if 'DAYS_BIRTH' in df.columns:
            df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365

            age_group = pd.cut(
                df['AGE_YEARS'],
                bins=[0, 25, 35, 45, 55, 100],
                labels=['<25', '25-35', '35-45', '45-55', '55+']
            )
            df['AGE_GROUP'] = age_group.cat.codes.astype(int)

        # 3. 就业相关特征
        if 'DAYS_EMPLOYED' in df.columns:
            # 这里不再重复处理 DAYS_EMPLOYED == 365243
            # 该规则已经在 data_processing.py 中完成
            df['EMPLOYMENT_YEARS'] = -df['DAYS_EMPLOYED'] / 365

            if 'AMT_INCOME_TOTAL' in df.columns:
                df['INCOME_PER_EMPLOYMENT_YEAR'] = df['AMT_INCOME_TOTAL'] / (df['EMPLOYMENT_YEARS'] + 1)

        # 4. 家庭相关特征
        if 'CNT_CHILDREN' in df.columns and 'CNT_FAM_MEMBERS' in df.columns:
            df['CHILDREN_RATIO'] = df['CNT_CHILDREN'] / (df['CNT_FAM_MEMBERS'] + 1)

        # 5. 资产相关特征
        if 'OWN_CAR_AGE' in df.columns:
            df['OWN_CAR_AGE_MISSING'] = df['OWN_CAR_AGE'].isnull().astype(int)
            df['CAR_AGE_FILLED'] = df['OWN_CAR_AGE'].fillna(0)

        # 6. 文档提交数量
        doc_cols = [col for col in df.columns if 'FLAG_DOCUMENT' in col]
        if doc_cols:
            df['DOCUMENT_COUNT'] = df[doc_cols].sum(axis=1)

        # 7. 外部数据源评分
        ext_source_cols = [col for col in df.columns if 'EXT_SOURCE' in col]
        if ext_source_cols:
            df['EXT_SOURCE_MEAN'] = df[ext_source_cols].mean(axis=1)
            df['EXT_SOURCE_STD'] = df[ext_source_cols].std(axis=1)
            df['EXT_SOURCE_MIN'] = df[ext_source_cols].min(axis=1)
            df['EXT_SOURCE_MAX'] = df[ext_source_cols].max(axis=1)

        new_columns = [col for col in df.columns if col not in original_columns]
        self.created_basic_features = new_columns

        print(f"创建了 {len(new_columns)} 个基础特征")
        print(f"新增特征: {new_columns}")

        self.last_summary['basic_features'] = {
            'created_count': len(new_columns),
            'created_features': new_columns
        }

        return df

    def aggregate_bureau_data(self, bureau_df: pd.DataFrame) -> pd.DataFrame:
        """
        聚合 Bureau 数据（征信局历史）

        Parameters:
        -----------
        bureau_df : pd.DataFrame
            Bureau 表数据

        Returns:
        --------
        pd.DataFrame : 聚合后的特征
        """
        print("\n聚合 Bureau 数据...")

        if bureau_df is None or bureau_df.empty:
            print("Bureau 数据为空，跳过聚合")
            self.last_summary['bureau_aggregation'] = {
                'executed': False,
                'output_shape': None,
                'feature_count': 0
            }
            return None

        if 'SK_ID_CURR' not in bureau_df.columns:
            print("Bureau 数据缺少 SK_ID_CURR，跳过聚合")
            self.last_summary['bureau_aggregation'] = {
                'executed': False,
                'output_shape': None,
                'feature_count': 0
            }
            return None

        candidate_agg_dict = {
            'DAYS_CREDIT': ['min', 'max', 'mean'],
            'CREDIT_DAY_OVERDUE': ['max', 'mean'],
            'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
            'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
            'AMT_CREDIT_SUM': ['sum', 'mean', 'max'],
            'AMT_CREDIT_SUM_DEBT': ['sum', 'mean', 'max'],
            'AMT_CREDIT_SUM_OVERDUE': ['sum', 'mean', 'max'],
        }

        agg_dict = {}
        skipped_cols = []

        for col, agg_funcs in candidate_agg_dict.items():
            if col in bureau_df.columns:
                agg_dict[col] = agg_funcs
            else:
                skipped_cols.append(col)

        if skipped_cols:
            print(f"Bureau 聚合时跳过缺失列: {skipped_cols}")

        if not agg_dict:
            print("Bureau 无可用聚合字段，跳过聚合")
            self.last_summary['bureau_aggregation'] = {
                'executed': False,
                'output_shape': None,
                'feature_count': 0
            }
            return None

        bureau_agg = bureau_df.groupby('SK_ID_CURR').agg(agg_dict)
        bureau_agg.columns = ['BUREAU_' + '_'.join(col).upper() for col in bureau_agg.columns]
        bureau_agg.reset_index(inplace=True)

        # 额外特征
        bureau_agg['BUREAU_LOAN_COUNT'] = bureau_df.groupby('SK_ID_CURR').size().reindex(
            bureau_agg['SK_ID_CURR']
        ).values

        if 'CREDIT_ACTIVE' in bureau_df.columns:
            active_count = bureau_df[bureau_df['CREDIT_ACTIVE'] == 'Active'].groupby('SK_ID_CURR').size()
            bureau_agg['BUREAU_ACTIVE_LOAN_COUNT'] = bureau_agg['SK_ID_CURR'].map(active_count).fillna(0)
        else:
            print("Bureau 缺少 CREDIT_ACTIVE，BUREAU_ACTIVE_LOAN_COUNT 用 0 填充")
            bureau_agg['BUREAU_ACTIVE_LOAN_COUNT'] = 0

        # 简单比率特征（可选增强）
        if 'BUREAU_AMT_CREDIT_SUM_DEBT_SUM' in bureau_agg.columns and 'BUREAU_AMT_CREDIT_SUM_SUM' in bureau_agg.columns:
            bureau_agg['BUREAU_DEBT_CREDIT_RATIO'] = (
                bureau_agg['BUREAU_AMT_CREDIT_SUM_DEBT_SUM'] /
                (bureau_agg['BUREAU_AMT_CREDIT_SUM_SUM'] + 1)
            )

        self.bureau_agg = bureau_agg

        feature_count = bureau_agg.shape[1] - 1
        print(f"Bureau 聚合完成: {feature_count} 个特征")

        self.last_summary['bureau_aggregation'] = {
            'executed': True,
            'output_shape': bureau_agg.shape,
            'feature_count': feature_count
        }

        return bureau_agg

    def aggregate_credit_card_balance(self, cc_df: pd.DataFrame) -> pd.DataFrame:
        """
        聚合 Credit Card Balance 数据（信用卡月度账单）

        Parameters:
        -----------
        cc_df : pd.DataFrame
            Credit Card Balance 表数据

        Returns:
        --------
        pd.DataFrame : 聚合后的特征
        """
        print("\n聚合 Credit Card Balance 数据...")

        if cc_df is None or cc_df.empty:
            print("  Credit Card Balance 数据为空，跳过聚合")
            self.last_summary['cc_aggregation'] = {'executed': False, 'feature_count': 0}
            return None

        if 'SK_ID_CURR' not in cc_df.columns:
            print("  Credit Card Balance 缺少 SK_ID_CURR，跳过聚合")
            self.last_summary['cc_aggregation'] = {'executed': False, 'feature_count': 0}
            return None

        # 只保留申请前的数据（MONTHS_BALANCE < 0 已经是历史数据）
        cc = cc_df.copy()

        agg_dict = {
            'AMT_BALANCE': ['mean', 'max', 'min', 'std'],
            'AMT_CREDIT_LIMIT_ACTUAL': ['mean', 'max'],
            'AMT_DRAWINGS_CURRENT': ['mean', 'max'],
            'AMT_PAYMENT_CURRENT': ['mean', 'max'],
            'AMT_PAYMENT_TOTAL_CURRENT': ['mean'],
            'AMT_TOTAL_RECEIVABLE': ['mean', 'max'],
            'CNT_DRAWINGS_CURRENT': ['mean', 'max'],
            'SK_DPD': ['mean', 'max'],
            'SK_DPD_DEF': ['mean', 'max'],
        }

        # 动态检查列是否存在
        final_agg = {col: funcs for col, funcs in agg_dict.items() if col in cc.columns}
        skipped = [col for col in agg_dict if col not in cc.columns]
        if skipped:
            print(f"  跳过缺失列: {skipped}")

        if not final_agg:
            print("  无可用聚合字段，跳过")
            self.last_summary['cc_aggregation'] = {'executed': False, 'feature_count': 0}
            return None

        cc_agg = cc.groupby('SK_ID_CURR').agg(final_agg)
        cc_agg.columns = ['CC_' + '_'.join(col).upper() for col in cc_agg.columns]
        cc_agg.reset_index(inplace=True)

        # 衍生特征：信用额度使用率
        if 'CC_AMT_BALANCE_MEAN' in cc_agg.columns and 'CC_AMT_CREDIT_LIMIT_ACTUAL_MEAN' in cc_agg.columns:
            cc_agg['CC_UTILIZATION_RATIO'] = (
                cc_agg['CC_AMT_BALANCE_MEAN'] /
                (cc_agg['CC_AMT_CREDIT_LIMIT_ACTUAL_MEAN'] + 1)
            )

        # 衍生特征：还款比率
        if 'CC_AMT_PAYMENT_CURRENT_MEAN' in cc_agg.columns and 'CC_AMT_BALANCE_MEAN' in cc_agg.columns:
            cc_agg['CC_PAYMENT_RATIO'] = (
                cc_agg['CC_AMT_PAYMENT_CURRENT_MEAN'] /
                (cc_agg['CC_AMT_BALANCE_MEAN'] + 1)
            )

        # 衍生特征：逾期比例
        cc_agg['CC_MONTHS_COUNT'] = cc.groupby('SK_ID_CURR').size().reindex(cc_agg['SK_ID_CURR']).values

        if 'SK_DPD' in cc.columns:
            cc_agg['CC_DPD_RATIO'] = cc.groupby('SK_ID_CURR')['SK_DPD'].apply(
                lambda x: (x > 0).mean()
            ).reindex(cc_agg['SK_ID_CURR']).fillna(0).values

        # 活跃账户状态
        if 'NAME_CONTRACT_STATUS' in cc.columns:
            active_ratio = cc[cc['NAME_CONTRACT_STATUS'] == 'Active'].groupby('SK_ID_CURR').size()
            total_count = cc.groupby('SK_ID_CURR').size()
            cc_agg['CC_ACTIVE_RATIO'] = (
                cc_agg['SK_ID_CURR'].map(active_ratio).fillna(0) /
                (cc_agg['SK_ID_CURR'].map(total_count) + 1)
            )

        feature_count = cc_agg.shape[1] - 1
        print(f"  Credit Card Balance 聚合完成: {feature_count} 个特征")

        self.last_summary['cc_aggregation'] = {
            'executed': True,
            'output_shape': cc_agg.shape,
            'feature_count': feature_count
        }

        return cc_agg

    def aggregate_installments_payments(self, ip_df: pd.DataFrame) -> pd.DataFrame:
        """
        聚合 Installments Payments 数据（还款记录）

        Parameters:
        -----------
        ip_df : pd.DataFrame
            Installments Payments 表数据

        Returns:
        --------
        pd.DataFrame : 聚合后的特征
        """
        print("\n聚合 Installments Payments 数据...")

        if ip_df is None or ip_df.empty:
            print("  Installments Payments 数据为空，跳过聚合")
            self.last_summary['ip_aggregation'] = {'executed': False, 'feature_count': 0}
            return None

        if 'SK_ID_CURR' not in ip_df.columns:
            print("  Installments Payments 缺少 SK_ID_CURR，跳过聚合")
            self.last_summary['ip_aggregation'] = {'executed': False, 'feature_count': 0}
            return None

        ip = ip_df.copy()

        # 计算还款延迟（天）：实际还款日 - 应还日，正值表示逾期
        if 'DAYS_ENTRY_PAYMENT' in ip.columns and 'DAYS_INSTALMENT' in ip.columns:
            ip['PAYMENT_DELAY'] = ip['DAYS_ENTRY_PAYMENT'] - ip['DAYS_INSTALMENT']

        # 计算还款差额：实际还款 - 应还金额，负值表示少还
        if 'AMT_PAYMENT' in ip.columns and 'AMT_INSTALMENT' in ip.columns:
            ip['PAYMENT_DIFF'] = ip['AMT_PAYMENT'] - ip['AMT_INSTALMENT']
            ip['PAYMENT_RATIO'] = ip['AMT_PAYMENT'] / (ip['AMT_INSTALMENT'] + 1)

        agg_dict = {}
        if 'PAYMENT_DELAY' in ip.columns:
            agg_dict['PAYMENT_DELAY'] = ['mean', 'max', 'min', 'std']
        if 'PAYMENT_DIFF' in ip.columns:
            agg_dict['PAYMENT_DIFF'] = ['mean', 'max', 'min']
        if 'PAYMENT_RATIO' in ip.columns:
            agg_dict['PAYMENT_RATIO'] = ['mean', 'min']
        if 'NUM_INSTALMENT_VERSION' in ip.columns:
            agg_dict['NUM_INSTALMENT_VERSION'] = ['max']
        if 'AMT_INSTALMENT' in ip.columns:
            agg_dict['AMT_INSTALMENT'] = ['sum', 'mean']
        if 'AMT_PAYMENT' in ip.columns:
            agg_dict['AMT_PAYMENT'] = ['sum', 'mean']

        if not agg_dict:
            print("  无可用聚合字段，跳过")
            self.last_summary['ip_aggregation'] = {'executed': False, 'feature_count': 0}
            return None

        ip_agg = ip.groupby('SK_ID_CURR').agg(agg_dict)
        ip_agg.columns = ['IP_' + '_'.join(col).upper() for col in ip_agg.columns]
        ip_agg.reset_index(inplace=True)

        # 衍生特征：逾期次数
        if 'PAYMENT_DELAY' in ip.columns:
            ip_agg['IP_LATE_COUNT'] = ip[ip['PAYMENT_DELAY'] > 0].groupby('SK_ID_CURR').size()
            ip_agg['IP_LATE_COUNT'] = ip_agg['IP_LATE_COUNT'].reindex(ip_agg['SK_ID_CURR']).fillna(0)

            total_per_user = ip.groupby('SK_ID_CURR').size()
            ip_agg['IP_LATE_RATIO'] = (
                ip_agg['IP_LATE_COUNT'] /
                (ip_agg['SK_ID_CURR'].map(total_per_user) + 1)
            )

        # 衍生特征：少还次数
        if 'PAYMENT_DIFF' in ip.columns:
            ip_agg['IP_UNDERPAY_COUNT'] = ip[ip['PAYMENT_DIFF'] < 0].groupby('SK_ID_CURR').size()
            ip_agg['IP_UNDERPAY_COUNT'] = ip_agg['IP_UNDERPAY_COUNT'].reindex(ip_agg['SK_ID_CURR']).fillna(0)

        ip_agg['IP_TOTAL_COUNT'] = ip.groupby('SK_ID_CURR').size().reindex(ip_agg['SK_ID_CURR']).values

        feature_count = ip_agg.shape[1] - 1
        print(f"  Installments Payments 聚合完成: {feature_count} 个特征")

        self.last_summary['ip_aggregation'] = {
            'executed': True,
            'output_shape': ip_agg.shape,
            'feature_count': feature_count
        }

        return ip_agg

    def aggregate_pos_cash_balance(self, pos_df: pd.DataFrame) -> pd.DataFrame:
        """
        聚合 POS_CASH Balance 数据（POS/现金贷款月度状态）

        Parameters:
        -----------
        pos_df : pd.DataFrame
            POS_CASH Balance 表数据

        Returns:
        --------
        pd.DataFrame : 聚合后的特征
        """
        print("\n聚合 POS_CASH Balance 数据...")

        if pos_df is None or pos_df.empty:
            print("  POS_CASH Balance 数据为空，跳过聚合")
            self.last_summary['pos_aggregation'] = {'executed': False, 'feature_count': 0}
            return None

        if 'SK_ID_CURR' not in pos_df.columns:
            print("  POS_CASH Balance 缺少 SK_ID_CURR，跳过聚合")
            self.last_summary['pos_aggregation'] = {'executed': False, 'feature_count': 0}
            return None

        pos = pos_df.copy()

        agg_dict = {
            'SK_DPD': ['mean', 'max'],
            'SK_DPD_DEF': ['mean', 'max'],
        }
        if 'CNT_INSTALMENT' in pos.columns:
            agg_dict['CNT_INSTALMENT'] = ['mean', 'max']
        if 'CNT_INSTALMENT_FUTURE' in pos.columns:
            agg_dict['CNT_INSTALMENT_FUTURE'] = ['mean', 'max']

        final_agg = {col: funcs for col, funcs in agg_dict.items() if col in pos.columns}

        if not final_agg:
            print("  无可用聚合字段，跳过")
            self.last_summary['pos_aggregation'] = {'executed': False, 'feature_count': 0}
            return None

        pos_agg = pos.groupby('SK_ID_CURR').agg(final_agg)
        pos_agg.columns = ['POS_' + '_'.join(col).upper() for col in pos_agg.columns]
        pos_agg.reset_index(inplace=True)

        # 衍生特征：逾期比例
        if 'SK_DPD' in pos.columns:
            pos_agg['POS_DPD_RATIO'] = pos.groupby('SK_ID_CURR')['SK_DPD'].apply(
                lambda x: (x > 0).mean()
            ).reindex(pos_agg['SK_ID_CURR']).fillna(0).values

        # 衍生特征：还款进度
        if 'CNT_INSTALMENT' in pos.columns and 'CNT_INSTALMENT_FUTURE' in pos.columns:
            pos_agg['POS_INSTALLMENT_PROGRESS'] = pos.groupby('SK_ID_CURR').apply(
                lambda g: (g['CNT_INSTALMENT'] - g['CNT_INSTALMENT_FUTURE']).mean()
            ).reindex(pos_agg['SK_ID_CURR']).fillna(0).values

        # 月度记录数
        pos_agg['POS_MONTHS_COUNT'] = pos.groupby('SK_ID_CURR').size().reindex(pos_agg['SK_ID_CURR']).values

        # 活跃账户比例
        if 'NAME_CONTRACT_STATUS' in pos.columns:
            active_ratio = pos[pos['NAME_CONTRACT_STATUS'] == 'Active'].groupby('SK_ID_CURR').size()
            total_count = pos.groupby('SK_ID_CURR').size()
            pos_agg['POS_ACTIVE_RATIO'] = (
                pos_agg['SK_ID_CURR'].map(active_ratio).fillna(0) /
                (pos_agg['SK_ID_CURR'].map(total_count) + 1)
            )

        feature_count = pos_agg.shape[1] - 1
        print(f"  POS_CASH Balance 聚合完成: {feature_count} 个特征")

        self.last_summary['pos_aggregation'] = {
            'executed': True,
            'output_shape': pos_agg.shape,
            'feature_count': feature_count
        }

        return pos_agg

    def aggregate_previous_application(self, prev_app_df: pd.DataFrame) -> pd.DataFrame:
        """
        聚合 Previous Application 数据（历史申请）

        Parameters:
        -----------
        prev_app_df : pd.DataFrame
            Previous Application 表数据

        Returns:
        --------
        pd.DataFrame : 聚合后的特征
        """
        print("\n聚合 Previous Application 数据...")

        if prev_app_df is None or prev_app_df.empty:
            print("Previous Application 数据为空，跳过聚合")
            self.last_summary['previous_application_aggregation'] = {
                'executed': False,
                'output_shape': None,
                'feature_count': 0
            }
            return None

        if 'SK_ID_CURR' not in prev_app_df.columns:
            print("Previous Application 缺少 SK_ID_CURR，跳过聚合")
            self.last_summary['previous_application_aggregation'] = {
                'executed': False,
                'output_shape': None,
                'feature_count': 0
            }
            return None

        candidate_agg_dict = {
            'AMT_ANNUITY': ['min', 'max', 'mean'],
            'AMT_APPLICATION': ['min', 'max', 'mean'],
            'AMT_CREDIT': ['min', 'max', 'mean'],
            'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
            'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
            'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
            'DAYS_DECISION': ['min', 'max', 'mean'],
            'CNT_PAYMENT': ['mean', 'sum'],
        }

        agg_dict = {}
        skipped_cols = []

        for col, agg_funcs in candidate_agg_dict.items():
            if col in prev_app_df.columns:
                agg_dict[col] = agg_funcs
            else:
                skipped_cols.append(col)

        if skipped_cols:
            print(f"Previous Application 聚合时跳过缺失列: {skipped_cols}")

        if not agg_dict:
            print("Previous Application 无可用聚合字段，跳过聚合")
            self.last_summary['previous_application_aggregation'] = {
                'executed': False,
                'output_shape': None,
                'feature_count': 0
            }
            return None

        prev_agg = prev_app_df.groupby('SK_ID_CURR').agg(agg_dict)
        prev_agg.columns = ['PREV_' + '_'.join(col).upper() for col in prev_agg.columns]
        prev_agg.reset_index(inplace=True)

        # 额外特征
        prev_count = prev_app_df.groupby('SK_ID_CURR').size()
        prev_agg['PREV_APP_COUNT'] = prev_agg['SK_ID_CURR'].map(prev_count).fillna(0)

        if 'NAME_CONTRACT_STATUS' in prev_app_df.columns:
            approved_count = prev_app_df[prev_app_df['NAME_CONTRACT_STATUS'] == 'Approved'].groupby('SK_ID_CURR').size()
            refused_count = prev_app_df[prev_app_df['NAME_CONTRACT_STATUS'] == 'Refused'].groupby('SK_ID_CURR').size()

            prev_agg['PREV_APPROVED_COUNT'] = prev_agg['SK_ID_CURR'].map(approved_count).fillna(0)
            prev_agg['PREV_REFUSED_COUNT'] = prev_agg['SK_ID_CURR'].map(refused_count).fillna(0)
        else:
            print("Previous Application 缺少 NAME_CONTRACT_STATUS，批准/拒绝计数用 0 填充")
            prev_agg['PREV_APPROVED_COUNT'] = 0
            prev_agg['PREV_REFUSED_COUNT'] = 0

        prev_agg['PREV_APPROVAL_RATE'] = prev_agg['PREV_APPROVED_COUNT'] / (prev_agg['PREV_APP_COUNT'] + 1)
        prev_agg['PREV_REFUSED_RATE'] = prev_agg['PREV_REFUSED_COUNT'] / (prev_agg['PREV_APP_COUNT'] + 1)

        self.prev_app_agg = prev_agg

        feature_count = prev_agg.shape[1] - 1
        print(f"Previous Application 聚合完成: {feature_count} 个特征")

        self.last_summary['previous_application_aggregation'] = {
            'executed': True,
            'output_shape': prev_agg.shape,
            'feature_count': feature_count
        }

        return prev_agg

    def merge_all_features(
        self,
        app_df: pd.DataFrame,
        bureau_agg: pd.DataFrame = None,
        prev_agg: pd.DataFrame = None,
        cc_agg: pd.DataFrame = None,
        ip_agg: pd.DataFrame = None,
        pos_agg: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        合并所有特征表

        Parameters:
        -----------
        app_df : pd.DataFrame
            Application 主表
        bureau_agg : pd.DataFrame
            Bureau 聚合特征
        prev_agg : pd.DataFrame
            Previous Application 聚合特征
        cc_agg : pd.DataFrame
            Credit Card Balance 聚合特征
        ip_agg : pd.DataFrame
            Installments Payments 聚合特征
        pos_agg : pd.DataFrame
            POS_CASH Balance 聚合特征

        Returns:
        --------
        pd.DataFrame : 合并后的完整特征表
        """
        print("\n合并所有特征...")

        df = app_df.copy()
        shape_before = df.shape
        print(f"合并前主表 shape: {shape_before}")

        agg_tables = [
            ("Bureau", bureau_agg),
            ("Previous Application", prev_agg),
            ("Credit Card Balance", cc_agg),
            ("Installments Payments", ip_agg),
            ("POS_CASH Balance", pos_agg),
        ]

        for name, agg_df in agg_tables:
            if agg_df is not None:
                before_shape = df.shape
                df = df.merge(agg_df, on='SK_ID_CURR', how='left')
                print(f"  合并 {name} 特征: {before_shape} -> {df.shape}")

        print(f"  特征合并完成: {df.shape[1]} 列")

        self.last_summary['merge'] = {
            'input_shape': shape_before,
            'output_shape': df.shape
        }

        return df

    def handle_merged_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理 merge 外部聚合特征后新增的缺失值
        默认对数值型缺失填充 0，因为通常表示没有历史记录
        """
        print("\n处理 merge 后新增缺失值...")

        df = df.copy()
        missing_before = int(df.isnull().sum().sum())

        numeric_missing_cols = [
            col for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].isnull().sum() > 0
        ]

        self.merged_missing_filled_cols = numeric_missing_cols.copy()

        for col in numeric_missing_cols:
            df[col] = df[col].fillna(0)

        missing_after = int(df.isnull().sum().sum())

        print(f"缺失值处理前总缺失: {missing_before:,}")
        print(f"缺失值处理后总缺失: {missing_after:,}")
        print(f"填充为 0 的数值列 ({len(numeric_missing_cols)}): {numeric_missing_cols}")
        print("Merge 后缺失值处理完成")

        self.last_summary['merged_missing_values'] = {
            'missing_before': missing_before,
            'missing_after': missing_after,
            'filled_columns': numeric_missing_cols
        }

        return df

    def print_feature_engineering_summary(self):
        """打印特征工程摘要"""
        summary = self.last_summary

        print("\n" + "=" * 70)
        print("特征工程总结")
        print("=" * 70)

        if 'dataset_name' in summary:
            print(f"数据集: {summary['dataset_name']}")
        if 'input_shape' in summary:
            print(f"输入主表 shape: {summary['input_shape']}")

        if 'basic_features' in summary:
            bf = summary['basic_features']
            print(f"基础衍生特征: {bf.get('created_count', 0)} 个")

        for key, label in [
            ('bureau_aggregation', 'Bureau'),
            ('previous_application_aggregation', 'Previous Application'),
            ('cc_aggregation', 'Credit Card Balance'),
            ('ip_aggregation', 'Installments Payments'),
            ('pos_aggregation', 'POS_CASH Balance'),
        ]:
            if key in summary:
                info = summary[key]
                status = "执行" if info.get('executed') else "跳过"
                count = info.get('feature_count', 0)
                print(f"  {label} 聚合: {status} ({count} 个特征)")

        if 'merge' in summary:
            print(f"Merge 后 shape: {summary['merge'].get('output_shape')}")
        if 'merged_missing_values' in summary:
            mm = summary['merged_missing_values']
            print(f"Merge 后缺失填充: {mm.get('missing_before', 0):,} -> {mm.get('missing_after', 0):,}")
        if 'final_shape' in summary:
            print(f"最终 shape: {summary['final_shape']}")
        if 'report_path' in summary:
            print(f"报告路径: {summary['report_path']}")

        print("=" * 70)

    def save_feature_engineering_report(self, output_path: Path, dataset_name: str = "train") -> None:
        """保存 markdown 特征工程报告"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary = self.last_summary

        basic_features = summary.get('basic_features', {})
        merge_info = summary.get('merge', {})
        merged_missing = summary.get('merged_missing_values', {})

        agg_sections = [
            ('bureau_aggregation', 'Bureau'),
            ('previous_application_aggregation', 'Previous Application'),
            ('cc_aggregation', 'Credit Card Balance'),
            ('ip_aggregation', 'Installments Payments'),
            ('pos_aggregation', 'POS_CASH Balance'),
        ]

        lines = [
            f"# 特征工程报告 - {dataset_name}",
            "",
            "## 1. 数据集概况",
            f"- 数据集名称: {dataset_name}",
            f"- 输入主表 shape: {summary.get('input_shape')}",
            f"- 最终 shape: {summary.get('final_shape')}",
            "",
            "## 2. 基础衍生特征",
            f"- 新增基础特征数量: {basic_features.get('created_count', 0)}",
            f"- 新增基础特征列表: {basic_features.get('created_features', [])}",
            "",
        ]

        section_num = 3
        for key, label in agg_sections:
            info = summary.get(key, {})
            lines.extend([
                f"## {section_num}. {label} 聚合特征",
                f"- 是否执行: {info.get('executed', False)}",
                f"- 输出 shape: {info.get('output_shape')}",
                f"- 新增特征数量: {info.get('feature_count', 0)}",
                "",
            ])
            section_num += 1

        lines.extend([
            f"## {section_num}. 特征合并结果",
            f"- Merge 前主表 shape: {merge_info.get('input_shape')}",
            f"- Merge 后 shape: {merge_info.get('output_shape')}",
            "",
            f"## {section_num + 1}. Merge 后缺失值处理",
            f"- 缺失填充前总缺失: {merged_missing.get('missing_before', 0)}",
            f"- 缺失填充后总缺失: {merged_missing.get('missing_after', 0)}",
            f"- 被填充为 0 的列数量: {len(merged_missing.get('filled_columns', []))}",
            "",
            f"## {section_num + 2}. 最终结果",
            f"- 最终 shape: {summary.get('final_shape')}",
            ""
        ])

        output_path.write_text("\n".join(lines), encoding="utf-8")
        self.last_summary['report_path'] = str(output_path)

    def _remove_correlated_features(self, df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """删除高度相关的特征（保留先出现的）"""
        print(f"\n删除高相关特征 (阈值: {threshold})...")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != TARGET]

        if len(numeric_cols) < 2:
            print("  数值特征过少，跳过")
            return df

        corr_matrix = df[numeric_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

        if to_drop:
            print(f"  删除 {len(to_drop)} 个高相关特征: {to_drop[:10]}{'...' if len(to_drop) > 10 else ''}")
            df = df.drop(columns=to_drop)
        else:
            print("  无高相关特征需要删除")

        print(f"  删除后特征数: {df.shape[1]}")
        return df

    def feature_engineering_pipeline(
        self,
        app_df: pd.DataFrame,
        bureau_df: pd.DataFrame = None,
        prev_app_df: pd.DataFrame = None,
        cc_df: pd.DataFrame = None,
        ip_df: pd.DataFrame = None,
        pos_df: pd.DataFrame = None,
        dataset_name: str = "train"
    ) -> pd.DataFrame:
        """
        完整的特征工程流程

        Parameters:
        -----------
        app_df : pd.DataFrame
            Application 主表（已处理）
        bureau_df : pd.DataFrame
            Bureau 表（可选）
        prev_app_df : pd.DataFrame
            Previous Application 表（可选）
        cc_df : pd.DataFrame
            Credit Card Balance 表（可选）
        ip_df : pd.DataFrame
            Installments Payments 表（可选）
        pos_df : pd.DataFrame
            POS_CASH Balance 表（可选）
        dataset_name : str
            数据集名称（train / test）

        Returns:
        --------
        pd.DataFrame : 完整特征表
        """
        print("\n" + "=" * 60)
        print("开始特征工程流程")
        print("=" * 60)

        self.last_summary = {}
        self.created_basic_features = []
        self.merged_missing_filled_cols = []

        self.last_summary['dataset_name'] = dataset_name
        self.last_summary['input_shape'] = app_df.shape

        # 1. 创建基础特征
        df = self.create_basic_features(app_df)

        # 2. 聚合 Bureau 数据
        if bureau_df is not None:
            bureau_agg = self.aggregate_bureau_data(bureau_df)
        else:
            print("\n未提供 Bureau 数据，跳过 Bureau 聚合")
            bureau_agg = None
            self.last_summary['bureau_aggregation'] = {
                'executed': False, 'output_shape': None, 'feature_count': 0
            }

        # 3. 聚合 Previous Application 数据
        if prev_app_df is not None:
            prev_agg = self.aggregate_previous_application(prev_app_df)
        else:
            print("\n未提供 Previous Application 数据，跳过聚合")
            prev_agg = None
            self.last_summary['previous_application_aggregation'] = {
                'executed': False, 'output_shape': None, 'feature_count': 0
            }

        # 4. 聚合 Credit Card Balance 数据
        if cc_df is not None:
            cc_agg = self.aggregate_credit_card_balance(cc_df)
        else:
            print("\n未提供 Credit Card Balance 数据，跳过聚合")
            cc_agg = None
            self.last_summary['cc_aggregation'] = {
                'executed': False, 'output_shape': None, 'feature_count': 0
            }

        # 5. 聚合 Installments Payments 数据
        if ip_df is not None:
            ip_agg = self.aggregate_installments_payments(ip_df)
        else:
            print("\n未提供 Installments Payments 数据，跳过聚合")
            ip_agg = None
            self.last_summary['ip_aggregation'] = {
                'executed': False, 'output_shape': None, 'feature_count': 0
            }

        # 6. 聚合 POS_CASH Balance 数据
        if pos_df is not None:
            pos_agg = self.aggregate_pos_cash_balance(pos_df)
        else:
            print("\n未提供 POS_CASH Balance 数据，跳过聚合")
            pos_agg = None
            self.last_summary['pos_aggregation'] = {
                'executed': False, 'output_shape': None, 'feature_count': 0
            }

        # 7. 合并所有特征
        df = self.merge_all_features(df, bureau_agg, prev_agg, cc_agg, ip_agg, pos_agg)

        # 8. 处理 merge 后新增缺失值
        df = self.handle_merged_missing_values(df)

        # 9. 删除高相关特征（仅训练集）
        if REMOVE_CORRELATED and dataset_name == "train":
            df = self._remove_correlated_features(df, threshold=CORRELATION_THRESHOLD)

        self.last_summary['final_shape'] = df.shape

        self.print_feature_engineering_summary()

        print("\n" + "=" * 60)
        print(f"特征工程完成: {df.shape[0]:,} 行 x {df.shape[1]:,} 列")
        print("=" * 60)

        return df


def main():
    """主函数 - 特征工程"""
    print("=" * 70)
    print("Home Credit 特征工程")
    print("=" * 70)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_input_path = PROCESSED_DATA_DIR / "train_processed.csv"
    test_input_path = PROCESSED_DATA_DIR / "test_processed.csv"

    # 检查主表处理结果是否存在
    if not train_input_path.exists():
        print(f"未找到处理后的训练集: {train_input_path}")
        print("请先运行 data_processing.py")
        return

    if not test_input_path.exists():
        print(f"未找到处理后的测试集: {test_input_path}")
        print("请先运行 data_processing.py")
        return

    # 加载所有外部辅助表（一次性加载，train/test 共用）
    auxiliary_tables = {
        'bureau': ('bureau.csv', None),
        'prev_app': ('previous_application.csv', None),
        'cc': ('credit_card_balance.csv', None),
        'ip': ('installments_payments.csv', None),
        'pos': ('POS_CASH_balance.csv', None),
    }

    loaded_tables = {}
    for key, (filename, _) in auxiliary_tables.items():
        path = RAW_DATA_DIR / filename
        if path.exists():
            print(f"加载 {filename}...")
            loaded_tables[key] = pd.read_csv(path, engine='c')
            mem_mb = loaded_tables[key].memory_usage(deep=True).sum() / 1e6
            print(f"  {filename}: {loaded_tables[key].shape[0]:,} rows x {loaded_tables[key].shape[1]} cols ({mem_mb:.1f} MB)")
        else:
            print(f"未找到 {filename}，将跳过")

    # 内存优化：float64 → float32
    for key in loaded_tables:
        df = loaded_tables[key]
        float_cols = df.select_dtypes(include=['float64']).columns
        if len(float_cols) > 0:
            before_mb = df.memory_usage(deep=True).sum() / 1e6
            df[float_cols] = df[float_cols].astype('float32')
            after_mb = df.memory_usage(deep=True).sum() / 1e6
            print(f"  {key}: float64→float32, {before_mb:.1f}→{after_mb:.1f} MB")

    # 使用单一 FeatureEngineer 实例处理 train 和 test
    # 先在 train 上 fit 聚合逻辑，再对 test 应用相同的聚合
    engineer = FeatureEngineer()

    # 处理训练集
    print("\n" + "=" * 70)
    print("开始处理训练集特征工程")
    print("=" * 70)

    train_app_df = pd.read_csv(train_input_path)
    print(f"训练集: {train_app_df.shape[0]:,} rows x {train_app_df.shape[1]} cols")

    train_featured = engineer.feature_engineering_pipeline(
        train_app_df,
        bureau_df=loaded_tables.get('bureau'),
        prev_app_df=loaded_tables.get('prev_app'),
        cc_df=loaded_tables.get('cc'),
        ip_df=loaded_tables.get('ip'),
        pos_df=loaded_tables.get('pos'),
        dataset_name="train"
    )

    train_output_path = PROCESSED_DATA_DIR / "train_with_features.csv"
    train_featured.to_csv(train_output_path, index=False)
    print(f"训练数据已保存至: {train_output_path}")

    train_report_path = PROCESSED_DATA_DIR / "train_feature_engineering_report.md"
    engineer.save_feature_engineering_report(train_report_path, dataset_name="train")

    # 处理测试集（复用同一个 engineer 实例）
    print("\n" + "=" * 70)
    print("开始处理测试集特征工程")
    print("=" * 70)

    test_app_df = pd.read_csv(test_input_path)
    print(f"测试集: {test_app_df.shape[0]:,} rows x {test_app_df.shape[1]} cols")

    test_engineer = FeatureEngineer()
    test_featured = test_engineer.feature_engineering_pipeline(
        test_app_df,
        bureau_df=loaded_tables.get('bureau'),
        prev_app_df=loaded_tables.get('prev_app'),
        cc_df=loaded_tables.get('cc'),
        ip_df=loaded_tables.get('ip'),
        pos_df=loaded_tables.get('pos'),
        dataset_name="test"
    )

    test_output_path = PROCESSED_DATA_DIR / "test_with_features.csv"
    test_featured.to_csv(test_output_path, index=False)
    print(f"测试数据已保存至: {test_output_path}")

    test_report_path = PROCESSED_DATA_DIR / "test_feature_engineering_report.md"
    test_engineer.save_feature_engineering_report(test_report_path, dataset_name="test")


if __name__ == "__main__":
    setup_logging()
    main()
