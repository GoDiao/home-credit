"""
数据处理模块
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


class DataProcessor:
    """数据处理类"""

    def __init__(self):
        self.train_df = None
        self.test_df = None
        self.last_summary: Dict[str, Any] = {}

        # 保存训练阶段编码信息，保证测试集与训练集一致
        self.encoding_strategy: Dict[str, str] = {}
        self.onehot_columns: Dict[str, List[str]] = {}
        self.frequency_maps: Dict[str, Dict[Any, float]] = {}
        self.target_maps: Dict[str, Dict[Any, float]] = {}
        self.label_maps: Dict[str, Dict[Any, int]] = {}
        self.train_feature_columns: List[str] = []

    def load_data(self, file_name: str = "application_train.csv") -> pd.DataFrame:
        """加载数据"""
        file_path = RAW_DATA_DIR / file_name
        print(f"📂 加载数据: {file_path}")

        df = pd.read_csv(file_path)
        print(f"数据加载完成: {df.shape[0]:,} 行 × {df.shape[1]:,} 列")

        return df

    def basic_info(self, df: pd.DataFrame, name: str = "Dataset"):
        """显示数据基本信息"""
        print("\n" + "=" * 60)
        print(f"{name} - 基本信息")
        print("=" * 60)
        print(f"样本数量: {df.shape[0]:,}")
        print(f"特征数量: {df.shape[1]:,}")

        if TARGET in df.columns:
            print(f"\n目标变量分布:")
            print(df[TARGET].value_counts())
            print(f"\n违约率: {df[TARGET].mean():.2%}")

        print(f"\n缺失值统计:")
        missing = df.isnull().sum()
        missing_pct = missing / len(df) * 100
        missing_df = pd.DataFrame({
            '缺失数量': missing,
            '缺失比例': missing_pct
        })
        missing_df = missing_df[missing_df['缺失数量'] > 0].sort_values('缺失比例', ascending=False)

        if len(missing_df) > 0:
            print(missing_df.head(10))
        else:
            print("无缺失值")

        print("=" * 60)

    def apply_business_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用业务规则清洗

        规则:
        - DAYS_EMPLOYED == 365243 视为退休人员占位值，替换为 np.nan 并标记 IS_RETIRED
        """
        print("\n应用业务规则清洗")
        df = df.copy()

        affected_rows = 0
        if 'DAYS_EMPLOYED' in df.columns:
            affected_rows = int((df['DAYS_EMPLOYED'] == 365243).sum())
            df['IS_RETIRED'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
            df.loc[df['DAYS_EMPLOYED'] == 365243, 'DAYS_EMPLOYED'] = np.nan
            print(f"  DAYS_EMPLOYED == 365243 -> np.nan, IS_RETIRED=1, 影响 {affected_rows:,} 行")
        else:
            df['IS_RETIRED'] = 0

        print("业务规则清洗完成")

        self.last_summary['business_rules'] = {
            'days_employed_placeholder_fixed': affected_rows
        }
        return df

    def handle_missing_values(self, df: pd.DataFrame, threshold: float = MISSING_THRESHOLD) -> pd.DataFrame:
        """
        处理缺失值

        策略:
        1. 删除缺失率超过阈值的列
        2. 数值列: 优先按 INCOME_TYPE 分组中位数填充，无分组列时用全局中位数
        3. 类别列: 众数填充
        """
        print(f"\n处理缺失值 (阈值: {threshold:.0%})")

        missing_before = int(df.isnull().sum().sum())

        # 计算缺失率
        missing_pct = df.isnull().sum() / len(df)

        # 删除缺失率过高的列
        cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
        if cols_to_drop:
            print(f"删除 {len(cols_to_drop)} 个缺失率过高的列: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)
        else:
            print("无缺失率过高列需要删除")

        # 确定分组列（用于分组填充）
        group_col = None
        if 'NAME_INCOME_TYPE' in df.columns:
            group_col = 'NAME_INCOME_TYPE'
            print(f"使用 {group_col} 进行分组中位数填充")

        # 对剩余列进行填充
        median_filled_cols = []
        mode_filled_cols = []
        grouped_fill_cols = []

        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(df[col]):
                    if group_col and group_col in df.columns:
                        # 分组中位数填充
                        group_medians = df.groupby(group_col)[col].transform('median')
                        filled_by_group = df[col].isnull() & group_medians.notna()
                        df[col] = df[col].fillna(group_medians)
                        if filled_by_group.sum() > 0:
                            grouped_fill_cols.append(col)
                    # 全局中位数兜底
                    if df[col].isnull().sum() > 0:
                        median_value = df[col].median()
                        df[col] = df[col].fillna(median_value)
                    median_filled_cols.append(col)
                else:
                    mode_series = df[col].mode()
                    mode_value = mode_series.iloc[0] if len(mode_series) > 0 else 'MISSING'
                    df[col] = df[col].fillna(mode_value)
                    mode_filled_cols.append(col)

        if grouped_fill_cols:
            print(f"分组中位数填充 ({len(grouped_fill_cols)}): {grouped_fill_cols}")
        print(f"数值列填充 ({len(median_filled_cols)})")
        print(f"类别列众数填充 ({len(mode_filled_cols)})")

        remaining_nulls = int(df.isnull().sum().sum())
        if remaining_nulls > 0:
            print(f"仍有 {remaining_nulls} 个 NaN 值，使用 0 填充")
            df = df.fillna(0)

        missing_after = int(df.isnull().sum().sum())

        print(f"缺失值处理前总缺失: {missing_before:,}")
        print(f"缺失值处理后总缺失: {missing_after:,}")
        print("缺失值处理完成")

        self.last_summary['missing_values'] = {
            'missing_before': missing_before,
            'missing_after': missing_after,
            'dropped_high_missing_cols': cols_to_drop,
            'median_filled_cols': median_filled_cols,
            'mode_filled_cols': mode_filled_cols
        }
        return df

    def encode_categorical(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        编码分类变量，并保证训练集和测试集编码一致
        """
        print(f"\n编码分类变量")

        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        shape_before = df.shape

        label_encoded_cols = []
        onehot_encoded_cols = []
        frequency_encoded_cols = []
        target_encoded_cols = []

        if is_train:
            if not cat_cols:
                print("无分类变量需要编码")
                self.last_summary['encoding'] = {
                    'label_encoded_cols': [],
                    'onehot_encoded_cols': [],
                    'frequency_encoded_cols': [],
                    'shape_before': df.shape,
                    'shape_after': df.shape,
                    'column_alignment': 'train_reference_saved'
                }
                self.train_feature_columns = [col for col in df.columns if col != TARGET]
                return df

            print(f"发现 {len(cat_cols)} 个分类变量")

            self.encoding_strategy = {}
            self.onehot_columns = {}
            self.frequency_maps = {}
            self.label_maps = {}

            for col in cat_cols:
                unique_values = df[col].nunique(dropna=False)

                if unique_values == 2:
                    self.encoding_strategy[col] = 'label'
                    categories = sorted(df[col].astype(str).dropna().unique().tolist())
                    label_map = {cat: idx for idx, cat in enumerate(categories)}
                    self.label_maps[col] = label_map

                    df[col] = df[col].astype(str).map(label_map).fillna(-1).astype(int)
                    label_encoded_cols.append(col)

                elif unique_values <= 10:
                    self.encoding_strategy[col] = 'onehot'
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    dummy_cols = dummies.columns.tolist()
                    self.onehot_columns[col] = dummy_cols

                    df = pd.concat([df, dummies], axis=1)
                    df = df.drop(columns=[col])
                    onehot_encoded_cols.append(col)

                else:
                    if USE_TARGET_ENCODING and TARGET in df.columns:
                        self.encoding_strategy[col] = 'target'
                        target_map = df.groupby(col)[TARGET].mean().to_dict()
                        self.target_maps[col] = target_map
                        df[col] = df[col].map(target_map).fillna(df[TARGET].mean())
                        target_encoded_cols.append(col)
                    else:
                        self.encoding_strategy[col] = 'frequency'
                        freq_map = df[col].value_counts(normalize=True).to_dict()
                        self.frequency_maps[col] = freq_map
                        df[col] = df[col].map(freq_map).fillna(0)
                        frequency_encoded_cols.append(col)

            self.train_feature_columns = [col for col in df.columns if col != TARGET]
            alignment_status = 'train_reference_saved'

        else:
            print(f"发现 {len(cat_cols)} 个分类变量（测试集将复用训练集编码规则）")

            train_cat_cols = list(self.encoding_strategy.keys())

            for col in train_cat_cols:
                if col not in df.columns:
                    print(f"测试集缺少训练集分类列: {col}，将跳过原始列处理并在后续对齐补列")
                    continue

                strategy = self.encoding_strategy[col]

                if strategy == 'label':
                    label_map = self.label_maps.get(col, {})
                    df[col] = df[col].astype(str).map(label_map).fillna(-1).astype(int)
                    label_encoded_cols.append(col)

                elif strategy == 'onehot':
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    train_dummy_cols = self.onehot_columns.get(col, [])

                    for dummy_col in train_dummy_cols:
                        if dummy_col not in dummies.columns:
                            dummies[dummy_col] = 0

                    extra_cols = [c for c in dummies.columns if c not in train_dummy_cols]
                    if extra_cols:
                        dummies = dummies.drop(columns=extra_cols)

                    dummies = dummies.reindex(columns=train_dummy_cols, fill_value=0)

                    df = pd.concat([df, dummies], axis=1)
                    df = df.drop(columns=[col])
                    onehot_encoded_cols.append(col)

                elif strategy == 'frequency':
                    freq_map = self.frequency_maps.get(col, {})
                    df[col] = df[col].map(freq_map).fillna(0)
                    frequency_encoded_cols.append(col)

                elif strategy == 'target':
                    target_map = self.target_maps.get(col, {})
                    global_mean = sum(target_map.values()) / len(target_map) if target_map else 0
                    df[col] = df[col].map(target_map).fillna(global_mean)
                    target_encoded_cols.append(col)

            # 测试集中若还有未处理 object 列，统一按训练集无此列处理：删除
            remaining_object_cols = df.select_dtypes(include=['object']).columns.tolist()
            if remaining_object_cols:
                print(f"测试集中存在训练集中未记录的分类列，已删除: {remaining_object_cols}")
                df = df.drop(columns=remaining_object_cols)

            # 按训练集特征对齐
            for col in self.train_feature_columns:
                if col not in df.columns:
                    df[col] = 0

            extra_feature_cols = [col for col in df.columns if col != TARGET and col not in self.train_feature_columns]
            if extra_feature_cols:
                print(f"测试集存在训练集没有的额外特征，已删除: {extra_feature_cols}")
                df = df.drop(columns=extra_feature_cols)

            ordered_cols = self.train_feature_columns.copy()
            if TARGET in df.columns:
                ordered_cols = [TARGET] + ordered_cols
            df = df.reindex(columns=ordered_cols, fill_value=0)

            test_features = [col for col in df.columns if col != TARGET]
            alignment_status = (test_features == self.train_feature_columns)

        shape_after = df.shape

        print(f"Label Encoding ({len(label_encoded_cols)}): {label_encoded_cols}")
        print(f"One-Hot Encoding ({len(onehot_encoded_cols)}): {onehot_encoded_cols}")
        print(f"Frequency Encoding ({len(frequency_encoded_cols)}): {frequency_encoded_cols}")
        if target_encoded_cols:
            print(f"Target Encoding ({len(target_encoded_cols)}): {target_encoded_cols}")
        print(f"编码前 shape: {shape_before}")
        print(f"编码后 shape: {shape_after}")

        if not is_train:
            print(f"训练/测试特征列是否对齐: {alignment_status}")

        print("分类变量编码完成")

        self.last_summary['encoding'] = {
            'label_encoded_cols': label_encoded_cols,
            'onehot_encoded_cols': onehot_encoded_cols,
            'frequency_encoded_cols': frequency_encoded_cols,
            'target_encoded_cols': target_encoded_cols,
            'shape_before': shape_before,
            'shape_after': shape_after,
            'column_alignment': alignment_status
        }
        return df

    def handle_outliers(
        self,
        df: pd.DataFrame,
        columns: List[str] = None,
        method: str = 'iqr',
        factor: float = 3.0
    ) -> pd.DataFrame:
        """
        处理异常值，并对 IQR=0 / std=0 的情况进行保护
        """
        print(f"\n处理异常值 (方法: {method})")

        if columns is None:
            key_continuous_cols = [
                'AMT_INCOME_TOTAL',
                'AMT_CREDIT',
                'AMT_ANNUITY',
                'AMT_GOODS_PRICE',
                'DAYS_EMPLOYED',
                'DAYS_BIRTH',
                'OWN_CAR_AGE',
            ]
            columns = [col for col in key_continuous_cols if col in df.columns]

        if not columns:
            print("无可处理的关键连续变量")
            self.last_summary['outliers'] = {
                'processed_columns': [],
                'skipped_columns': {},
                'details': {}
            }
            return df

        print(f"处理列 ({len(columns)}): {columns}")

        outlier_details = {}
        skipped_columns = {}

        for col in columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                skipped_columns[col] = '非数值列，跳过'
                print(f"{col}: 非数值列，跳过异常值处理")
                continue

            if method == 'iqr':
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1

                if pd.isna(iqr) or iqr == 0:
                    skipped_columns[col] = 'IQR=0 或 NaN，跳过异常值处理'
                    print(f"{col}: IQR=0 或 NaN，跳过异常值处理")
                    continue

                lower_bound = q1 - factor * iqr
                upper_bound = q3 + factor * iqr

            elif method == 'zscore':
                mean = df[col].mean()
                std = df[col].std()

                if pd.isna(std) or std == 0:
                    skipped_columns[col] = 'std=0 或 NaN，跳过异常值处理'
                    print(f"{col}: std=0 或 NaN，跳过异常值处理")
                    continue

                lower_bound = mean - factor * std
                upper_bound = mean + factor * std
            else:
                raise ValueError(f"未知方法: {method}")

            original = df[col].copy()
            lower_clipped_count = int((original < lower_bound).sum())
            upper_clipped_count = int((original > upper_bound).sum())

            df[col] = original.clip(lower=lower_bound, upper=upper_bound)

            print(
                f"{col}: lower={lower_bound:.4f}, upper={upper_bound:.4f}, "
                f"下界截断={lower_clipped_count:,}, 上界截断={upper_clipped_count:,}"
            )

            outlier_details[col] = {
                'method': method,
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'lower_clipped_count': lower_clipped_count,
                'upper_clipped_count': upper_clipped_count
            }

        print("异常值处理完成")

        self.last_summary['outliers'] = {
            'processed_columns': list(outlier_details.keys()),
            'skipped_columns': skipped_columns,
            'details': outlier_details
        }
        return df

    def remove_correlated_features(
        self,
        df: pd.DataFrame,
        threshold: float = CORRELATION_THRESHOLD
    ) -> pd.DataFrame:
        """
        删除高度相关的特征
        """
        print(f"\n删除高度相关特征 (阈值: {threshold})")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != TARGET]

        if len(numeric_cols) < 2:
            print("数值特征过少，无需计算相关性")
            self.last_summary['correlated_features'] = {
                'dropped_columns': []
            }
            return df

        corr_matrix = df[numeric_cols].corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > threshold)]

        if to_drop:
            print(f"删除 {len(to_drop)} 个高度相关特征")
            print(f"删除列: {to_drop}")
            df = df.drop(columns=to_drop)
        else:
            print("无高度相关特征需要删除")

        print("特征相关性处理完成")

        self.last_summary['correlated_features'] = {
            'dropped_columns': to_drop
        }
        return df

    def print_processing_summary(self):
        """打印最终处理摘要"""
        summary = self.last_summary

        print("\n" + "=" * 70)
        print("📋 数据处理总结")
        print("=" * 70)

        if 'dataset_name' in summary:
            print(f"数据集: {summary['dataset_name']}")

        if 'original_shape' in summary:
            print(f"原始 shape: {summary['original_shape']}")

        if 'business_rules' in summary:
            affected = summary['business_rules'].get('days_employed_placeholder_fixed', 0)
            print(f"业务规则修正: DAYS_EMPLOYED 占位值修正 {affected:,} 行")

        if 'missing_values' in summary:
            mv = summary['missing_values']
            print(f"删除高缺失列: {len(mv.get('dropped_high_missing_cols', []))} 个")
            print(f"中位数填补列: {len(mv.get('median_filled_cols', []))} 个")
            print(f"众数填补列: {len(mv.get('mode_filled_cols', []))} 个")
            print(f"缺失值处理前总缺失: {mv.get('missing_before', 0):,}")
            print(f"缺失值处理后总缺失: {mv.get('missing_after', 0):,}")

        if 'encoding' in summary:
            enc = summary['encoding']
            print(f"Label Encoding: {len(enc.get('label_encoded_cols', []))} 列")
            print(f"One-Hot Encoding: {len(enc.get('onehot_encoded_cols', []))} 列")
            print(f"Frequency Encoding: {len(enc.get('frequency_encoded_cols', []))} 列")
            print(f"训练/测试特征列是否对齐: {enc.get('column_alignment')}")

        if 'outliers' in summary:
            out = summary['outliers']
            print(f"异常值实际处理列数: {len(out.get('processed_columns', []))} 列")
            skipped = out.get('skipped_columns', {})
            if skipped:
                print(f"跳过异常值处理列数: {len(skipped)} 列")
                for col, reason in skipped.items():
                    print(f"  - {col}: {reason}")

        if 'remove_correlated' in summary:
            print(f"是否执行高相关特征删除: {summary['remove_correlated']}")

        if 'correlated_features' in summary:
            dropped = summary['correlated_features'].get('dropped_columns', [])
            print(f"高相关删除列数: {len(dropped)}")

        if 'final_shape' in summary:
            print(f"最终 shape: {summary['final_shape']}")

        if 'final_missing' in summary:
            print(f"最终剩余缺失值: {summary['final_missing']:,}")

        if 'report_path' in summary:
            print(f"Markdown 报告路径: {summary['report_path']}")

        print("=" * 70)

    def save_processing_report(self, output_path: Path, dataset_name: str = "train") -> None:
        """保存 markdown 处理报告"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary = self.last_summary

        business_rules = summary.get('business_rules', {})
        missing_values = summary.get('missing_values', {})
        encoding = summary.get('encoding', {})
        outliers = summary.get('outliers', {})
        correlated = summary.get('correlated_features', {})

        lines = [
            f"# 数据处理报告 - {dataset_name}",
            "",
            "## 1. 数据集概况",
            f"- 数据集名称: {dataset_name}",
            f"- 原始 shape: {summary.get('original_shape')}",
            f"- 最终 shape: {summary.get('final_shape')}",
            f"- 最终剩余缺失值: {summary.get('final_missing', 0)}",
            "",
            "## 2. 业务规则清洗",
            f"- DAYS_EMPLOYED 占位值修正行数: {business_rules.get('days_employed_placeholder_fixed', 0)}",
            "",
            "## 3. 缺失值处理",
            f"- 删除高缺失列数量: {len(missing_values.get('dropped_high_missing_cols', []))}",
            f"- 删除高缺失列: {missing_values.get('dropped_high_missing_cols', [])}",
            f"- 数值列中位数填充数量: {len(missing_values.get('median_filled_cols', []))}",
            f"- 数值列中位数填充列: {missing_values.get('median_filled_cols', [])}",
            f"- 类别列众数填充数量: {len(missing_values.get('mode_filled_cols', []))}",
            f"- 类别列众数填充列: {missing_values.get('mode_filled_cols', [])}",
            f"- 缺失值处理前总缺失: {missing_values.get('missing_before', 0)}",
            f"- 缺失值处理后总缺失: {missing_values.get('missing_after', 0)}",
            "",
            "## 4. 分类变量编码",
            f"- Label Encoding 数量: {len(encoding.get('label_encoded_cols', []))}",
            f"- Label Encoding 列: {encoding.get('label_encoded_cols', [])}",
            f"- One-Hot Encoding 数量: {len(encoding.get('onehot_encoded_cols', []))}",
            f"- One-Hot Encoding 列: {encoding.get('onehot_encoded_cols', [])}",
            f"- Frequency Encoding 数量: {len(encoding.get('frequency_encoded_cols', []))}",
            f"- Frequency Encoding 列: {encoding.get('frequency_encoded_cols', [])}",
            f"- 编码前 shape: {encoding.get('shape_before')}",
            f"- 编码后 shape: {encoding.get('shape_after')}",
            f"- 训练/测试特征列是否对齐: {encoding.get('column_alignment')}",
            "",
            "## 5. 异常值处理",
            f"- 实际处理列: {outliers.get('processed_columns', [])}",
            ""
        ]

        details = outliers.get('details', {})
        if details:
            lines.append("### 5.1 各列异常值处理明细")
            lines.append("")
            for col, info in details.items():
                lines.extend([
                    f"#### {col}",
                    f"- 方法: {info.get('method')}",
                    f"- 下界: {info.get('lower_bound')}",
                    f"- 上界: {info.get('upper_bound')}",
                    f"- 下界截断数量: {info.get('lower_clipped_count')}",
                    f"- 上界截断数量: {info.get('upper_clipped_count')}",
                    ""
                ])

        skipped = outliers.get('skipped_columns', {})
        lines.extend([
            "### 5.2 跳过异常值处理的列",
            ""
        ])
        if skipped:
            for col, reason in skipped.items():
                lines.append(f"- {col}: {reason}")
        else:
            lines.append("- 无")

        lines.extend([
            "",
            "## 6. 高相关特征处理",
            f"- 是否执行高相关特征删除: {summary.get('remove_correlated')}",
            f"- 删除的高相关特征: {correlated.get('dropped_columns', [])}",
            "",
            "## 7. 最终结论",
            f"- 最终 shape: {summary.get('final_shape')}",
            f"- 最终剩余缺失值: {summary.get('final_missing', 0)}",
            ""
        ])

        output_path.write_text("\n".join(lines), encoding="utf-8")
        self.last_summary['report_path'] = str(output_path)

    def process_pipeline(
        self,
        df: pd.DataFrame,
        is_train: bool = True,
        remove_correlated: bool = False,
        dataset_name: str = "train"
    ) -> pd.DataFrame:
        """
        完整的数据处理流程
        """
        print("\n" + "=" * 70)
        print("开始数据处理 Pipeline")
        print("=" * 70)

        self.last_summary = {}
        self.last_summary['dataset_name'] = dataset_name
        self.last_summary['original_shape'] = df.shape

        print("\nStep 1/5: 基本信息检查")
        self.basic_info(df, "输入数据")

        print("\nStep 2/5: 业务规则清洗")
        df = self.apply_business_rules(df)

        print("\nStep 3/5: 缺失值处理")
        df = self.handle_missing_values(df)

        print("\nStep 4/5: 分类变量编码")
        df = self.encode_categorical(df, is_train=is_train)

        print("\nStep 5/5: 异常值处理")
        df = self.handle_outliers(df)

        if remove_correlated and is_train:
            print("\n附加步骤: 删除高相关特征")
            df = self.remove_correlated_features(df)
        else:
            print("\n附加步骤: 跳过高相关特征删除（默认关闭，适合树模型）")
            self.last_summary['correlated_features'] = {
                'dropped_columns': []
            }

        self.last_summary['remove_correlated'] = remove_correlated and is_train
        self.last_summary['final_shape'] = df.shape
        self.last_summary['final_missing'] = int(df.isnull().sum().sum())

        self.print_processing_summary()

        print("\n数据处理 Pipeline 完成")
        return df


def main():
    """主函数"""
    print("=" * 70)
    print("🏦 Home Credit 数据处理")
    print("=" * 70)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    processor = DataProcessor()

    # 处理训练集
    train_df = processor.load_data("application_train.csv")
    train_processed = processor.process_pipeline(
        train_df,
        is_train=True,
        remove_correlated=False,
        dataset_name="train"
    )

    train_output_path = PROCESSED_DATA_DIR / "train_processed.csv"
    train_processed.to_csv(train_output_path, index=False)
    print(f"\n处理后的训练数据已保存至: {train_output_path}")

    train_report_path = PROCESSED_DATA_DIR / "train_processing_report.md"
    processor.save_processing_report(train_report_path, dataset_name="train")
    processor.print_processing_summary()

    # 处理测试集
    test_file = RAW_DATA_DIR / "application_test.csv"
    if test_file.exists():
        print("\n" + "=" * 70)
        print("开始处理测试集")
        print("=" * 70)

        test_df = processor.load_data("application_test.csv")
        test_processed = processor.process_pipeline(
            test_df,
            is_train=False,
            remove_correlated=False,
            dataset_name="test"
        )

        test_output_path = PROCESSED_DATA_DIR / "test_processed.csv"
        test_processed.to_csv(test_output_path, index=False)
        print(f"\n处理后的测试数据已保存至: {test_output_path}")

        test_report_path = PROCESSED_DATA_DIR / "test_processing_report.md"
        processor.save_processing_report(test_report_path, dataset_name="test")
        processor.print_processing_summary()


if __name__ == "__main__":
    main()
