"""
工具函数模块
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

def calculate_ks(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    计算 KS 统计量 (Kolmogorov-Smirnov)
    
    Parameters:
    -----------
    y_true : array-like
        真实标签
    y_pred_proba : array-like
        预测概率
    
    Returns:
    --------
    float : KS 值
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    ks = max(tpr - fpr)
    return ks

def calculate_gini(auc: float) -> float:
    """
    计算 Gini 系数
    
    Parameters:
    -----------
    auc : float
        AUC 值
    
    Returns:
    --------
    float : Gini 系数
    """
    return 2 * auc - 1

def calculate_all_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                         threshold: float = 0.5) -> Dict[str, float]:
    """
    计算所有评估指标
    
    Parameters:
    -----------
    y_true : array-like
        真实标签
    y_pred_proba : array-like
        预测概率
    threshold : float
        分类阈值
    
    Returns:
    --------
    dict : 包含所有指标的字典
    """
    # 二分类预测
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # 计算指标
    auc = roc_auc_score(y_true, y_pred_proba)
    ks = calculate_ks(y_true, y_pred_proba)
    gini = calculate_gini(auc)
    
    # 混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # 准确率、精确率、召回率
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "AUC": auc,
        "KS": ks,
        "Gini": gini,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }

def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                   model_name: str = "Model", save_path: str = None):
    """
    绘制 ROC 曲线
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5000)')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_ks_curve(y_true: np.ndarray, y_pred_proba: np.ndarray,
                  model_name: str = "Model", save_path: str = None):
    """
    绘制 KS 曲线
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    ks_values = tpr - fpr
    ks = max(ks_values)
    ks_threshold = thresholds[np.argmax(ks_values)]
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, tpr, label='True Positive Rate', linewidth=2)
    plt.plot(thresholds, fpr, label='False Positive Rate', linewidth=2)
    plt.plot(thresholds, ks_values, label=f'KS (max = {ks:.4f} at {ks_threshold:.4f})', 
             linewidth=2, linestyle='--')
    
    # 标记最大 KS 点
    plt.scatter([ks_threshold], [ks], color='red', s=100, zorder=5)
    
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Rate', fontsize=12)
    plt.title(f'KS Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         labels: List[str] = None, save_path: str = None):
    """
    绘制混淆矩阵
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if labels is None:
        labels = ['Non-Default', 'Default']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_importance(feature_names: List[str], importances: np.ndarray,
                           top_n: int = 20, save_path: str = None):
    """
    绘制特征重要性
    """
    # 排序
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importances[indices], align='center')
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_gini_curve(y_true: np.ndarray, y_pred_proba: np.ndarray,
                    model_name: str = "Model", save_path: str = None):
    """
    绘制 Gini / Lorenz 曲线
    X 轴: 按预测违约概率排序的客户累积占比
    Y 轴: 对应的实际违约累积占比
    对角线 = 随机模型, 曲线越靠左上 = 模型越好
    """
    sorted_idx = np.argsort(y_pred_proba)[::-1]
    y_sorted = y_true[sorted_idx]

    n = len(y_true)
    x_cum = np.arange(1, n + 1) / n
    y_cum = np.cumsum(y_sorted) / y_sorted.sum()

    auc = roc_auc_score(y_true, y_pred_proba)
    gini = 2 * auc - 1

    plt.figure(figsize=(8, 6))
    plt.plot(x_cum, y_cum, label=f'{model_name} (Gini = {gini:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random (Gini = 0)')
    plt.xlabel('累积客户占比 (按预测风险排序)', fontsize=12)
    plt.ylabel('累积违约占比', fontsize=12)
    plt.title('Gini / Lorenz 曲线', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def calculate_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """
    计算 PSI (Population Stability Index)
    用于监控特征分布的稳定性

    Parameters:
    -----------
    expected : pd.Series
        基准期数据
    actual : pd.Series
        当前期数据
    bins : int
        分箱数量

    Returns:
    --------
    float : PSI 值
    """
    # 零方差保护
    if expected.std() == 0 and actual.std() == 0:
        return 0.0

    # 分箱
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) < 2:
        return 0.0

    expected_percents = pd.cut(expected, bins=breakpoints, include_lowest=True).value_counts(normalize=True)
    actual_percents = pd.cut(actual, bins=breakpoints, include_lowest=True).value_counts(normalize=True)

    # 对齐索引
    expected_percents, actual_percents = expected_percents.align(actual_percents, fill_value=0.0001)

    # 避免 log(0)
    expected_percents = expected_percents.clip(lower=0.0001)
    actual_percents = actual_percents.clip(lower=0.0001)

    # 计算 PSI
    psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))

    return float(psi)

def calculate_woe_iv(df: pd.DataFrame, feature: str, target: str, bins: int = 10) -> Tuple[pd.DataFrame, float]:
    """
    计算 WOE (Weight of Evidence) 和 IV (Information Value)
    
    Parameters:
    -----------
    df : pd.DataFrame
        数据集
    feature : str
        特征名
    target : str
        目标变量名
    bins : int
        分箱数量
    
    Returns:
    --------
    tuple : (WOE 表, IV 值)
    """
    # 分箱
    df_copy = df[[feature, target]].copy()
    df_copy['bin'] = pd.qcut(df_copy[feature], q=bins, duplicates='drop')
    
    # 计算每个箱的好坏客户数
    grouped = df_copy.groupby('bin')[target].agg(['count', 'sum'])
    grouped.columns = ['total', 'bad']
    grouped['good'] = grouped['total'] - grouped['bad']
    
    # 计算分布
    grouped['bad_dist'] = grouped['bad'] / grouped['bad'].sum()
    grouped['good_dist'] = grouped['good'] / grouped['good'].sum()
    
    # 避免除零
    grouped['bad_dist'] = grouped['bad_dist'].replace(0, 0.0001)
    grouped['good_dist'] = grouped['good_dist'].replace(0, 0.0001)
    
    # 计算 WOE 和 IV
    grouped['woe'] = np.log(grouped['good_dist'] / grouped['bad_dist'])
    grouped['iv'] = (grouped['good_dist'] - grouped['bad_dist']) * grouped['woe']
    
    iv = grouped['iv'].sum()
    
    return grouped, iv

def print_metrics_report(metrics: Dict[str, float], model_name: str = "Model"):
    """
    打印模型评估报告
    """
    print("=" * 60)
    print(f"{model_name} - 评估报告")
    print("=" * 60)
    print(f"AUC:       {metrics['AUC']:.4f}")
    print(f"KS:        {metrics['KS']:.4f}")
    print(f"Gini:      {metrics['Gini']:.4f}")
    print(f"Accuracy:  {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall:    {metrics['Recall']:.4f}")
    print(f"F1-Score:  {metrics['F1-Score']:.4f}")
    print("-" * 60)
    print("混淆矩阵:")
    print(f"  TP: {metrics['TP']:>6}  |  FP: {metrics['FP']:>6}")
    print(f"  FN: {metrics['FN']:>6}  |  TN: {metrics['TN']:>6}")
    print("=" * 60)
