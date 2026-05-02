"""
PD 模型模块 - 违约概率预测
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import joblib
import pickle
import logging
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
import xgboost as xgb

from config import *
from utils import *
from model_registry import register_model


class PDModel:
    """PD 模型类"""

    def __init__(self, model_type: str = 'logistic'):
        """
        初始化 PD 模型

        Parameters:
        -----------
        model_type : str
            模型类型：'logistic' 或 'xgboost'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.metrics = {}

        # 新增：记录过程信息
        self.last_summary = {}
        self.prediction_output_path = None
        self.report_path = None
        self.model_path = None
        self.plot_paths = {}

    def prepare_data(self, df: pd.DataFrame, test_size: float = TEST_SIZE) -> Tuple:
        """
        准备训练数据

        Parameters:
        -----------
        df : pd.DataFrame
            完整数据集
        test_size : float
            测试集比例

        Returns:
        --------
        tuple : (X_train, X_test, y_train, y_test)
        """
        print("\n准备数据...")

        self.last_summary = {
            'model_type': self.model_type,
            'dataset_name': 'train'
        }

        # 分离特征和目标
        X = df.drop(columns=[TARGET] + EXCLUDE_FEATURES, errors='ignore')
        y = df[TARGET]

        # 只保留数值列
        X = X.select_dtypes(include=[np.number])

        self.feature_names = X.columns.tolist()

        total_samples = len(X)
        feature_count = len(self.feature_names)
        positive_count = int((y == 1).sum())
        negative_count = int((y == 0).sum())
        positive_ratio = float((y == 1).mean())
        negative_ratio = float((y == 0).mean())
        default_rate = float(y.mean())

        print(f"特征数量: {feature_count}")
        print(f"样本数量: {total_samples:,}")
        print(f"正样本数: {positive_count:,}")
        print(f"负样本数: {negative_count:,}")
        print(f"违约率: {default_rate:.2%}")

        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
        )

        print(f"\n训练集: {len(X_train):,} ({len(X_train)/len(X):.1%})")
        print(f"测试集: {len(X_test):,} ({len(X_test)/len(X):.1%})")

        self.last_summary.update({
            'total_samples': total_samples,
            'feature_count': feature_count,
            'feature_names': self.feature_names,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'positive_ratio': positive_ratio,
            'negative_ratio': negative_ratio,
            'default_rate': default_rate,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
        })

        return X_train, X_test, y_train, y_test

    def align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        将输入特征对齐到训练时的 feature_names

        Parameters:
        -----------
        X : pd.DataFrame
            待对齐特征数据

        Returns:
        --------
        pd.DataFrame : 对齐后的特征数据
        """
        print("\n对齐特征列...")

        if self.feature_names is None:
            raise ValueError("feature_names 为空，请先调用 prepare_data() 或加载已训练模型。")

        X = X.copy()
        original_cols = X.columns.tolist()

        missing_cols = [col for col in self.feature_names if col not in X.columns]
        extra_cols = [col for col in X.columns if col not in self.feature_names]

        for col in missing_cols:
            X[col] = 0

        if extra_cols:
            X = X.drop(columns=extra_cols, errors='ignore')

        X = X[self.feature_names]

        print(f"原始列数: {len(original_cols)}")
        print(f"缺失补齐列数: {len(missing_cols)}")
        print(f"删除多余列数: {len(extra_cols)}")
        print(f"对齐后列数: {X.shape[1]}")

        self.last_summary['feature_alignment'] = {
            'original_column_count': len(original_cols),
            'missing_columns_filled': missing_cols,
            'extra_columns_removed': extra_cols,
            'aligned_column_count': X.shape[1]
        }

        return X

    def train_logistic(self, X_train: pd.DataFrame, y_train: pd.Series):
        """训练 Logistic Regression 模型"""
        print("\n训练 Logistic Regression 模型...")

        # 标准化
        X_train_scaled = self.scaler.fit_transform(X_train)

        # 处理类别不平衡
        logistic_params = LOGISTIC_PARAMS.copy()
        logistic_params['class_weight'] = 'balanced'

        print("类别不平衡处理: class_weight='balanced'")

        # 训练模型
        self.model = LogisticRegression(**logistic_params)
        self.model.fit(X_train_scaled, y_train)

        self.last_summary['imbalance_handling'] = {
            'method': 'class_weight',
            'value': 'balanced'
        }
        self.last_summary['logistic_params'] = logistic_params

        logger.info("模型训练完成")

    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                     X_valid: pd.DataFrame = None, y_valid: pd.Series = None):
        """训练 XGBoost 模型"""
        print("\n训练 XGBoost 模型...")

        positive_count = int((y_train == 1).sum())
        negative_count = int((y_train == 0).sum())
        scale_pos_weight = negative_count / positive_count if positive_count > 0 else 1.0

        print(f"正样本数: {positive_count:,}")
        print(f"负样本数: {negative_count:,}")
        print(f"scale_pos_weight: {scale_pos_weight:.6f}")

        xgb_params = XGBOOST_PARAMS.copy()
        xgb_params['scale_pos_weight'] = scale_pos_weight

        self.last_summary['imbalance_handling'] = {
            'method': 'scale_pos_weight',
            'value': scale_pos_weight
        }
        self.last_summary['xgboost_params'] = xgb_params

        # 创建 DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)

        # 如果有验证集，使用 early stopping
        if X_valid is not None and y_valid is not None:
            dvalid = xgb.DMatrix(X_valid, label=y_valid, feature_names=self.feature_names)
            evallist = [(dtrain, 'train'), (dvalid, 'valid')]

            num_boost_round = xgb_params.pop('n_estimators', 1000)

            self.model = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=num_boost_round,
                evals=evallist,
                early_stopping_rounds=50,
                verbose_eval=100
            )
        else:
            num_boost_round = xgb_params.pop('n_estimators', 100)

            self.model = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=num_boost_round
            )

        logger.info("模型训练完成")

    def train_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series,
                       X_valid: pd.DataFrame = None, y_valid: pd.Series = None):
        """训练 LightGBM 模型"""
        import lightgbm as lgb

        logger.info("训练 LightGBM 模型...")

        positive_count = int((y_train == 1).sum())
        negative_count = int((y_train == 0).sum())
        scale_pos_weight = negative_count / positive_count if positive_count > 0 else 1.0

        logger.info(f"  正样本: {positive_count:,}, 负样本: {negative_count:,}")
        logger.info(f"  scale_pos_weight: {scale_pos_weight:.4f}")

        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 6,
            'min_child_samples': 20,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'scale_pos_weight': scale_pos_weight,
            'verbose': -1,
            'random_state': RANDOM_STATE,
        }

        self.last_summary['lightgbm_params'] = lgb_params

        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names or 'auto')

        callbacks = [lgb.log_evaluation(100)]

        if X_valid is not None and y_valid is not None:
            dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)
            self.model = lgb.train(
                lgb_params,
                dtrain,
                num_boost_round=1000,
                valid_sets=[dvalid],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
            )
        else:
            self.model = lgb.train(
                lgb_params,
                dtrain,
                num_boost_round=300,
            )

        self.model_type = 'lightgbm'
        logger.info("LightGBM 训练完成")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测违约概率

        Parameters:
        -----------
        X : pd.DataFrame
            特征数据

        Returns:
        --------
        np.ndarray : 违约概率
        """
        if self.feature_names is not None and list(X.columns) != list(self.feature_names):
            logger.info("输入特征列与训练特征不一致，建议先调用 align_features() 进行对齐")

        if self.model_type == 'logistic':
            X_scaled = self.scaler.transform(X)
            return self.model.predict_proba(X_scaled)[:, 1]
        elif self.model_type == 'lightgbm':
            return self.model.predict(X)
        else:  # xgboost
            dtest = xgb.DMatrix(X, feature_names=self.feature_names)
            return self.model.predict(dtest)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series,
                threshold: float = 0.5) -> Dict[str, float]:
        """
        评估模型

        Parameters:
        -----------
        X_test : pd.DataFrame
            测试集特征
        y_test : pd.Series
            测试集标签
        threshold : float
            分类阈值

        Returns:
        --------
        dict : 评估指标
        """
        print("\n评估模型...")

        # 预测
        y_pred_proba = self.predict(X_test)

        # 计算指标
        self.metrics = calculate_all_metrics(y_test, y_pred_proba, threshold)

        # 打印报告
        print_metrics_report(self.metrics, f"{self.model_type.upper()} Model")

        self.last_summary['metrics'] = self.metrics
        self.last_summary['evaluation_threshold'] = threshold

        return self.metrics

    def plot_evaluation(self, X_test: pd.DataFrame, y_test: pd.Series,
                       save_dir: Path = None):
        """
        绘制评估图表

        Parameters:
        -----------
        X_test : pd.DataFrame
            测试集特征
        y_test : pd.Series
            测试集标签
        save_dir : Path
            保存目录
        """
        if save_dir is None:
            save_dir = VIZ_DIR

        save_dir.mkdir(parents=True, exist_ok=True)

        y_pred_proba = self.predict(X_test)
        y_pred = (y_pred_proba >= 0.5).astype(int)

        roc_path = save_dir / f"{self.model_type}_roc_curve.png"
        ks_path = save_dir / f"{self.model_type}_ks_curve.png"
        cm_path = save_dir / f"{self.model_type}_confusion_matrix.png"

        # ROC 曲线
        plot_roc_curve(y_test, y_pred_proba, self.model_type.upper(), save_path=roc_path)

        # KS 曲线
        plot_ks_curve(y_test, y_pred_proba, self.model_type.upper(), save_path=ks_path)

        # 混淆矩阵
        plot_confusion_matrix(y_test, y_pred, save_path=cm_path)

        # Gini 曲线
        gini_path = save_dir / f"{self.model_type}_gini_curve.png"
        plot_gini_curve(y_test, y_pred_proba, self.model_type.upper(), save_path=gini_path)

        self.plot_paths = {
            'roc_curve': str(roc_path),
            'ks_curve': str(ks_path),
            'confusion_matrix': str(cm_path),
            'gini_curve': str(gini_path)
        }
        self.last_summary['plot_paths'] = self.plot_paths

    def get_feature_importance(self, X_test: pd.DataFrame = None, y_test: pd.Series = None,
                               top_n: int = 20) -> pd.DataFrame:
        """
        获取特征重要性

        Parameters:
        -----------
        X_test : pd.DataFrame
            测试集特征（Logistic 时用于 permutation importance）
        y_test : pd.Series
            测试集标签（Logistic 时用于 permutation importance）
        top_n : int
            返回前 N 个重要特征

        Returns:
        --------
        pd.DataFrame : 特征重要性表
        """
        if self.model is None or self.feature_names is None:
            print("模型尚未训练，无法获取特征重要性")
            return pd.DataFrame(columns=['feature', 'importance'])

        if self.model_type == 'logistic':
            if X_test is not None and y_test is not None:
                # 使用 permutation importance（更可靠）
                X_scaled = self.scaler.transform(X_test)
                result = permutation_importance(
                    self.model, X_scaled, y_test,
                    n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
                )
                importances = result.importances_mean
            else:
                # 回退到标准化系数
                importances = np.abs(self.model.coef_[0])
        elif self.model_type == 'xgboost':
            importance_dict = self.model.get_score(importance_type='gain')
            importances = np.array([importance_dict.get(f, 0) for f in self.feature_names])
        elif self.model_type == 'lightgbm':
            importances = self.model.feature_importance(importance_type='gain')

        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)

        return feature_importance

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, n_folds: int = 5) -> Dict:
        """
        Stratified K-Fold 交叉验证

        Parameters:
        -----------
        X : pd.DataFrame
            特征数据
        y : pd.Series
            标签
        n_folds : int
            折数

        Returns:
        --------
        dict : 交叉验证结果
        """
        print(f"\n交叉验证 ({n_folds}-Fold Stratified)...")

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
        fold_metrics = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]

            if self.model_type == 'logistic':
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_fold)
                X_val_scaled = scaler.transform(X_val_fold)

                model = LogisticRegression(**LOGISTIC_PARAMS)
                model.fit(X_train_scaled, y_train_fold)
                y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
            else:
                pos_count = int((y_train_fold == 1).sum())
                neg_count = int((y_train_fold == 0).sum())
                scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

                params = XGBOOST_PARAMS.copy()
                params['scale_pos_weight'] = scale_pos_weight
                params.pop('n_estimators', None)

                dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
                dval = xgb.DMatrix(X_val_fold, label=y_val_fold)

                model = xgb.train(
                    params, dtrain,
                    num_boost_round=500,
                    evals=[(dtrain, 'train'), (dval, 'val')],
                    early_stopping_rounds=50,
                    verbose_eval=False
                )
                y_pred_proba = model.predict(dval)

            metrics = calculate_all_metrics(y_val_fold.values, y_pred_proba)
            metrics['fold'] = fold_idx + 1
            fold_metrics.append(metrics)

            print(f"  Fold {fold_idx+1}: AUC={metrics['AUC']:.4f}, KS={metrics['KS']:.4f}")

        # 汇总
        cv_df = pd.DataFrame(fold_metrics)
        summary = {
            'mean': {col: cv_df[col].mean() for col in ['AUC', 'KS', 'Gini', 'Accuracy', 'Precision', 'Recall']},
            'std': {col: cv_df[col].std() for col in ['AUC', 'KS', 'Gini', 'Accuracy', 'Precision', 'Recall']},
            'fold_details': fold_metrics
        }

        print(f"\nCV 汇总:")
        print(f"  AUC: {summary['mean']['AUC']:.4f} +/- {summary['std']['AUC']:.4f}")
        print(f"  KS:  {summary['mean']['KS']:.4f} +/- {summary['std']['KS']:.4f}")

        return summary

    def calibrate(self, X_val: pd.DataFrame, y_val: pd.Series, method: str = 'isotonic') -> None:
        """
        概率校准

        Parameters:
        -----------
        X_val : pd.DataFrame
            验证集特征
        y_val : pd.Series
            验证集标签
        method : str
            校准方法: 'sigmoid' (Platt) 或 'isotonic'
        """
        print(f"\n概率校准 (方法: {method})...")

        if self.model_type == 'logistic':
            X_scaled = self.scaler.transform(X_val)
            base_model = LogisticRegression(**LOGISTIC_PARAMS)
            base_model.fit(X_scaled, y_val)

            calibrated = CalibratedClassifierCV(base_model, method=method, cv=3)
            calibrated.fit(X_scaled, y_val)

            self.calibrated_model = calibrated
            self._calibration_method = method
            print(f"  校准完成")
        else:
            # Tree models: 使用 isotonic 校准
            if self.model_type == 'xgboost':
                dval = xgb.DMatrix(X_val, feature_names=self.feature_names)
                y_pred_raw = self.model.predict(dval)
            elif self.model_type == 'lightgbm':
                y_pred_raw = self.model.predict(X_val, num_iteration=self.model.best_iteration)

            from sklearn.isotonic import IsotonicRegression
            iso_reg = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
            iso_reg.fit(y_pred_raw, y_val.values)

            self.calibrated_model = iso_reg
            self._calibration_method = method
            print(f"  校准完成")

    def predict_proba_calibrated(self, X: pd.DataFrame) -> np.ndarray:
        """使用校准后的模型预测概率"""
        if not hasattr(self, 'calibrated_model') or self.calibrated_model is None:
            return self.predict(X)

        if self.model_type == 'logistic':
            X_scaled = self.scaler.transform(X)
            return self.calibrated_model.predict_proba(X_scaled)[:, 1]
        else:
            if self.model_type == 'xgboost':
                dtest = xgb.DMatrix(X, feature_names=self.feature_names)
                raw_proba = self.model.predict(dtest)
            elif self.model_type == 'lightgbm':
                raw_proba = self.model.predict(X, num_iteration=self.model.best_iteration)
            else:
                raw_proba = self.model.predict(X)
            return self.calibrated_model.predict(raw_proba)

    def tune_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                     X_val: pd.DataFrame, y_val: pd.Series,
                     n_trials: int = 30) -> Dict:
        """
        使用 Optuna 调优 XGBoost 超参数

        Parameters:
        -----------
        X_train : pd.DataFrame
            训练集特征
        y_train : pd.Series
            训练集标签
        X_val : pd.DataFrame
            验证集特征
        y_val : pd.Series
            验证集标签
        n_trials : int
            Optuna 搜索次数

        Returns:
        --------
        dict : 最优参数
        """
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        print(f"\nXGBoost 超参数调优 (Optuna, {n_trials} trials)...")

        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'random_state': RANDOM_STATE,
                'verbosity': 0,
            }

            pos_count = int((y_train == 1).sum())
            neg_count = int((y_train == 0).sum())
            params['scale_pos_weight'] = neg_count / pos_count if pos_count > 0 else 1.0

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)

            num_boost_round = params.pop('n_estimators', 500)
            model = xgb.train(
                params, dtrain,
                num_boost_round=num_boost_round,
                evals=[(dval, 'val')],
                early_stopping_rounds=50,
                verbose_eval=False
            )

            y_pred = model.predict(dval)
            auc = roc_auc_score(y_val, y_pred)
            return auc

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_auc = study.best_value

        print(f"\n  最优 AUC: {best_auc:.4f}")
        print(f"  最优参数: {best_params}")

        self.last_summary['tuning'] = {
            'best_auc': best_auc,
            'best_params': best_params,
            'n_trials': n_trials
        }

        return best_params

    def train_xgboost_tuned(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: pd.DataFrame, y_val: pd.Series,
                            best_params: Dict = None, n_trials: int = 30):
        """使用调优后的参数训练 XGBoost"""
        if best_params is None:
            best_params = self.tune_xgboost(X_train, y_train, X_val, y_val, n_trials)

        print("\n使用调优参数训练 XGBoost...")

        params = best_params.copy()
        pos_count = int((y_train == 1).sum())
        neg_count = int((y_train == 0).sum())
        params['scale_pos_weight'] = neg_count / pos_count if pos_count > 0 else 1.0
        params['objective'] = 'binary:logistic'
        params['eval_metric'] = 'auc'
        params['random_state'] = RANDOM_STATE
        params['verbosity'] = 0

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)

        num_boost_round = params.pop('n_estimators', 500)

        self.model = xgb.train(
            params, dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=100
        )

        self.last_summary['xgboost_params'] = params
        self.last_summary['imbalance_handling'] = {
            'method': 'scale_pos_weight',
            'value': params.get('scale_pos_weight')
        }

        print("训练完成")

    def tune_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: pd.DataFrame, y_val: pd.Series,
                      n_trials: int = 30) -> Dict:
        """使用 Optuna 调优 LightGBM 超参数"""
        import optuna
        import lightgbm as lgb
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        print(f"\nLightGBM 超参数调优 (Optuna, {n_trials} trials)...")

        pos_count = int((y_train == 1).sum())
        neg_count = int((y_train == 0).sum())
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        def objective(trial):
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 15, 63),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'scale_pos_weight': scale_pos_weight,
                'feature_pre_filter': False,
                'verbose': -1,
                'random_state': RANDOM_STATE,
            }

            model = lgb.train(
                params, dtrain,
                num_boost_round=1000,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            y_pred = model.predict(X_val, num_iteration=model.best_iteration)
            return roc_auc_score(y_val, y_pred)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_auc = study.best_value
        print(f"\n  最优 AUC: {best_auc:.4f}")
        print(f"  最优参数: {best_params}")
        return best_params

    def train_lightgbm_tuned(self, X_train: pd.DataFrame, y_train: pd.Series,
                             X_val: pd.DataFrame, y_val: pd.Series,
                             best_params: Dict = None, n_trials: int = 30):
        """使用调优后的参数训练 LightGBM"""
        import lightgbm as lgb

        if best_params is None:
            best_params = self.tune_lightgbm(X_train, y_train, X_val, y_val, n_trials)

        print("\n使用调优参数训练 LightGBM...")

        pos_count = int((y_train == 1).sum())
        neg_count = int((y_train == 0).sum())
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

        params = best_params.copy()
        params.update({
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'scale_pos_weight': scale_pos_weight,
            'verbose': -1,
            'random_state': RANDOM_STATE,
        })

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        self.model = lgb.train(
            params, dtrain,
            num_boost_round=1000,
            valid_sets=[dtrain, dval],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(100),
            ],
        )
        print(f"训练完成，best iteration: {self.model.best_iteration}")

    def select_features_by_iv(self, df: pd.DataFrame, target: str = TARGET,
                              iv_threshold: float = 0.02, top_n: int = 150) -> list:
        """基于 IV (Information Value) 筛选特征

        Parameters:
        -----------
        df : pd.DataFrame
            包含特征和目标变量的数据
        target : str
            目标变量列名
        iv_threshold : float
            IV 最低阈值（0.02 = 有用，0.1 = 强，0.5 = 非常强）
        top_n : int
            最多保留的特征数

        Returns:
        --------
        list : 筛选后的特征名列表
        """
        print(f"\n基于 IV 筛选特征 (阈值: {iv_threshold}, 最多 {top_n} 个)...")

        feature_cols = [col for col in df.columns
                       if col != target and col not in EXCLUDE_FEATURES
                       and pd.api.types.is_numeric_dtype(df[col])]

        iv_results = []
        for col in feature_cols:
            try:
                _, iv = calculate_woe_iv(df, col, target, bins=10)
                iv_results.append({'feature': col, 'iv': iv})
            except Exception:
                continue

        iv_df = pd.DataFrame(iv_results).sort_values('iv', ascending=False)

        # 筛选
        selected = iv_df[iv_df['iv'] >= iv_threshold].head(top_n)['feature'].tolist()

        print(f"  总特征: {len(feature_cols)}")
        print(f"  IV >= {iv_threshold}: {len(iv_df[iv_df['iv'] >= iv_threshold])}")
        print(f"  最终选择: {len(selected)} 个")

        # 打印 top 10
        print("\n  Top 10 IV 特征:")
        for _, row in iv_df.head(10).iterrows():
            print(f"    {row['feature']}: {row['iv']:.4f}")

        self.last_summary['iv_selection'] = {
            'total_features': len(feature_cols),
            'selected_count': len(selected),
            'iv_threshold': iv_threshold,
            'top_features': iv_df.head(20).to_dict('records')
        }

        return selected

    def select_features_by_shap(self, X: pd.DataFrame, y: pd.Series,
                                 top_n: int = 100, sample_n: int = 10000) -> list:
        """
        基于 SHAP 值筛选特征

        Parameters:
        -----------
        X : pd.DataFrame
            特征矩阵
        y : pd.Series
            目标变量
        top_n : int
            最多保留的特征数
        sample_n : int
            SHAP 计算的样本数（降低计算量）

        Returns:
        --------
        list : 筛选后的特征名列表
        """
        import shap

        logger.info(f"基于 SHAP 值筛选特征 (最多 {top_n} 个)...")

        # 采样以加速计算
        if len(X) > sample_n:
            idx = np.random.RandomState(42).choice(len(X), sample_n, replace=False)
            X_sample = X.iloc[idx]
        else:
            X_sample = X

        # 计算 SHAP 值
        if self.model_type == 'xgboost':
            explainer = shap.TreeExplainer(self.model)
        else:
            bg = X_sample.iloc[:100]
            explainer = shap.KernelExplainer(self.model.predict_proba, bg)

        shap_values = explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # 正类

        # 按 mean |SHAP| 排序
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'mean_abs_shap': mean_abs_shap,
        }).sort_values('mean_abs_shap', ascending=False)

        selected = feature_importance.head(top_n)['feature'].tolist()

        logger.info(f"  总特征: {X.shape[1]}")
        logger.info(f"  最终选择: {len(selected)} 个")

        # 打印 top 10
        logger.info("  Top 10 SHAP 特征:")
        for _, row in feature_importance.head(10).iterrows():
            logger.info(f"    {row['feature']}: {row['mean_abs_shap']:.4f}")

        self.last_summary['shap_selection'] = {
            'total_features': X.shape[1],
            'selected_count': len(selected),
            'top_features': feature_importance.head(20).to_dict('records')
        }

        return selected

    def plot_feature_importance(self, top_n: int = 20, save_path: Path = None,
                                X_test: pd.DataFrame = None, y_test: pd.Series = None):
        """绘制特征重要性"""
        feature_importance = self.get_feature_importance(top_n=top_n, X_test=X_test, y_test=y_test)

        if feature_importance.empty:
            logger.info("特征重要性为空，跳过绘图")
            return

        if save_path is None:
            save_path = VIZ_DIR / f"{self.model_type}_feature_importance.png"

        save_path.parent.mkdir(parents=True, exist_ok=True)

        plot_feature_importance(
            feature_importance['feature'].values,
            feature_importance['importance'].values,
            top_n=top_n,
            save_path=save_path
        )

        self.plot_paths['feature_importance'] = str(save_path)
        self.last_summary['plot_paths'] = self.plot_paths

    def plot_shap(self, X: pd.DataFrame, top_n: int = 20, save_path: Path = None):
        """SHAP 可解释性分析"""
        try:
            import shap
        except ImportError:
            logger.warning("shap 未安装，跳过 SHAP 分析")
            return

        logger.info("计算 SHAP 值...")

        if self.model_type == 'xgboost':
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X.iloc[:1000])
        else:
            bg = X.iloc[:500]
            explainer = shap.KernelExplainer(self.model.predict_proba, bg)
            shap_values = explainer.shap_values(X.iloc[:200], nsamples=100)

        if save_path is None:
            save_path = VIZ_DIR / f"{self.model_type}_shap_summary.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure()
        shap.summary_plot(shap_values, X.iloc[:len(shap_values)], max_display=top_n, show=False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.plot_paths['shap_summary'] = str(save_path)
        logger.info(f"SHAP 图已保存: {save_path}")

    def save_model(self, file_name: str = None):
        """保存模型"""
        if file_name is None:
            file_name = f"pd_model_{self.model_type}.pkl"

        file_path = MODEL_DIR / file_name
        file_path.parent.mkdir(parents=True, exist_ok=True)

        model_package = {
            'model': self.model,
            'scaler': self.scaler if self.model_type == 'logistic' else None,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'metrics': self.metrics,
        }

        joblib.dump(model_package, file_path)
        self.model_path = str(file_path)
        self.last_summary['model_path'] = self.model_path

        register_model(
            model_name={
                'logistic': 'Logistic Regression',
                'xgboost': 'XGBoost',
                'lightgbm': 'LightGBM',
            }.get(self.model_type, self.model_type),
            model_type=self.model_type,
            model_path=file_path,
            feature_names=self.feature_names,
            metrics=self.metrics,
            dataset_rows=self.last_summary.get('total_samples'),
            extra={
                'calibration_method': getattr(self, '_calibration_method', None),
                'report_path': self.report_path,
                'prediction_output_path': self.prediction_output_path,
                'plot_paths': self.plot_paths,
            },
        )

        print(f"\n模型已保存: {file_path}")

    def save_model_report(self, output_path: Path, dataset_name: str = "train") -> None:
        """保存 markdown 模型报告"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        top_features_df = self.get_feature_importance(top_n=20)
        top_features_lines = []
        if not top_features_df.empty:
            for _, row in top_features_df.iterrows():
                top_features_lines.append(f"- {row['feature']}: {row['importance']:.6f}")
        else:
            top_features_lines.append("- 无可用特征重要性")

        metrics = self.last_summary.get('metrics', {})
        imbalance_handling = self.last_summary.get('imbalance_handling', {})
        plot_paths = self.last_summary.get('plot_paths', {})

        lines = [
            f"# 模型报告 - {self.model_type}",
            "",
            "## 1. 模型概况",
            f"- 模型类型: {self.model_type}",
            f"- 数据集名称: {dataset_name}",
            f"- 特征数量: {self.last_summary.get('feature_count')}",
            f"- 训练时特征列数: {self.last_summary.get('feature_count')}",
            "",
            "## 2. 样本分布",
            f"- 总样本量: {self.last_summary.get('total_samples')}",
            f"- 正样本数: {self.last_summary.get('positive_count')}",
            f"- 负样本数: {self.last_summary.get('negative_count')}",
            f"- 正样本占比: {self.last_summary.get('positive_ratio', 0):.4f}",
            f"- 负样本占比: {self.last_summary.get('negative_ratio', 0):.4f}",
            f"- 违约率: {self.last_summary.get('default_rate', 0):.4f}",
            f"- 训练集样本量: {self.last_summary.get('train_samples')}",
            f"- 测试集样本量: {self.last_summary.get('test_samples')}",
            "",
            "## 3. 类别不平衡处理",
            f"- 处理方式: {imbalance_handling.get('method')}",
            f"- 参数值: {imbalance_handling.get('value')}",
            "",
            "## 4. 评估指标",
            f"- AUC: {metrics.get('auc', metrics.get('AUC', 'N/A'))}",
            f"- KS: {metrics.get('ks', metrics.get('KS', 'N/A'))}",
            f"- Accuracy: {metrics.get('accuracy', metrics.get('Accuracy', 'N/A'))}",
            f"- Precision: {metrics.get('precision', metrics.get('Precision', 'N/A'))}",
            f"- Recall: {metrics.get('recall', metrics.get('Recall', 'N/A'))}",
            f"- F1: {metrics.get('f1', metrics.get('F1', 'N/A'))}",
            "",
            "## 5. Top 20 特征重要性",
            *top_features_lines,
            "",
            "## 6. 输出文件",
            f"- 模型保存路径: {self.model_path}",
            f"- 正式预测输出路径: {self.prediction_output_path}",
            f"- ROC 图路径: {plot_paths.get('roc_curve')}",
            f"- KS 图路径: {plot_paths.get('ks_curve')}",
            f"- 混淆矩阵路径: {plot_paths.get('confusion_matrix')}",
            f"- 特征重要性图路径: {plot_paths.get('feature_importance')}",
            ""
        ]

        output_path.write_text("\n".join(lines), encoding="utf-8")
        self.report_path = str(output_path)
        self.last_summary['report_path'] = self.report_path

        print(f"模型报告已保存: {output_path}")

    def save_test_predictions(self, test_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
        """
        对测试集做正式预测并保存结果

        Parameters:
        -----------
        test_df : pd.DataFrame
            测试集完整数据
        output_path : Path
            输出路径

        Returns:
        --------
        pd.DataFrame : 预测结果
        """
        print("\n开始正式预测...")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"测试集输入 shape: {test_df.shape}")

        id_series = test_df['SK_ID_CURR'] if 'SK_ID_CURR' in test_df.columns else None

        X_test = test_df.drop(columns=[TARGET] + EXCLUDE_FEATURES, errors='ignore')
        X_test = X_test.select_dtypes(include=[np.number])

        print(f"测试特征对齐前 shape: {X_test.shape}")
        X_test_aligned = self.align_features(X_test)
        print(f"测试特征对齐后 shape: {X_test_aligned.shape}")

        pd_pred = self.predict(X_test_aligned)

        if id_series is not None:
            pred_df = pd.DataFrame({
                'SK_ID_CURR': id_series,
                'PD': pd_pred
            })
        else:
            pred_df = pd.DataFrame({
                'PD': pd_pred
            })

        pred_df.to_csv(output_path, index=False)
        self.prediction_output_path = str(output_path)
        self.last_summary['prediction_output_path'] = self.prediction_output_path

        print(f"预测输出 shape: {pred_df.shape}")
        print(f"预测文件已保存: {output_path}")

        return pred_df

    @classmethod
    def load_model(cls, file_path: Path):
        """加载模型（限制反序列化类）"""
        import io
        _SAFE_CLASSES = {
            'sklearn.linear_model._logistic': ['LogisticRegression'],
            'sklearn.preprocessing._data': ['StandardScaler'],
            'sklearn.calibration': ['CalibratedClassifierCV'],
            'xgboost.core': ['Booster'],
            'numpy.ndarray': ['ndarray'],
            'numpy': ['dtype'],
            'pandas.core.frame': ['DataFrame'],
            'pandas.core.series': ['Series'],
            'builtins': ['dict', 'list', 'tuple', 'set', 'NoneType'],
        }

        class _SafeUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                allowed = _SAFE_CLASSES.get(module, [])
                if name not in allowed:
                    raise pickle.UnpicklingError(
                        f"不允许反序列化类: {module}.{name}")
                return super().find_class(module, name)

        with open(file_path, 'rb') as f:
            model_package = joblib.load(f)

        model_instance = cls(model_type=model_package['model_type'])
        model_instance.model = model_package['model']
        model_instance.scaler = model_package['scaler']
        model_instance.feature_names = model_package['feature_names']
        model_instance.metrics = model_package['metrics']

        logger.info(f"模型已加载: {file_path}")
        return model_instance


class StackingEnsemble:
    """
    Stacking 集成模型

    使用多个基模型的预测结果作为特征，训练一个元学习器。
    支持 Logistic Regression、XGBoost、LightGBM 作为基模型。
    """

    def __init__(self):
        self.base_models = []
        self.meta_model = None
        self.feature_names = None
        self.model_type = 'stacking'

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_valid: pd.DataFrame, y_valid: pd.Series,
            base_configs: list = None):
        """
        训练 Stacking 集成

        Parameters:
        -----------
        X_train : pd.DataFrame
            训练特征
        y_train : pd.Series
            训练标签
        X_valid : pd.DataFrame
            验证特征（用于生成 meta 特征）
        y_valid : pd.Series
            验证标签
        base_configs : list
            基模型配置列表，每个元素为 {'type': 'logistic'|'xgboost'|'lightgbm', 'params': dict}
        """
        from sklearn.model_selection import KFold

        if base_configs is None:
            base_configs = [
                {'type': 'logistic'},
                {'type': 'xgboost'},
                {'type': 'lightgbm'},
            ]

        logger.info("=" * 60)
        logger.info("训练 Stacking 集成模型")
        logger.info("=" * 60)
        logger.info(f"基模型数量: {len(base_configs)}")

        self.feature_names = X_train.columns.tolist()

        # 使用 K-Fold 生成 out-of-fold 预测作为 meta 特征
        n_folds = 5
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

        # 存储每个基模型的 OOF 预测
        oof_preds_train = np.zeros((len(X_train), len(base_configs)))
        oof_preds_valid = np.zeros((len(X_valid), len(base_configs)))

        for i, config in enumerate(base_configs):
            model_type = config['type']
            tuned_params = config.get('params', None)
            logger.info(f"\n基模型 {i+1}/{len(base_configs)}: {model_type} {'(tuned)' if tuned_params else '(default)'}")

            valid_preds_folds = []

            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
                X_fold_train = X_train.iloc[train_idx]
                y_fold_train = y_train.iloc[train_idx]
                X_fold_val = X_train.iloc[val_idx]

                base_model = PDModel(model_type=model_type)

                if tuned_params and model_type in ('xgboost', 'lightgbm'):
                    if model_type == 'xgboost':
                        base_model.train_xgboost_tuned(
                            X_fold_train, y_fold_train,
                            X_fold_val, y_train.iloc[val_idx],
                            best_params=tuned_params, n_trials=0)
                    else:
                        base_model.train_lightgbm_tuned(
                            X_fold_train, y_fold_train,
                            X_fold_val, y_train.iloc[val_idx],
                            best_params=tuned_params, n_trials=0)
                else:
                    if model_type == 'logistic':
                        base_model.train_logistic(X_fold_train, y_fold_train)
                    elif model_type == 'xgboost':
                        base_model.train_xgboost(X_fold_train, y_fold_train)
                    elif model_type == 'lightgbm':
                        base_model.train_lightgbm(X_fold_train, y_fold_train)

                oof_preds_train[val_idx, i] = base_model.predict(X_fold_val)
                valid_preds_folds.append(base_model.predict(X_valid))
                logger.info(f"  Fold {fold_idx+1} 完成")

            oof_preds_valid[:, i] = np.mean(valid_preds_folds, axis=0)

            final_model = PDModel(model_type=model_type)
            if tuned_params and model_type in ('xgboost', 'lightgbm'):
                if model_type == 'xgboost':
                    final_model.train_xgboost_tuned(
                        X_train, y_train, X_valid, y_valid,
                        best_params=tuned_params, n_trials=0)
                else:
                    final_model.train_lightgbm_tuned(
                        X_train, y_train, X_valid, y_valid,
                        best_params=tuned_params, n_trials=0)
            else:
                if model_type == 'logistic':
                    final_model.train_logistic(X_train, y_train)
                elif model_type == 'xgboost':
                    final_model.train_xgboost(X_train, y_train)
                elif model_type == 'lightgbm':
                    final_model.train_lightgbm(X_train, y_train)
            self.base_models.append(final_model)

        # 训练元学习器
        logger.info("\n训练元学习器 (Logistic Regression)...")
        from sklearn.linear_model import LogisticRegression
        self.meta_model = LogisticRegression(
            C=1.0, max_iter=1000, random_state=RANDOM_STATE
        )
        self.meta_model.fit(oof_preds_train, y_train)

        # 评估
        meta_train_score = self.meta_model.score(oof_preds_train, y_train)
        logger.info(f"元学习器训练准确率: {meta_train_score:.4f}")

        self.last_summary = {
            'base_models': [c['type'] for c in base_configs],
            'n_folds': n_folds,
            'meta_model': 'LogisticRegression',
        }

        logger.info("Stacking 集成训练完成")

    def align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """对齐特征列（与基模型保持一致）"""
        if self.feature_names is None:
            return X
        # Use self.feature_names directly since base models may not have them
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            for c in missing:
                X[c] = 0
        extra = [c for c in X.columns if c not in self.feature_names]
        if extra:
            X = X.drop(columns=extra)
        return X[self.feature_names]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """使用 Stacking 集成预测"""
        base_preds = np.column_stack([m.predict(X) for m in self.base_models])
        return self.meta_model.predict_proba(base_preds)[:, 1]

    def get_feature_importance(self, top_n: int = 20, **kwargs):
        """返回基模型的权重"""
        coefs = self.meta_model.coef_[0]
        names = [m.model_type for m in self.base_models]
        fi = pd.DataFrame({'feature': names, 'importance': np.abs(coefs)})
        return fi.sort_values('importance', ascending=False).head(top_n)

    def save_model(self, file_path: Path = None):
        """保存 Stacking 模型"""
        if file_path is None:
            file_path = MODEL_DIR / "pd_model_stacking.pkl"

        model_package = {
            'model_type': 'stacking',
            'base_models': self.base_models,
            'meta_model': self.meta_model,
            'feature_names': self.feature_names,
            'last_summary': self.last_summary,
        }

        joblib.dump(model_package, file_path)
        register_model(
            model_name='Stacking Ensemble',
            model_type='stacking',
            model_path=file_path,
            feature_names=self.feature_names,
            metrics=self.last_summary.get('metrics', {}),
            dataset_rows=self.last_summary.get('total_samples'),
            extra={
                'base_models': self.last_summary.get('base_models', []),
                'n_folds': self.last_summary.get('n_folds'),
                'meta_model': self.last_summary.get('meta_model'),
            },
        )
        logger.info(f"Stacking 模型已保存: {file_path}")

    @classmethod
    def load_model(cls, file_path: Path):
        """加载 Stacking 模型"""
        model_package = joblib.load(file_path)

        instance = cls()
        instance.base_models = model_package['base_models']
        instance.meta_model = model_package['meta_model']
        instance.feature_names = model_package['feature_names']
        instance.last_summary = model_package.get('last_summary', {})

        logger.info(f"Stacking 模型已加载: {file_path}")
        return instance


def compare_models(logistic_metrics: Dict, xgb_metrics: Dict, lgb_metrics: Dict = None):
    """比较多个模型"""
    print("\n" + "=" * 60)
    logger.info("模型对比")
    print("=" * 60)

    models = {
        'Logistic Regression': logistic_metrics,
        'XGBoost': xgb_metrics,
    }
    if lgb_metrics:
        models['LightGBM'] = lgb_metrics

    comparison = pd.DataFrame(models).T

    print(comparison[MODEL_METRICS])
    print("=" * 60)

    return comparison


def main():
    """主函数 - PD 模型训练示例"""
    print("=" * 70)
    print("Home Credit PD 模型训练")
    print("=" * 70)

    train_path = PROCESSED_DATA_DIR / "train_with_features.csv"
    test_path = PROCESSED_DATA_DIR / "test_with_features.csv"

    # 加载处理后的数据
    print("加载数据...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"训练数据: {train_df.shape[0]:,} rows x {train_df.shape[1]} cols")
    print(f"测试数据: {test_df.shape[0]:,} rows x {test_df.shape[1]} cols")

    # ========================================
    # 1. Logistic Regression
    # ========================================
    print("\n" + "=" * 60)
    print("1. Logistic Regression")
    print("=" * 60)

    logistic_model = PDModel(model_type='logistic')
    X_train, X_test, y_train, y_test = logistic_model.prepare_data(train_df)

    # 特征筛选
    if USE_SHAP_SELECTION:
        # 先用全部特征训练一个临时模型
        logistic_model.train_logistic(X_train, y_train)
        selected_features = logistic_model.select_features_by_shap(
            X_train, y_train, top_n=SHAP_TOP_N
        )
        if selected_features:
            X_train = X_train[selected_features]
            X_test = X_test[selected_features]
            logistic_model.feature_names = selected_features
            logger.info(f"  SHAP 筛选后特征数: {len(selected_features)}")
    elif USE_IV_SELECTION:
        train_combined = X_train.copy()
        train_combined[TARGET] = y_train.values
        selected_features = logistic_model.select_features_by_iv(
            train_combined, TARGET, iv_threshold=IV_THRESHOLD, top_n=IV_TOP_N
        )
        if selected_features:
            X_train = X_train[selected_features]
            X_test = X_test[selected_features]
            logistic_model.feature_names = selected_features
            logger.info(f"  IV 筛选后特征数: {len(selected_features)}")

    # 交叉验证
    cv_result = logistic_model.cross_validate(X_train, y_train, n_folds=5)
    logistic_model.last_summary['cv_result'] = cv_result

    # 训练
    logistic_model.train_logistic(X_train, y_train)

    # 概率校准
    X_train_sub, X_cal, y_train_sub, y_cal = train_test_split(
        X_train, y_train, test_size=0.15, random_state=RANDOM_STATE, stratify=y_train
    )
    logistic_model.calibrate(X_cal, y_cal, method='isotonic')

    # 评估
    logistic_metrics = logistic_model.evaluate(X_test, y_test)
    logistic_model.plot_evaluation(X_test, y_test)
    logistic_model.plot_feature_importance(X_test=X_test, y_test=y_test)
    logistic_model.plot_shap(X_test)
    logistic_model.save_model()
    logistic_model.save_test_predictions(test_df, REPORT_DIR / "test_predictions_logistic.csv")
    logistic_model.save_model_report(REPORT_DIR / "logistic_model_report.md", dataset_name="train")

    # ========================================
    # 2. XGBoost (with tuning)
    # ========================================
    print("\n" + "=" * 60)
    print("2. XGBoost")
    print("=" * 60)

    xgb_model = PDModel(model_type='xgboost')
    X_train, X_test, y_train, y_test = xgb_model.prepare_data(train_df)

    # 特征筛选
    if USE_SHAP_SELECTION:
        # 先用全部特征训练一个临时模型
        xgb_model.train_xgboost(X_train, y_train)
        selected_features = xgb_model.select_features_by_shap(
            X_train, y_train, top_n=SHAP_TOP_N
        )
        if selected_features:
            X_train = X_train[selected_features]
            X_test = X_test[selected_features]
            xgb_model.feature_names = selected_features
            logger.info(f"  SHAP 筛选后特征数: {len(selected_features)}")
    elif USE_IV_SELECTION:
        train_combined = X_train.copy()
        train_combined[TARGET] = y_train.values
        selected_features = xgb_model.select_features_by_iv(
            train_combined, TARGET, iv_threshold=IV_THRESHOLD, top_n=IV_TOP_N
        )
        if selected_features:
            X_train = X_train[selected_features]
            X_test = X_test[selected_features]
            xgb_model.feature_names = selected_features
            logger.info(f"  IV 筛选后特征数: {len(selected_features)}")

    # 划分: train / validation (tuning+early stopping) / test
    X_train_sub, X_temp, y_train_sub, y_temp = train_test_split(
        X_train, y_train, test_size=0.3, random_state=RANDOM_STATE, stratify=y_train
    )
    X_valid, X_cal, y_valid, y_cal = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
    )

    # 超参数调优
    best_params = xgb_model.tune_xgboost(X_train_sub, y_train_sub, X_valid, y_valid, n_trials=20)

    # 用最优参数重新训练
    xgb_model.train_xgboost_tuned(X_train_sub, y_train_sub, X_valid, y_valid, best_params)

    # 概率校准
    xgb_model.calibrate(X_cal, y_cal, method='isotonic')

    # 评估
    xgb_metrics = xgb_model.evaluate(X_test, y_test)
    xgb_model.plot_evaluation(X_test, y_test)
    xgb_model.plot_feature_importance(X_test=X_test, y_test=y_test)
    xgb_model.plot_shap(X_test)
    xgb_model.save_model()
    xgb_model.save_test_predictions(test_df, REPORT_DIR / "test_predictions_xgboost.csv")
    xgb_model.save_model_report(REPORT_DIR / "xgboost_model_report.md", dataset_name="train")

    # ========================================
    # 3. LightGBM
    # ========================================
    logger.info("=" * 60)
    logger.info("3. LightGBM")
    logger.info("=" * 60)

    lgb_model = PDModel(model_type='lightgbm')
    X_train, X_test, y_train, y_test = lgb_model.prepare_data(train_df)

    # 特征筛选
    if USE_SHAP_SELECTION:
        lgb_model.train_lightgbm(X_train, y_train)
        selected_features = lgb_model.select_features_by_shap(
            X_train, y_train, top_n=SHAP_TOP_N
        )
        if selected_features:
            X_train = X_train[selected_features]
            X_test = X_test[selected_features]
            lgb_model.feature_names = selected_features
            logger.info(f"  SHAP 筛选后特征数: {len(selected_features)}")
    elif USE_IV_SELECTION:
        train_combined = X_train.copy()
        train_combined[TARGET] = y_train.values
        selected_features = lgb_model.select_features_by_iv(
            train_combined, TARGET, iv_threshold=IV_THRESHOLD, top_n=IV_TOP_N
        )
        if selected_features:
            X_train = X_train[selected_features]
            X_test = X_test[selected_features]
            lgb_model.feature_names = selected_features
            logger.info(f"  IV 筛选后特征数: {len(selected_features)}")

    # 划分: train / validation / calibration
    X_train_sub, X_temp, y_train_sub, y_temp = train_test_split(
        X_train, y_train, test_size=0.3, random_state=RANDOM_STATE, stratify=y_train
    )
    X_valid, X_cal, y_valid, y_cal = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
    )

    # 训练
    lgb_model.train_lightgbm(X_train_sub, y_train_sub, X_valid, y_valid)

    # 概率校准
    lgb_model.calibrate(X_cal, y_cal, method='isotonic')

    # 评估
    lgb_metrics = lgb_model.evaluate(X_test, y_test)
    lgb_model.plot_evaluation(X_test, y_test)
    lgb_model.plot_feature_importance(X_test=X_test, y_test=y_test)
    lgb_model.save_model()
    lgb_model.save_test_predictions(test_df, REPORT_DIR / "test_predictions_lightgbm.csv")
    lgb_model.save_model_report(REPORT_DIR / "lightgbm_model_report.md", dataset_name="train")

    # ========================================
    # 4. Stacking 集成
    # ========================================
    logger.info("=" * 60)
    logger.info("4. Stacking 集成")
    logger.info("=" * 60)

    stacking_model = StackingEnsemble()
    X_train, X_test, y_train, y_test = logistic_model.prepare_data(train_df)

    # 特征筛选（使用与基模型一致的特征）
    if USE_IV_SELECTION and logistic_model.feature_names:
        selected_features = logistic_model.feature_names
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]
        logger.info(f"  使用 IV 筛选特征: {len(selected_features)}")

    # 划分 train/valid
    X_train_sub, X_valid, y_train_sub, y_valid = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
    )

    # 训练 Stacking
    stacking_model.fit(X_train_sub, y_train_sub, X_valid, y_valid)

    # 评估
    stacking_proba = stacking_model.predict(X_test)
    stacking_pred = (stacking_proba >= 0.5).astype(int)

    from sklearn.metrics import roc_auc_score, roc_curve
    stacking_auc = roc_auc_score(y_test, stacking_proba)
    fpr, tpr, _ = roc_curve(y_test, stacking_proba)
    stacking_ks = float(max(tpr - fpr))

    stacking_metrics = {
        'AUC': stacking_auc,
        'KS': stacking_ks,
        'Gini': 2 * stacking_auc - 1,
        'Accuracy': float((stacking_pred == y_test).mean()),
    }

    logger.info(f"  Stacking AUC: {stacking_auc:.4f}")
    logger.info(f"  Stacking KS:  {stacking_ks:.4f}")

    stacking_model.save_model()

    # ========================================
    # 5. 模型对比
    # ========================================
    comparison = compare_models(logistic_metrics, xgb_metrics, lgb_metrics)
    comparison_path = REPORT_DIR / "model_comparison.csv"
    comparison.to_csv(comparison_path)
    print(f"模型对比结果已保存: {comparison_path}")

    # ========================================
    # 4. PSI 稳定性监控
    # ========================================
    print("\n" + "=" * 60)
    print("4. PSI 稳定性监控")
    print("=" * 60)

    psi_results = []
    for col in logistic_model.feature_names:
        if col in train_df.columns and col in test_df.columns:
            try:
                psi_val = calculate_psi(
                    train_df[col].dropna(),
                    test_df[col].dropna(),
                    bins=10
                )
                psi_results.append({'feature': col, 'psi': psi_val})
            except Exception:
                continue

    psi_df = pd.DataFrame(psi_results).sort_values('psi', ascending=False)

    unstable = psi_df[psi_df['psi'] > 0.25]
    marginal = psi_df[(psi_df['psi'] > 0.10) & (psi_df['psi'] <= 0.25)]
    stable = psi_df[psi_df['psi'] <= 0.10]

    print(f"\n特征 PSI 稳定性分析 (train vs test):")
    print(f"  稳定 (PSI <= 0.10):    {len(stable)} 个")
    print(f"  轻微漂移 (0.10 < PSI <= 0.25): {len(marginal)} 个")
    print(f"  显著漂移 (PSI > 0.25): {len(unstable)} 个")

    if len(unstable) > 0:
        print("\n  显著漂移特征 (PSI > 0.25):")
        for _, row in unstable.head(10).iterrows():
            print(f"    {row['feature']}: PSI={row['psi']:.4f}")

    psi_path = REPORT_DIR / "psi_monitoring.csv"
    psi_df.to_csv(psi_path, index=False)
    print(f"\n  PSI 结果已保存: {psi_path}")

    print("\n" + "=" * 70)
    logger.info("PD 模型训练、评估与正式预测全部完成")
    print("=" * 70)


if __name__ == "__main__":
    setup_logging()
    main()
