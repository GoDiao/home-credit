"""Train Stacking Ensemble model with tuned base model hyperparams"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from config import (
    PROCESSED_DATA_DIR, MODEL_DIR, TARGET,
    RANDOM_STATE, USE_IV_SELECTION
)
from model_registry import register_model
from pd_model import StackingEnsemble, PDModel

print("Loading data...")
train_df = pd.read_csv(PROCESSED_DATA_DIR / "train_with_features.csv")
test_df = pd.read_csv(PROCESSED_DATA_DIR / "test_with_features.csv")
print(f"Train: {train_df.shape[0]:,} x {train_df.shape[1]}")

print("\n=== Stacking Ensemble Training (with tuned params) ===")
base = PDModel(model_type='logistic')
X_train, X_test, y_train, y_test = base.prepare_data(train_df)

if USE_IV_SELECTION and base.feature_names:
    selected_features = base.feature_names
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    print(f"  Using IV-selected features: {len(selected_features)}")

X_train_sub, X_valid, y_train_sub, y_valid = train_test_split(
    X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
)
print(f"  Train: {len(X_train_sub):,}, Valid: {len(X_valid):,}")

# Extract tuned LightGBM params from saved model
print("Extracting tuned LightGBM params...")
lgb_pkg = joblib.load(MODEL_DIR / 'pd_model_lightgbm.pkl')
lgb_tuned = {k: v for k, v in lgb_pkg['model'].params.items()
             if k in ('num_leaves', 'max_depth', 'learning_rate', 'feature_fraction',
                      'bagging_fraction', 'bagging_freq', 'min_child_samples',
                      'reg_alpha', 'reg_lambda')}
print(f"  LightGBM tuned params: {lgb_tuned}")

# XGBoost: run a quick Optuna search
print("Running XGBoost Optuna (15 trials)...")
xgb_model = PDModel(model_type='xgboost')
xgb_model.feature_names = base.feature_names
xgb_best = xgb_model.tune_xgboost(X_train_sub, y_train_sub, X_valid, y_valid, n_trials=15)
print(f"  XGBoost best params: {xgb_best}")

base_configs = [
    {'type': 'logistic'},
    {'type': 'xgboost', 'params': xgb_best},
    {'type': 'lightgbm', 'params': lgb_tuned},
]

print(f"\nTraining Stacking (3 base models x 5 folds, tuned params)...")
stacking_model = StackingEnsemble()
stacking_model.fit(X_train_sub, y_train_sub, X_valid, y_valid, base_configs=base_configs)

print("\nEvaluating...")
stacking_proba = stacking_model.predict(X_test)

auc = roc_auc_score(y_test, stacking_proba)
fpr, tpr, thresholds = roc_curve(y_test, stacking_proba)
ks_vals = tpr - fpr
ks_idx = np.argmax(ks_vals)
optimal_threshold = float(thresholds[ks_idx])
ks = float(max(ks_vals))
gini = 2 * auc - 1

print(f"  Optimal threshold (KS): {optimal_threshold:.4f}")
stacking_pred = (stacking_proba >= optimal_threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, stacking_pred).ravel()
total = tp + tn + fp + fn
accuracy = (tp + tn) / total
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"  AUC:       {auc:.4f}")
print(f"  KS:        {ks:.4f}")
print(f"  Gini:      {gini:.4f}")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")
print(f"  TP: {tp}  FP: {fp}  FN: {fn}  TN: {tn}")

print("Saving model...")
stacking_model.last_summary.update({
    'metrics': {
        'AUC': float(auc),
        'KS': float(ks),
        'Gini': float(gini),
        'Accuracy': float(accuracy),
        'Precision': float(precision),
        'Recall': float(recall),
        'F1-Score': float(f1),
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
    },
    'total_samples': int(len(train_df)),
    'evaluation_threshold': optimal_threshold,
})
stacking_model.save_model()

# Update model_comparison.csv
print("Updating model_comparison.csv...")
comp_path = PROCESSED_DATA_DIR / "model_comparison.csv"
if comp_path.exists():
    comp = pd.read_csv(comp_path)
    name_col = "Model" if "Model" in comp.columns else comp.columns[0]
    comp = comp[comp[name_col] != "Stacking Ensemble"]
else:
    comp = pd.DataFrame()
    name_col = "Model"

row = {
    name_col: "Stacking Ensemble",
    "AUC": round(auc, 6),
    "KS": round(ks, 6),
    "Gini": round(gini, 6),
    "Accuracy": round(accuracy, 6),
    "Precision": round(precision, 6),
    "Recall": round(recall, 6),
    "F1-Score": round(f1, 6),
    "TP": tp, "TN": tn, "FP": fp, "FN": fn,
}

comp = pd.concat([comp, pd.DataFrame([row])], ignore_index=True)
comp.to_csv(comp_path, index=False)
print(f"Saved model_comparison.csv with {len(comp)} models")
register_model(
    model_name="Stacking Ensemble",
    model_type="stacking",
    model_path=MODEL_DIR / "pd_model_stacking.pkl",
    feature_names=stacking_model.feature_names,
    metrics=row,
    dataset_rows=len(train_df),
    extra={"evaluation_threshold": optimal_threshold, "source": "train_stacking.py"},
)
print("\nDone!")
