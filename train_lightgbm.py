"""Quick script to train LightGBM model only"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from sklearn.model_selection import train_test_split
from config import (
    PROCESSED_DATA_DIR, MODEL_DIR, REPORT_DIR, TARGET,
    RANDOM_STATE, USE_IV_SELECTION, IV_THRESHOLD, IV_TOP_N
)
from model_registry import register_model
from pd_model import PDModel

print("Loading data...")
train_df = pd.read_csv(PROCESSED_DATA_DIR / "train_with_features.csv")
test_df = pd.read_csv(PROCESSED_DATA_DIR / "test_with_features.csv")
print(f"Train: {train_df.shape[0]:,} x {train_df.shape[1]}")

print("\n=== LightGBM Training ===")
lgb_model = PDModel(model_type='lightgbm')
X_train, X_test, y_train, y_test = lgb_model.prepare_data(train_df)

# IV feature selection
if USE_IV_SELECTION:
    train_combined = X_train.copy()
    train_combined[TARGET] = y_train.values
    selected_features = lgb_model.select_features_by_iv(
        train_combined, TARGET, iv_threshold=IV_THRESHOLD, top_n=IV_TOP_N
    )
    if selected_features:
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]
        lgb_model.feature_names = selected_features
        print(f"  IV selection: {len(selected_features)} features")

# Split: train 70% / valid 15% / calib 15%
X_train_sub, X_temp, y_train_sub, y_temp = train_test_split(
    X_train, y_train, test_size=0.3, random_state=RANDOM_STATE, stratify=y_train
)
X_valid, X_cal, y_valid, y_cal = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
)

print("Training LightGBM with Optuna tuning (30 trials)...")
lgb_model.train_lightgbm_tuned(X_train_sub, y_train_sub, X_valid, y_valid, n_trials=30)

print("Calibrating...")
lgb_model.calibrate(X_cal, y_cal, method='isotonic')

print("Evaluating...")
lgb_metrics = lgb_model.evaluate(X_test, y_test)
print(f"  AUC: {lgb_metrics.get('AUC', 0):.4f}")
print(f"  KS:  {lgb_metrics.get('KS', 0):.4f}")
print(f"  Gini: {lgb_metrics.get('Gini', 0):.4f}")

print("Saving model...")
lgb_model.save_model()
lgb_model.save_test_predictions(test_df, REPORT_DIR / "test_predictions_lightgbm.csv")
lgb_model.save_model_report(REPORT_DIR / "lightgbm_model_report.md", dataset_name="train")

# Update model_comparison.csv
print("Updating model_comparison.csv...")
comp_path = PROCESSED_DATA_DIR / "model_comparison.csv"
if comp_path.exists():
    comp = pd.read_csv(comp_path)
    name_col = "Model" if "Model" in comp.columns else comp.columns[0]
    comp = comp[comp[name_col] != "LightGBM"]
else:
    comp = pd.DataFrame()
    name_col = "Model"

row = {
    name_col: "LightGBM",
    "AUC": round(lgb_metrics.get("AUC", 0), 6),
    "KS": round(lgb_metrics.get("KS", 0), 6),
    "Gini": round(lgb_metrics.get("Gini", 0), 6),
    "Accuracy": round(lgb_metrics.get("Accuracy", 0), 6),
    "Precision": round(lgb_metrics.get("Precision", 0), 6),
    "Recall": round(lgb_metrics.get("Recall", 0), 6),
    "F1-Score": round(lgb_metrics.get("F1-Score", 0), 6),
    "TP": lgb_metrics.get("TP", 0),
    "TN": lgb_metrics.get("TN", 0),
    "FP": lgb_metrics.get("FP", 0),
    "FN": lgb_metrics.get("FN", 0),
}
row[name_col] = "LightGBM"

comp = pd.concat([comp, pd.DataFrame([row])], ignore_index=True)
comp.to_csv(comp_path, index=False)
print(f"Saved model_comparison.csv with {len(comp)} models")
register_model(
    model_name="LightGBM",
    model_type="lightgbm",
    model_path=MODEL_DIR / "pd_model_lightgbm.pkl",
    feature_names=lgb_model.feature_names,
    metrics=lgb_metrics,
    dataset_rows=len(train_df),
    extra={"source": "train_lightgbm.py"},
)
print("\nDone!")
