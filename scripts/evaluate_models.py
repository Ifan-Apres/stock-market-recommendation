#!/usr/bin/env python3
"""
scripts/evaluate_models.py
Quantitative Model Evaluation & Comparative Benchmark Harness.
Compares HistGradientBoosting (Baseline) vs CatBoost (Candidate) vs Blend Ensemble.
"""

import sys
import json
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from catboost import CatBoostClassifier

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_FILE, DATA_DIR, LOG_FORMAT
import importlib
inference_mod = importlib.import_module("src.03_model_inference")
FEATURE_COLUMNS = inference_mod.FEATURE_COLUMNS
TARGET_COLUMN = inference_mod.TARGET_COLUMN

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("ModelEvaluator")

BASELINE_METRICS_FILE = DATA_DIR / "processed" / "benchmark_baseline.json"
CANDIDATE_METRICS_FILE = DATA_DIR / "processed" / "benchmark_candidate.json"
CATBOOST_METRICS_FILE = DATA_DIR / "processed" / "benchmark_catboost.json"


def evaluate_single_model(
    model_name: str,
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_cols: list
) -> Tuple[Dict[str, float], np.ndarray, Dict[str, float]]:
    logger.info(f"Training {model_name} on {len(X_train)} samples with {len(feature_cols)} features...")
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, test_preds)
    prec = precision_score(y_test, test_preds, zero_division=0)
    rec = recall_score(y_test, test_preds, zero_division=0)
    f1 = f1_score(y_test, test_preds, zero_division=0)
    auc = roc_auc_score(y_test, test_probs)

    # High Conviction (Threshold >= 0.55)
    high_conv_mask = (test_probs >= 0.55)
    high_conv_prec = precision_score(y_test[high_conv_mask], (test_probs[high_conv_mask] >= 0.5).astype(int), zero_division=0) if np.sum(high_conv_mask) > 0 else prec

    # Top Decile (Top 10% highest conviction predictions)
    top_decile_cutoff = np.percentile(test_probs, 90)
    top_decile_mask = (test_probs >= top_decile_cutoff)
    top_decile_prec = precision_score(y_test[top_decile_mask], (test_probs[top_decile_mask] >= 0.5).astype(int), zero_division=0)

    metrics = {
        "Accuracy": round(float(acc), 4),
        "Precision": round(float(prec), 4),
        "Recall": round(float(rec), 4),
        "F1_Score": round(float(f1), 4),
        "ROC_AUC": round(float(auc), 4),
        "High_Conviction_Precision_55": round(float(high_conv_prec), 4),
        "Top_Decile_Precision": round(float(top_decile_prec), 4),
        "Train_Time_Sec": round(float(train_time), 2),
        "Sample_Size_Test": int(len(X_test)),
        "Feature_Count": int(len(feature_cols))
    }

    # Feature importances
    feature_importance = {}
    if hasattr(model, "feature_importances_"):
        raw_imp = model.feature_importances_
        feature_importance = {feature_cols[i]: round(float(raw_imp[i]), 5) for i in np.argsort(raw_imp)[::-1]}
    elif hasattr(model, "get_feature_importance"):
        raw_imp = model.get_feature_importance()
        feature_importance = {feature_cols[i]: round(float(raw_imp[i]), 5) for i in np.argsort(raw_imp)[::-1]}

    return metrics, test_probs, feature_importance


def run_experiment_1() -> Dict[str, Any]:
    if not PROCESSED_DATA_FILE.exists():
        logger.error(f"Processed dataset not found at {PROCESSED_DATA_FILE}")
        return {}

    df = pd.read_csv(PROCESSED_DATA_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values(by=["Date", "Ticker"], inplace=True)

    valid_df = df.dropna(subset=[TARGET_COLUMN]).copy()
    valid_df.sort_values(by="Date", inplace=True)

    feature_cols = FEATURE_COLUMNS
    for c in feature_cols:
        if c not in valid_df.columns:
            valid_df[c] = 0.0
        else:
            valid_df[c] = valid_df[c].fillna(0.0)

    split_idx = int(len(valid_df) * 0.8)
    train_df = valid_df.iloc[:split_idx]
    test_df = valid_df.iloc[split_idx:]

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COLUMN].astype(int)
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COLUMN].astype(int)

    # 1. Model A: HistGradientBoosting (Current Baseline)
    gbdt_model = HistGradientBoostingClassifier(
        learning_rate=0.02,
        max_iter=150,
        max_depth=4,
        min_samples_leaf=40,
        l2_regularization=3.0,
        random_state=42,
    )
    gbdt_metrics, gbdt_probs, _ = evaluate_single_model(
        "HistGradientBoosting", gbdt_model, X_train, y_train, X_test, y_test, feature_cols
    )

    # 2. Model B: CatBoost (Candidate Experiment 1)
    cat_model = CatBoostClassifier(
        iterations=250,
        learning_rate=0.03,
        depth=5,
        l2_leaf_reg=5.0,
        random_seed=42,
        verbose=False,
    )
    cat_metrics, cat_probs, cat_imp = evaluate_single_model(
        "CatBoostClassifier", cat_model, X_train, y_train, X_test, y_test, feature_cols
    )

    # 3. Model C: Blend Ensemble (50% HistGB + 50% CatBoost)
    blend_probs = 0.5 * gbdt_probs + 0.5 * cat_probs
    blend_preds = (blend_probs >= 0.5).astype(int)
    blend_acc = accuracy_score(y_test, blend_preds)
    blend_prec = precision_score(y_test, blend_preds, zero_division=0)
    blend_rec = recall_score(y_test, blend_preds, zero_division=0)
    blend_f1 = f1_score(y_test, blend_preds, zero_division=0)
    blend_auc = roc_auc_score(y_test, blend_probs)
    
    top_decile_cutoff = np.percentile(blend_probs, 90)
    top_decile_mask = (blend_probs >= top_decile_cutoff)
    blend_top_decile = precision_score(y_test[top_decile_mask], (blend_probs[top_decile_mask] >= 0.5).astype(int), zero_division=0)

    high_conv_mask = (blend_probs >= 0.55)
    blend_high_conv = precision_score(y_test[high_conv_mask], (blend_probs[high_conv_mask] >= 0.5).astype(int), zero_division=0) if np.sum(high_conv_mask) > 0 else blend_prec

    blend_metrics = {
        "Accuracy": round(float(blend_acc), 4),
        "Precision": round(float(blend_prec), 4),
        "Recall": round(float(blend_rec), 4),
        "F1_Score": round(float(blend_f1), 4),
        "ROC_AUC": round(float(blend_auc), 4),
        "High_Conviction_Precision_55": round(float(blend_high_conv), 4),
        "Top_Decile_Precision": round(float(blend_top_decile), 4),
        "Train_Time_Sec": round(gbdt_metrics["Train_Time_Sec"] + cat_metrics["Train_Time_Sec"], 2),
        "Sample_Size_Test": int(len(X_test)),
        "Feature_Count": int(len(feature_cols))
    }

    # Save CatBoost metrics
    CATBOOST_METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CATBOOST_METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "gbdt": gbdt_metrics,
            "catboost": cat_metrics,
            "blend_ensemble": blend_metrics,
            "catboost_feature_importance": cat_imp
        }, f, indent=2)

    # Print Clean Comparative Report
    print("\n" + "=" * 80)
    print("EKSPERIMEN 1: HISTGRADIENTBOOSTING VS CATBOOST VS BLEND ENSEMBLE")
    print("=" * 80)
    print(f"{'Metric':<28} | {'HistGB (Base)':<14} | {'CatBoost (Cand)':<15} | {'Blend (50:50)':<13}")
    print("-" * 80)

    for k in ["Accuracy", "ROC_AUC", "Top_Decile_Precision", "High_Conviction_Precision_55", "Precision", "F1_Score", "Train_Time_Sec"]:
        v_base = gbdt_metrics.get(k, 0.0)
        v_cat = cat_metrics.get(k, 0.0)
        v_blend = blend_metrics.get(k, 0.0)
        
        # Highlight best in row
        best_val = max(v_base, v_cat, v_blend) if k != "Train_Time_Sec" else min(v_base, v_cat, v_blend)
        star_cat = " *" if v_cat == best_val else ""
        star_blend = " *" if v_blend == best_val else ""
        
        print(f"{k:<28} | {v_base:<14.4f} | {v_cat:<13.4f}{star_cat:<2} | {v_blend:<11.4f}{star_blend:<2}")

    print("=" * 80)
    print("Top 10 Most Important Features in CatBoost:")
    for i, (feat, imp) in enumerate(list(cat_imp.items())[:10]):
        print(f" {i+1:2d}. {feat:<28} : {imp:.2f}%")
    print("=" * 80 + "\n")

    return {
        "gbdt": gbdt_metrics,
        "catboost": cat_metrics,
        "blend": blend_metrics
    }


if __name__ == "__main__":
    run_experiment_1()
