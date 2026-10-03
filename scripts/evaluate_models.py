#!/usr/bin/env python3
"""
scripts/evaluate_models.py
Quantitative Model Evaluation & Comparative Benchmark Harness.
Compares Baseline (Main) vs Candidate (RND) Machine Learning Models.
"""

import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier

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


def evaluate_tabular_features(df: pd.DataFrame, feature_cols: list) -> Tuple[Dict[str, float], Dict[str, float]]:
    valid_df = df.dropna(subset=[TARGET_COLUMN]).copy()
    valid_df.sort_values(by="Date", inplace=True)

    # Fill any missing feature cols with median or 0.0
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

    logger.info(f"Training HistGradientBoosting on {len(X_train)} samples with {len(feature_cols)} features...")
    model = HistGradientBoostingClassifier(
        learning_rate=0.02,
        max_iter=150,
        max_depth=4,
        min_samples_leaf=40,
        l2_regularization=3.0,
        random_state=42,
    )
    model.fit(X_train, y_train)

    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, test_preds)
    prec = precision_score(y_test, test_preds, zero_division=0)
    rec = recall_score(y_test, test_preds, zero_division=0)
    f1 = f1_score(y_test, test_preds, zero_division=0)
    auc = roc_auc_score(y_test, test_probs)

    # High Conviction (Threshold >= 0.55)
    high_conv_mask = (test_probs >= 0.55)
    if np.sum(high_conv_mask) > 0:
        high_conv_prec = precision_score(y_test[high_conv_mask], (test_probs[high_conv_mask] >= 0.5).astype(int), zero_division=0)
    else:
        high_conv_prec = prec

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
        "Sample_Size_Test": int(len(X_test)),
        "Feature_Count": int(len(feature_cols))
    }

    # Calculate permutation-based feature importance
    from sklearn.inspection import permutation_importance
    perm_imp = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    feature_importance = {}
    for idx in np.argsort(perm_imp.importances_mean)[::-1]:
        feature_importance[feature_cols[idx]] = round(float(perm_imp.importances_mean[idx]), 5)

    return metrics, feature_importance


def run_benchmark(save_as_candidate: bool = True) -> Dict[str, Any]:
    if not PROCESSED_DATA_FILE.exists():
        logger.error(f"Processed dataset not found at {PROCESSED_DATA_FILE}")
        return {}

    df = pd.read_csv(PROCESSED_DATA_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values(by=["Date", "Ticker"], inplace=True)

    metrics, importance = evaluate_tabular_features(df, FEATURE_COLUMNS)

    target_file = CANDIDATE_METRICS_FILE if save_as_candidate else BASELINE_METRICS_FILE
    target_file.parent.mkdir(parents=True, exist_ok=True)
    with open(target_file, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "feature_importance": importance}, f, indent=2)

    logger.info(f"Metrics saved to {target_file}")

    # If baseline exists, print comparison report
    if BASELINE_METRICS_FILE.exists() and save_as_candidate:
        with open(BASELINE_METRICS_FILE, "r", encoding="utf-8") as f:
            base_data = json.load(f)
            base_metrics = base_data.get("metrics", {})

        print("\n" + "=" * 70)
        print("QUANTITATIVE BENCHMARK: BASELINE (MAIN) VS CANDIDATE (RND)")
        print("=" * 70)
        print(f"{'Metric':<30} | {'Baseline':<12} | {'Candidate':<12} | {'Delta':<10} | {'Verdict'}")
        print("-" * 70)

        for k in ["Accuracy", "ROC_AUC", "Top_Decile_Precision", "High_Conviction_Precision_55", "Precision", "F1_Score"]:
            b_val = base_metrics.get(k, 0.0)
            c_val = metrics.get(k, 0.0)
            delta = c_val - b_val
            delta_str = f"{delta:+.4f}"
            if delta > 0.002:
                verdict = "[+] OUTPERFORMS"
            elif delta < -0.002:
                verdict = "[-] UNDERPERFORMS"
            else:
                verdict = "[=] PAR"
            print(f"{k:<30} | {b_val:<12.4f} | {c_val:<12.4f} | {delta_str:<10} | {verdict}")

        print("=" * 70)
        print("Top 10 Most Influential Features in Candidate Model:")
        for i, (feat, imp) in enumerate(list(importance.items())[:10]):
            print(f" {i+1:2d}. {feat:<28} : {imp:+.5f}")
        print("=" * 70 + "\n")

    return metrics


if __name__ == "__main__":
    is_base = "--baseline" in sys.argv
    run_benchmark(save_as_candidate=not is_base)
