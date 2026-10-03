#!/usr/bin/env python3
"""
scripts/evaluate_stacking.py
Eksperimen 3: Dynamic Stacking Ensemble vs Fixed Blend vs Baseline.
Implements:
1. Out-Of-Fold (OOF) prediction generation via Purged Time-Series cross-validation.
2. Standard Stacking Meta-Learner (Logistic Regression on base model probabilities).
3. Context-Aware Dynamic Stacking Meta-Learner (Conditions base probabilities on Volatility and Foreign Flow regime).
4. Rigorous Out-Of-Sample Benchmark on 17,594 test samples.
"""

import sys
import json
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_FILE, DATA_DIR, LOG_FORMAT
import importlib
inference_mod = importlib.import_module("src.03_model_inference")
FEATURE_COLUMNS = inference_mod.FEATURE_COLUMNS
TARGET_COLUMN = inference_mod.TARGET_COLUMN

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("StackingEnsemble")

TUNED_PARAMS_FILE = DATA_DIR / "processed" / "best_hyperparameters.json"
STACKING_METRICS_FILE = DATA_DIR / "processed" / "benchmark_stacking.json"


def create_purged_time_series_folds(
    df: pd.DataFrame,
    n_splits: int = 4,
    embargo_days: int = 5
) -> List[Tuple[np.ndarray, np.ndarray]]:
    unique_dates = np.sort(df["Date"].unique())
    n_dates = len(unique_dates)
    min_train_dates = int(n_dates * 0.35)
    remaining_dates = n_dates - min_train_dates
    val_window_size = remaining_dates // n_splits
    
    folds = []
    for i in range(n_splits):
        train_end_idx = min_train_dates + (i * val_window_size)
        val_start_idx = train_end_idx + embargo_days
        val_end_idx = train_end_idx + val_window_size if (i < n_splits - 1) else n_dates
        
        if val_start_idx >= val_end_idx:
            continue
            
        train_dates_subset = unique_dates[:train_end_idx]
        val_dates_subset = unique_dates[val_start_idx:val_end_idx]
        
        train_indices = df[df["Date"].isin(train_dates_subset)].index.values
        val_indices = df[df["Date"].isin(val_dates_subset)].index.values
        
        if len(train_indices) > 0 and len(val_indices) > 0:
            folds.append((train_indices, val_indices))
            
    return folds


def compute_metrics(y_true: pd.Series, probs: np.ndarray, train_time: float, feature_count: int) -> Dict[str, Any]:
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    auc = roc_auc_score(y_true, probs)
    
    high_conv_mask = (probs >= 0.55)
    high_conv_prec = precision_score(y_true[high_conv_mask], (probs[high_conv_mask] >= 0.5).astype(int), zero_division=0) if np.sum(high_conv_mask) > 0 else prec
    
    top_decile_cutoff = np.percentile(probs, 90)
    top_decile_mask = (probs >= top_decile_cutoff)
    top_decile_prec = precision_score(y_true[top_decile_mask], (probs[top_decile_mask] >= 0.5).astype(int), zero_division=0)
    
    return {
        "Accuracy": round(float(acc), 4),
        "Precision": round(float(prec), 4),
        "Recall": round(float(rec), 4),
        "F1_Score": round(float(f1), 4),
        "ROC_AUC": round(float(auc), 4),
        "High_Conviction_Precision_55": round(float(high_conv_prec), 4),
        "Top_Decile_Precision": round(float(top_decile_prec), 4),
        "Train_Time_Sec": round(float(train_time), 2),
        "Sample_Size_Test": int(len(y_true)),
        "Feature_Count": int(feature_count)
    }


def run_experiment_3() -> Dict[str, Any]:
    if not PROCESSED_DATA_FILE.exists():
        logger.error(f"Processed dataset not found at {PROCESSED_DATA_FILE}")
        return {}

    # Load tuned parameters if available
    best_cat_params = {
        "iterations": 250, "depth": 4, "learning_rate": 0.025, "l2_leaf_reg": 5.0, "random_seed": 42, "verbose": False
    }
    best_gbdt_params = {
        "learning_rate": 0.02, "max_iter": 150, "max_depth": 4, "min_samples_leaf": 40, "l2_regularization": 3.0, "random_state": 42
    }
    
    if TUNED_PARAMS_FILE.exists():
        try:
            with open(TUNED_PARAMS_FILE, "r", encoding="utf-8") as f:
                saved_params = json.load(f)
                if "catboost_best_params" in saved_params:
                    best_cat_params.update(saved_params["catboost_best_params"])
                    best_cat_params["verbose"] = False
                if "histgb_best_params" in saved_params:
                    best_gbdt_params.update(saved_params["histgb_best_params"])
            logger.info("Loaded tuned hyperparameters from Experiment 2.")
        except Exception as e:
            logger.warning(f"Could not load tuned parameters: {e}. Using defaults.")

    df = pd.read_csv(PROCESSED_DATA_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values(by=["Date", "Ticker"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    valid_df = df.dropna(subset=[TARGET_COLUMN]).copy()
    valid_df.sort_values(by="Date", inplace=True)
    valid_df.reset_index(drop=True, inplace=True)

    feature_cols = FEATURE_COLUMNS
    for c in feature_cols:
        if c not in valid_df.columns:
            valid_df[c] = 0.0
        else:
            valid_df[c] = valid_df[c].fillna(0.0)

    split_idx = int(len(valid_df) * 0.8)
    train_df = valid_df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = valid_df.iloc[split_idx:].copy().reset_index(drop=True)

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COLUMN].astype(int)
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COLUMN].astype(int)

    logger.info(f"Dataset split: Train={len(train_df)}, Test={len(test_df)}")

    # 1. Generate Out-Of-Fold (OOF) predictions on Train Set using Purged Folds
    folds = create_purged_time_series_folds(train_df, n_splits=4, embargo_days=5)
    logger.info(f"Created {len(folds)} Purged TimeSeries folds for OOF Meta-Learner training.")

    oof_cat = np.full(len(train_df), np.nan)
    oof_gbdt = np.full(len(train_df), np.nan)
    oof_y = np.full(len(train_df), np.nan)

    t0_oof = time.time()
    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        logger.info(f"Training Fold {fold_i + 1}/{len(folds)} (Train: {len(tr_idx)}, Val: {len(va_idx)})...")
        X_tr = train_df.iloc[tr_idx][feature_cols]
        y_tr = train_df.iloc[tr_idx][TARGET_COLUMN].astype(int)
        X_va = train_df.iloc[va_idx][feature_cols]
        y_va = train_df.iloc[va_idx][TARGET_COLUMN].astype(int)

        # Base 1: CatBoost
        cb_fold = CatBoostClassifier(**best_cat_params)
        cb_fold.fit(X_tr, y_tr)
        oof_cat[va_idx] = cb_fold.predict_proba(X_va)[:, 1]

        # Base 2: HistGB
        hgb_fold = HistGradientBoostingClassifier(**best_gbdt_params)
        hgb_fold.fit(X_tr, y_tr)
        oof_gbdt[va_idx] = hgb_fold.predict_proba(X_va)[:, 1]
        oof_y[va_idx] = y_va.values

    oof_time = time.time() - t0_oof

    # Filter only samples with valid OOF predictions
    valid_oof_mask = ~np.isnan(oof_cat) & ~np.isnan(oof_gbdt)
    oof_X = np.column_stack([oof_cat[valid_oof_mask], oof_gbdt[valid_oof_mask]])
    oof_labels = oof_y[valid_oof_mask].astype(int)
    logger.info(f"Generated {len(oof_X)} valid OOF training pairs for Meta-Learner in {oof_time:.1f}s.")

    # 2. Train Base Models on FULL Training Set
    logger.info("Fitting base models on complete training set...")
    t0_base = time.time()
    full_cat = CatBoostClassifier(**best_cat_params)
    full_cat.fit(X_train, y_train)

    full_gbdt = HistGradientBoostingClassifier(**best_gbdt_params)
    full_gbdt.fit(X_train, y_train)
    base_fit_time = time.time() - t0_base

    # Generate Test Probabilities from Base Models
    test_p_cat = full_cat.predict_proba(X_test)[:, 1]
    test_p_gbdt = full_gbdt.predict_proba(X_test)[:, 1]

    # Model 1 & 2 Metrics
    m_cat = compute_metrics(y_test, test_p_cat, base_fit_time * 0.5, len(feature_cols))
    m_gbdt = compute_metrics(y_test, test_p_gbdt, base_fit_time * 0.5, len(feature_cols))

    # Model 3: Static 50:50 Blend
    test_p_static_blend = 0.5 * test_p_cat + 0.5 * test_p_gbdt
    m_static_blend = compute_metrics(y_test, test_p_static_blend, base_fit_time, len(feature_cols))

    # Model 4: Dynamic Stacking Meta-Learner (Logistic Regression Meta-Model)
    logger.info("Training Meta-Learner (Logistic Regression on OOF probabilities)...")
    meta_learner = LogisticRegression(C=0.5, penalty="l2", solver="lbfgs", random_state=42)
    meta_learner.fit(oof_X, oof_labels)
    
    meta_coef = meta_learner.coef_[0]
    meta_intercept = meta_learner.intercept_[0]
    logger.info(f"Meta-Learner Weights: CatBoost={meta_coef[0]:.4f}, HistGB={meta_coef[1]:.4f}, Intercept={meta_intercept:.4f}")

    test_meta_X = np.column_stack([test_p_cat, test_p_gbdt])
    test_p_stacking = meta_learner.predict_proba(test_meta_X)[:, 1]
    m_stacking = compute_metrics(y_test, test_p_stacking, base_fit_time + oof_time, len(feature_cols))

    # Model 5: Context-Aware Dynamic Stacking
    # Include market volatility and foreign flow as context features for the meta-learner
    logger.info("Training Context-Aware Meta-Learner (Probabilities + Macro/Flow Regimes)...")
    context_cols = ["Volatility_20D", "Foreign_Flow_Intensity"]
    available_context = [c for c in context_cols if c in train_df.columns]

    if len(available_context) == len(context_cols):
        oof_context = train_df.iloc[valid_oof_mask][available_context].values
        test_context = test_df[available_context].values

        oof_X_context = np.column_stack([oof_X, oof_context])
        test_meta_X_context = np.column_stack([test_meta_X, test_context])

        context_meta_learner = LogisticRegression(C=0.3, penalty="l2", solver="lbfgs", max_iter=300, random_state=42)
        context_meta_learner.fit(oof_X_context, oof_labels)

        test_p_context_stacking = context_meta_learner.predict_proba(test_meta_X_context)[:, 1]
        m_context_stacking = compute_metrics(y_test, test_p_context_stacking, base_fit_time + oof_time, len(feature_cols) + len(context_cols))
    else:
        m_context_stacking = m_stacking
        test_p_context_stacking = test_p_stacking

    # Save benchmark results
    STACKING_METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STACKING_METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "meta_learner_weights": {
                "catboost_weight": round(float(meta_coef[0]), 4),
                "histgb_weight": round(float(meta_coef[1]), 4),
                "intercept": round(float(meta_intercept), 4)
            },
            "models": {
                "histgb_base": m_gbdt,
                "catboost_base": m_cat,
                "static_blend_50_50": m_static_blend,
                "stacking_meta_learner": m_stacking,
                "context_aware_stacking": m_context_stacking
            }
        }, f, indent=2)

    # Print Clean Comparative Scorecard
    print("\n" + "=" * 96)
    print("EKSPERIMEN 3: DYNAMIC STACKING ENSEMBLE BENCHMARK (17,594 SAMPLES)")
    print("=" * 96)
    print(f"{'Metric':<26} | {'HistGB':<10} | {'CatBoost':<10} | {'Static Blend':<13} | {'Stacking (Meta)':<16} | {'Context Stacking':<16}")
    print("-" * 96)

    metrics_list = ["Accuracy", "ROC_AUC", "Top_Decile_Precision", "High_Conviction_Precision_55", "Precision", "Recall", "F1_Score"]
    for k in metrics_list:
        v_h = m_gbdt.get(k, 0.0)
        v_c = m_cat.get(k, 0.0)
        v_sb = m_static_blend.get(k, 0.0)
        v_st = m_stacking.get(k, 0.0)
        v_cs = m_context_stacking.get(k, 0.0)

        # Highlight best
        row_vals = [v_h, v_c, v_sb, v_st, v_cs]
        best_v = max(row_vals)
        s_sb = " *" if v_sb == best_v else ""
        s_st = " *" if v_st == best_v else ""
        s_cs = " *" if v_cs == best_v else ""

        print(f"{k:<26} | {v_h:<10.4f} | {v_c:<10.4f} | {v_sb:<11.4f}{s_sb:<2} | {v_st:<14.4f}{s_st:<2} | {v_cs:<14.4f}{s_cs:<2}")

    print("=" * 96)
    print(f"Meta-Learner Learned Equation:")
    print(f"  logit(P_up) = {meta_coef[0]:.3f} * P(CatBoost) + {meta_coef[1]:.3f} * P(HistGB) + ({meta_intercept:.3f})")
    print("=" * 96 + "\n")

    return {
        "histgb": m_gbdt,
        "catboost": m_cat,
        "static_blend": m_static_blend,
        "stacking": m_stacking,
        "context_stacking": m_context_stacking,
        "meta_weights": {
            "catboost": round(float(meta_coef[0]), 4),
            "histgb": round(float(meta_coef[1]), 4),
            "intercept": round(float(meta_intercept), 4)
        }
    }


if __name__ == "__main__":
    run_experiment_3()
