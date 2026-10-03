#!/usr/bin/env python3
"""
scripts/tune_hyperparameters.py
Eksperimen 2: Hyperparameter Optimization with Optuna (Bayesian Optimization TPE)
Using Purged Time-Series Cross Validation with a 5-day embargo.
Optimizes both CatBoost and HistGradientBoosting on train split, then evaluates on out-of-sample test split.
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
from catboost import CatBoostClassifier
import optuna

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_FILE, DATA_DIR, LOG_FORMAT
import importlib
inference_mod = importlib.import_module("src.03_model_inference")
FEATURE_COLUMNS = inference_mod.FEATURE_COLUMNS
TARGET_COLUMN = inference_mod.TARGET_COLUMN

# Suppress Optuna info logs during trials for clean output
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("HyperparameterTuner")

TUNED_PARAMS_FILE = DATA_DIR / "processed" / "best_hyperparameters.json"
TUNED_METRICS_FILE = DATA_DIR / "processed" / "benchmark_tuned.json"
CATBOOST_METRICS_FILE = DATA_DIR / "processed" / "benchmark_catboost.json"


def create_purged_time_series_folds(
    df: pd.DataFrame,
    n_splits: int = 3,
    embargo_days: int = 5
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Creates Purged TimeSeries cross-validation folds with an embargo period
    to avoid data leakage from the 5-day forward target.
    """
    unique_dates = np.sort(df["Date"].unique())
    n_dates = len(unique_dates)
    
    # Reserve initial 40% of dates for initial training window
    min_train_dates = int(n_dates * 0.40)
    remaining_dates = n_dates - min_train_dates
    val_window_size = remaining_dates // n_splits
    
    folds = []
    for i in range(n_splits):
        train_end_idx = min_train_dates + (i * val_window_size)
        val_start_idx = train_end_idx + embargo_days  # Embargo gap
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


def optimize_catboost(
    train_df: pd.DataFrame,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    feature_cols: list,
    n_trials: int = 25
) -> Tuple[Dict[str, Any], float]:
    logger.info(f"Starting CatBoost Hyperparameter Optimization ({n_trials} trials, {len(folds)} folds)...")
    
    def objective(trial: optuna.Trial) -> float:
        depth = trial.suggest_int("depth", 3, 6)
        learning_rate = trial.suggest_float("learning_rate", 0.015, 0.07, log=True)
        l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 2.0, 15.0)
        subsample = trial.suggest_float("subsample", 0.65, 0.95)
        random_strength = trial.suggest_float("random_strength", 0.5, 4.0)
        
        fold_scores = []
        for train_idx, val_idx in folds:
            X_tr = train_df.iloc[train_idx][feature_cols]
            y_tr = train_df.iloc[train_idx][TARGET_COLUMN].astype(int)
            X_va = train_df.iloc[val_idx][feature_cols]
            y_va = train_df.iloc[val_idx][TARGET_COLUMN].astype(int)
            
            clf = CatBoostClassifier(
                iterations=180,
                depth=depth,
                learning_rate=learning_rate,
                l2_leaf_reg=l2_leaf_reg,
                subsample=subsample,
                random_strength=random_strength,
                random_seed=42,
                verbose=False
            )
            clf.fit(X_tr, y_tr)
            probs = clf.predict_proba(X_va)[:, 1]
            fold_scores.append(roc_auc_score(y_va, probs))
            
        return float(np.mean(fold_scores))
    
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    
    best_params = study.best_params
    best_params["iterations"] = 250  # scale up iterations for full fit
    logger.info(f"CatBoost Best CV ROC-AUC: {study.best_value:.4f}")
    logger.info(f"CatBoost Best Params: {best_params}")
    return best_params, study.best_value


def optimize_histgb(
    train_df: pd.DataFrame,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    feature_cols: list,
    n_trials: int = 25
) -> Tuple[Dict[str, Any], float]:
    logger.info(f"Starting HistGB Hyperparameter Optimization ({n_trials} trials, {len(folds)} folds)...")
    
    def objective(trial: optuna.Trial) -> float:
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.05, log=True)
        max_iter = trial.suggest_int("max_iter", 100, 220, step=20)
        max_depth = trial.suggest_int("max_depth", 3, 5)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 30, 90, step=10)
        l2_reg = trial.suggest_float("l2_regularization", 1.0, 10.0)
        
        fold_scores = []
        for train_idx, val_idx in folds:
            X_tr = train_df.iloc[train_idx][feature_cols]
            y_tr = train_df.iloc[train_idx][TARGET_COLUMN].astype(int)
            X_va = train_df.iloc[val_idx][feature_cols]
            y_va = train_df.iloc[val_idx][TARGET_COLUMN].astype(int)
            
            clf = HistGradientBoostingClassifier(
                learning_rate=learning_rate,
                max_iter=max_iter,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                l2_regularization=l2_reg,
                random_state=42
            )
            clf.fit(X_tr, y_tr)
            probs = clf.predict_proba(X_va)[:, 1]
            fold_scores.append(roc_auc_score(y_va, probs))
            
        return float(np.mean(fold_scores))
    
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    
    best_params = study.best_params
    logger.info(f"HistGB Best CV ROC-AUC: {study.best_value:.4f}")
    logger.info(f"HistGB Best Params: {best_params}")
    return best_params, study.best_value


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


def run_experiment_2(n_trials: int = 25) -> Dict[str, Any]:
    if not PROCESSED_DATA_FILE.exists():
        logger.error(f"Processed dataset not found at {PROCESSED_DATA_FILE}")
        return {}

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

    logger.info(f"Dataset split: Train={len(train_df)} rows, Test={len(test_df)} rows.")

    # 1. Create Purged Time Series folds for tuning
    folds = create_purged_time_series_folds(train_df, n_splits=3, embargo_days=5)
    logger.info(f"Generated {len(folds)} Purged Time-Series folds with 5-day embargo.")

    # 2. Optimize CatBoost
    t0 = time.time()
    best_cat_params, cat_cv_score = optimize_catboost(train_df, folds, feature_cols, n_trials=n_trials)
    cat_tune_time = time.time() - t0

    # 3. Optimize HistGB
    t0 = time.time()
    best_gbdt_params, gbdt_cv_score = optimize_histgb(train_df, folds, feature_cols, n_trials=n_trials)
    gbdt_tune_time = time.time() - t0

    # 4. Train Tuned Models on Full Train Set & Evaluate on Out-Of-Sample Test Set
    logger.info("Training tuned models on full training set...")
    
    # Tuned CatBoost
    t0 = time.time()
    tuned_cat = CatBoostClassifier(
        **best_cat_params,
        random_seed=42,
        verbose=False
    )
    tuned_cat.fit(X_train, y_train)
    cat_fit_time = time.time() - t0
    tuned_cat_probs = tuned_cat.predict_proba(X_test)[:, 1]
    tuned_cat_metrics = compute_metrics(y_test, tuned_cat_probs, cat_fit_time, len(feature_cols))

    # Tuned HistGB
    t0 = time.time()
    tuned_gbdt = HistGradientBoostingClassifier(
        **best_gbdt_params,
        random_state=42
    )
    tuned_gbdt.fit(X_train, y_train)
    gbdt_fit_time = time.time() - t0
    tuned_gbdt_probs = tuned_gbdt.predict_proba(X_test)[:, 1]
    tuned_gbdt_metrics = compute_metrics(y_test, tuned_gbdt_probs, gbdt_fit_time, len(feature_cols))

    # Tuned Blend Ensemble (50% Tuned CatBoost + 50% Tuned HistGB)
    tuned_blend_probs = 0.5 * tuned_cat_probs + 0.5 * tuned_gbdt_probs
    tuned_blend_metrics = compute_metrics(y_test, tuned_blend_probs, cat_fit_time + gbdt_fit_time, len(feature_cols))

    # Load Experiment 1 baseline / defaults for direct comparison
    exp1_data = {}
    if CATBOOST_METRICS_FILE.exists():
        with open(CATBOOST_METRICS_FILE, "r", encoding="utf-8") as f:
            exp1_data = json.load(f)

    default_gbdt = exp1_data.get("gbdt", {})
    default_cat = exp1_data.get("catboost", {})
    default_blend = exp1_data.get("blend_ensemble", {})

    # Save Best Hyperparameters
    TUNED_PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TUNED_PARAMS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "catboost_best_params": best_cat_params,
            "catboost_cv_auc": round(float(cat_cv_score), 4),
            "histgb_best_params": best_gbdt_params,
            "histgb_cv_auc": round(float(gbdt_cv_score), 4),
            "tuning_time_sec": round(cat_tune_time + gbdt_tune_time, 2)
        }, f, indent=2)

    # Save Tuned Metrics
    with open(TUNED_METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "tuned_histgb": tuned_gbdt_metrics,
            "tuned_catboost": tuned_cat_metrics,
            "tuned_blend": tuned_blend_metrics,
            "comparison_with_exp1": {
                "default_histgb": default_gbdt,
                "default_catboost": default_cat,
                "default_blend": default_blend
            }
        }, f, indent=2)

    # Print Clean Comparative Scorecard
    print("\n" + "=" * 90)
    print("EKSPERIMEN 2: HYPERPARAMETER TUNING BENCHMARK (OPTUNA TPE + PURGED CV)")
    print("=" * 90)
    print(f"{'Metric':<26} | {'HistGB (Def)':<13} | {'HistGB (Tune)':<13} | {'Cat (Def)':<11} | {'Cat (Tune)':<11} | {'Blend (Tune)':<12}")
    print("-" * 90)

    metrics_list = ["Accuracy", "ROC_AUC", "Top_Decile_Precision", "High_Conviction_Precision_55", "Precision", "F1_Score", "Train_Time_Sec"]
    for k in metrics_list:
        v_h_def = default_gbdt.get(k, 0.0)
        v_h_tune = tuned_gbdt_metrics.get(k, 0.0)
        v_c_def = default_cat.get(k, 0.0)
        v_c_tune = tuned_cat_metrics.get(k, 0.0)
        v_b_tune = tuned_blend_metrics.get(k, 0.0)
        
        # Mark best
        row_vals = [v_h_def, v_h_tune, v_c_def, v_c_tune, v_b_tune]
        best_v = max(row_vals) if k != "Train_Time_Sec" else min(row_vals)
        s_b = " *" if v_b_tune == best_v else ""
        s_c = " *" if v_c_tune == best_v else ""

        print(f"{k:<26} | {v_h_def:<13.4f} | {v_h_tune:<13.4f} | {v_c_def:<11.4f} | {v_c_tune:<9.4f}{s_c:<2} | {v_b_tune:<10.4f}{s_b:<2}")

    print("=" * 90)
    print(f"CatBoost Best Params: {best_cat_params}")
    print(f"HistGB Best Params  : {best_gbdt_params}")
    print("=" * 90 + "\n")

    return {
        "tuned_histgb": tuned_gbdt_metrics,
        "tuned_catboost": tuned_cat_metrics,
        "tuned_blend": tuned_blend_metrics,
        "best_cat_params": best_cat_params,
        "best_gbdt_params": best_gbdt_params
    }


if __name__ == "__main__":
    trials = 25
    if len(sys.argv) > 1:
        trials = int(sys.argv[1])
    run_experiment_2(n_trials=trials)
