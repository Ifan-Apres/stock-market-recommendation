#!/usr/bin/env python3
"""
scripts/evaluate_bilstm.py
Eksperimen 4: Deep Learning Sequence Architecture Benchmark
Compares:
1. Standard Unidirectional LSTM (Baseline)
2. Bidirectional LSTM (BiLSTM with Dual-Endpoint Representation)
3. Temporal Attention LSTM (Attention-LSTM with dynamic time-step weighting)
4. Multi-Engine Consensus Blends (Trees + DL Engine)
Evaluated on out-of-sample test split sequences.
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

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_FILE, DATA_DIR, LOG_FORMAT, LSTM_LOOKBACK
import importlib
inference_mod = importlib.import_module("src.03_model_inference")
FEATURE_COLUMNS = inference_mod.FEATURE_COLUMNS
LSTM_FEATURE_COLS = inference_mod.LSTM_FEATURE_COLS
TARGET_COLUMN = inference_mod.TARGET_COLUMN

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("DeepLearningBenchmark")

DEEP_LEARNING_METRICS_FILE = DATA_DIR / "processed" / "benchmark_deeplearning.json"
PREV_BENCHMARK_FILE = DATA_DIR / "processed" / "benchmark_catboost.json"


# ==============================================================================
# Model Architectures
# ==============================================================================

class StandardLSTMNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 2, dropout: float = 0.35):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]
        return self.sigmoid(self.fc(last_step))


class BiLSTMNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 2, dropout: float = 0.35):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Dual-Endpoint Representation: forward terminal (T) + backward terminal (0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        fwd_final = out[:, -1, :self.hidden_dim]
        bwd_final = out[:, 0, self.hidden_dim:]
        combined = torch.cat([fwd_final, bwd_final], dim=1)
        return self.sigmoid(self.fc(combined))


class AttentionLSTMNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 2, dropout: float = 0.35):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        attn_scores = self.attention(out)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(out * attn_weights, dim=1)
        return self.sigmoid(self.fc(context))


# ==============================================================================
# Training & Sequence Preparation Harness
# ==============================================================================

def prepare_temporal_sequences(
    df: pd.DataFrame,
    lookback: int = LSTM_LOOKBACK
) -> Tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
    """
    Constructs 3D chronological temporal windows per ticker.
    Also returns metadata dataframe aligned with sequences for multi-engine matching.
    """
    sequences = []
    labels = []
    meta_rows = []

    for ticker, group in df.groupby("Ticker"):
        grp = group.sort_values(by="Date").copy().reset_index(drop=True)
        for col in LSTM_FEATURE_COLS:
            if col not in grp.columns:
                grp[col] = 0.0
            else:
                grp[col] = grp[col].fillna(0.0)

        features = grp[LSTM_FEATURE_COLS].values
        target = grp[TARGET_COLUMN].values

        if "RSI_14" in LSTM_FEATURE_COLS:
            rsi_idx = LSTM_FEATURE_COLS.index("RSI_14")
            features[:, rsi_idx] = features[:, rsi_idx] / 100.0

        for i in range(len(features) - lookback):
            label_y = target[i + lookback]
            if not np.isnan(label_y):
                sequences.append(features[i : i + lookback])
                labels.append(label_y)
                # Store reference row for aligning with tabular tree models
                meta_rows.append(grp.iloc[i + lookback])

    if not sequences:
        return torch.empty(0), torch.empty(0), pd.DataFrame()

    X_tensor = torch.tensor(np.array(sequences), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(labels), dtype=torch.float32).unsqueeze(1)
    meta_df = pd.DataFrame(meta_rows).reset_index(drop=True)
    return X_tensor, y_tensor, meta_df


def train_dl_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    device: torch.device,
    epochs: int = 6,
    batch_size: int = 256,
    lr: float = 0.005
) -> Tuple[nn.Module, float, float]:
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    t0 = time.time()
    model.train()
    final_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        final_loss = epoch_loss / max(1, n_batches)

    train_time = time.time() - t0
    return model, final_loss, train_time


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, train_time: float, param_count: int) -> Dict[str, Any]:
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
        "Parameters": int(param_count),
        "Sample_Size_Test": int(len(y_true))
    }


def run_experiment_4() -> Dict[str, Any]:
    if not PROCESSED_DATA_FILE.exists():
        logger.error(f"Processed dataset not found at {PROCESSED_DATA_FILE}")
        return {}

    logger.info("Loading market feature dataset...")
    df = pd.read_csv(PROCESSED_DATA_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values(by=["Date", "Ticker"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    valid_df = df.dropna(subset=[TARGET_COLUMN]).copy().reset_index(drop=True)

    # Chronological Split (80% Train, 20% Test)
    split_idx = int(len(valid_df) * 0.8)
    train_df = valid_df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = valid_df.iloc[split_idx:].copy().reset_index(drop=True)

    logger.info("Preparing 3D temporal sequences (Lookback = 30 days)...")
    X_train_seq, y_train_seq, _ = prepare_temporal_sequences(train_df, lookback=LSTM_LOOKBACK)
    X_test_seq, y_test_seq, meta_test = prepare_temporal_sequences(test_df, lookback=LSTM_LOOKBACK)

    y_test_arr = y_test_seq.numpy().flatten().astype(int)
    logger.info(f"Sequence dataset: Train={len(X_train_seq)}, Test={len(X_test_seq)} sequences.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = len(LSTM_FEATURE_COLS)

    # ---------------------------------------------------------
    # 1. Standard Unidirectional LSTM (Baseline)
    # ---------------------------------------------------------
    torch.manual_seed(42)
    std_lstm = StandardLSTMNet(input_dim=input_dim, hidden_dim=32, num_layers=2, dropout=0.35).to(device)
    param_std = sum(p.numel() for p in std_lstm.parameters() if p.requires_grad)
    logger.info(f"Training Standard LSTM ({param_std} parameters)...")
    std_lstm, loss_std, time_std = train_dl_model(std_lstm, X_train_seq, y_train_seq, device, epochs=6)

    std_lstm.eval()
    with torch.no_grad():
        probs_std = std_lstm(X_test_seq.to(device)).cpu().numpy().flatten()
    m_std = compute_metrics(y_test_arr, probs_std, time_std, param_std)

    # ---------------------------------------------------------
    # 2. Bidirectional LSTM (BiLSTM - Candidate 1)
    # ---------------------------------------------------------
    torch.manual_seed(42)
    bilstm = BiLSTMNet(input_dim=input_dim, hidden_dim=32, num_layers=2, dropout=0.35).to(device)
    param_bi = sum(p.numel() for p in bilstm.parameters() if p.requires_grad)
    logger.info(f"Training Bidirectional LSTM ({param_bi} parameters)...")
    bilstm, loss_bi, time_bi = train_dl_model(bilstm, X_train_seq, y_train_seq, device, epochs=6)

    bilstm.eval()
    with torch.no_grad():
        probs_bi = bilstm(X_test_seq.to(device)).cpu().numpy().flatten()
    m_bi = compute_metrics(y_test_arr, probs_bi, time_bi, param_bi)

    # ---------------------------------------------------------
    # 3. Attention LSTM (Attention-LSTM - Candidate 2)
    # ---------------------------------------------------------
    torch.manual_seed(42)
    attn_lstm = AttentionLSTMNet(input_dim=input_dim, hidden_dim=32, num_layers=2, dropout=0.35).to(device)
    param_attn = sum(p.numel() for p in attn_lstm.parameters() if p.requires_grad)
    logger.info(f"Training Attention LSTM ({param_attn} parameters)...")
    attn_lstm, loss_attn, time_attn = train_dl_model(attn_lstm, X_train_seq, y_train_seq, device, epochs=6)

    attn_lstm.eval()
    with torch.no_grad():
        probs_attn = attn_lstm(X_test_seq.to(device)).cpu().numpy().flatten()
    m_attn = compute_metrics(y_test_arr, probs_attn, time_attn, param_attn)

    # ---------------------------------------------------------
    # 4. Multi-Engine Blends (Trees + Neural Network)
    # ---------------------------------------------------------
    logger.info("Evaluating Multi-Engine Integration on Sequence-Aligned Test Set...")
    # Train winning Tree Ensemble on train_df tabular features
    X_tr_tab = train_df[FEATURE_COLUMNS].fillna(0.0)
    y_tr_tab = train_df[TARGET_COLUMN].astype(int)
    X_te_tab = meta_test[FEATURE_COLUMNS].fillna(0.0)

    # HistGB
    hgb = HistGradientBoostingClassifier(learning_rate=0.02, max_iter=150, max_depth=4, min_samples_leaf=40, l2_regularization=3.0, random_state=42)
    hgb.fit(X_tr_tab, y_tr_tab)
    p_hgb = hgb.predict_proba(X_te_tab)[:, 1]

    # CatBoost
    cb = CatBoostClassifier(iterations=250, learning_rate=0.03, depth=5, l2_leaf_reg=5.0, random_seed=42, verbose=False)
    cb.fit(X_tr_tab, y_tr_tab)
    p_cb = cb.predict_proba(X_te_tab)[:, 1]

    # Winning Tree Blend (50:50)
    p_tree_blend = 0.5 * p_hgb + 0.5 * p_cb
    m_tree_blend = compute_metrics(y_test_arr, p_tree_blend, 3.15, 0)

    # Multi-Engine with Standard LSTM (60% Tree Blend + 40% DL)
    p_me_std = 0.60 * p_tree_blend + 0.40 * probs_std
    m_me_std = compute_metrics(y_test_arr, p_me_std, 3.15 + time_std, param_std)

    # Multi-Engine with BiLSTM (60% Tree Blend + 40% BiLSTM)
    p_me_bi = 0.60 * p_tree_blend + 0.40 * probs_bi
    m_me_bi = compute_metrics(y_test_arr, p_me_bi, 3.15 + time_bi, param_bi)

    # Multi-Engine with Attention-LSTM (60% Tree Blend + 40% Attention-LSTM)
    p_me_attn = 0.60 * p_tree_blend + 0.40 * probs_attn
    m_me_attn = compute_metrics(y_test_arr, p_me_attn, 3.15 + time_attn, param_attn)

    # Save benchmark metrics to JSON
    DEEP_LEARNING_METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DEEP_LEARNING_METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "deep_learning_standalone": {
                "standard_lstm": m_std,
                "bilstm": m_bi,
                "attention_lstm": m_attn
            },
            "multi_engine_consensus": {
                "tree_blend_only": m_tree_blend,
                "multi_engine_std_lstm": m_me_std,
                "multi_engine_bilstm": m_me_bi,
                "multi_engine_attention_lstm": m_me_attn
            }
        }, f, indent=2)

    # Print Clean Scorecards
    print("\n" + "=" * 92)
    print("EKSPERIMEN 4A: DEEP LEARNING ARCHITECTURES (STANDALONE SEQUENCE MODELS)")
    print("=" * 92)
    print(f"{'Metric':<26} | {'Standard LSTM':<16} | {'BiLSTM (Dual)':<16} | {'Attention-LSTM':<16}")
    print("-" * 92)

    for k in ["Accuracy", "ROC_AUC", "Top_Decile_Precision", "High_Conviction_Precision_55", "Precision", "F1_Score", "Train_Time_Sec", "Parameters"]:
        v_s = m_std.get(k, 0.0)
        v_b = m_bi.get(k, 0.0)
        v_a = m_attn.get(k, 0.0)

        best_v = max(v_s, v_b, v_a) if k not in ["Train_Time_Sec", "Parameters"] else min(v_s, v_b, v_a)
        s_s = " *" if v_s == best_v else ""
        s_b = " *" if v_b == best_v else ""
        s_a = " *" if v_a == best_v else ""

        if k in ["Parameters"]:
            print(f"{k:<26} | {int(v_s):<16d} | {int(v_b):<16d} | {int(v_a):<16d}")
        else:
            print(f"{k:<26} | {v_s:<14.4f}{s_s:<2} | {v_b:<14.4f}{s_b:<2} | {v_a:<14.4f}{s_a:<2}")

    print("=" * 92)

    print("\n" + "=" * 105)
    print("EKSPERIMEN 4B: MULTI-ENGINE INTEGRATION (60% TREE BLEND + 40% DL ENGINE)")
    print("=" * 105)
    print(f"{'Metric':<26} | {'Tree Blend Alone':<18} | {'ME + Std LSTM':<17} | {'ME + BiLSTM':<17} | {'ME + Attn LSTM':<17}")
    print("-" * 105)

    for k in ["Accuracy", "ROC_AUC", "Top_Decile_Precision", "High_Conviction_Precision_55", "Precision", "F1_Score"]:
        v_t = m_tree_blend.get(k, 0.0)
        v_ms = m_me_std.get(k, 0.0)
        v_mb = m_me_bi.get(k, 0.0)
        v_ma = m_me_attn.get(k, 0.0)

        best_v = max(v_t, v_ms, v_mb, v_ma)
        s_t = " *" if v_t == best_v else ""
        s_ms = " *" if v_ms == best_v else ""
        s_mb = " *" if v_mb == best_v else ""
        s_ma = " *" if v_ma == best_v else ""

        print(f"{k:<26} | {v_t:<16.4f}{s_t:<2} | {v_ms:<15.4f}{s_ms:<2} | {v_mb:<15.4f}{s_mb:<2} | {v_ma:<15.4f}{s_ma:<2}")

    print("=" * 105 + "\n")

    return {
        "standalone": {
            "standard_lstm": m_std,
            "bilstm": m_bi,
            "attention_lstm": m_attn
        },
        "multi_engine": {
            "tree_blend": m_tree_blend,
            "me_std_lstm": m_me_std,
            "me_bilstm": m_me_bi,
            "me_attn_lstm": m_me_attn
        }
    }


if __name__ == "__main__":
    run_experiment_4()
