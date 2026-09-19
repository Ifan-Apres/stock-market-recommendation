import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from statsmodels.tsa.arima.model import ARIMA
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    ARIMA_FORECAST_STEPS,
    DATA_DIR,
    DEFAULT_TICKERS,
    DIVIDEND_RECOMMENDATION_FILE,
    DIVIDEND_TICKERS,
    FAVORITE_TICKERS,
    FAVORITES_RECOMMENDATION_FILE,
    LOG_FORMAT,
    LSTM_LOOKBACK,
    PORTFOLIO_ALLOCATION_FILE,
    PROCESSED_DATA_FILE,
    SECTOR_MAP,
    SWING_RECOMMENDATION_FILE,
    SWING_TICKERS,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("ModelInference")

FEATURE_COLUMNS = [
    "Log_Return",
    "Dist_SMA_200",
    "Dist_SMA_20",
    "Volatility_20D",
    "EMA_12",
    "CMF_20",
    "Profit_Margin",
    "Bollinger_Upper",
    "Volume_Ratio",
    "Dividend_Yield",
    "MACD_Hist",
    "Volume_SMA_20",
    "Market_Cap",
    "PB_Ratio",
    "MACD_Signal",
]

LSTM_FEATURE_COLS = ["Return_1D", "RSI_14", "Dist_SMA_20", "CMF_20", "Volume_Ratio"]
TARGET_COLUMN = "Target_Class_5D"
MODEL_PATH = DATA_DIR / "processed" / "alpha_model.joblib"
LSTM_MODEL_PATH = DATA_DIR / "processed" / "lstm_model.pth"


# ==============================================================================
# 1. ARIMA Statistical Time-Series Predictor
# ==============================================================================
class ARIMAPredictor:
    @staticmethod
    def predict_ticker(close_series: pd.Series, steps: int = ARIMA_FORECAST_STEPS) -> Tuple[float, float]:
        clean_prices = close_series.dropna().values
        if len(clean_prices) < 30:
            return 0.0, 0.50

        recent_prices = clean_prices[-120:]
        current_price = recent_prices[-1]

        try:
            model = ARIMA(recent_prices, order=(1, 1, 1))
            fit_res = model.fit()
            forecast = fit_res.forecast(steps=steps)
            expected_future_price = float(forecast[-1])
            expected_return = (expected_future_price - current_price) / (current_price + 1e-9)
            prob = float(1.0 / (1.0 + np.exp(-25.0 * expected_return)))
            return round(expected_return, 4), round(np.clip(prob, 0.10, 0.90), 4)
        except Exception:
            try:
                model = ARIMA(recent_prices, order=(1, 0, 0))
                fit_res = model.fit()
                forecast = fit_res.forecast(steps=steps)
                expected_future_price = float(forecast[-1])
                expected_return = (expected_future_price - current_price) / (current_price + 1e-9)
                prob = float(1.0 / (1.0 + np.exp(-25.0 * expected_return)))
                return round(expected_return, 4), round(np.clip(prob, 0.10, 0.90), 4)
            except Exception:
                return 0.0, 0.50


# ==============================================================================
# 2. PyTorch Deep Learning LSTM Sequence Model
# ==============================================================================
class PyTorchLSTMNet(nn.Module):
    def __init__(self, input_dim: int = 5, hidden_dim: int = 32, num_layers: int = 2, dropout: float = 0.2):
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


class PyTorchLSTMTrainer:
    def __init__(self, lookback: int = LSTM_LOOKBACK):
        self.lookback = lookback
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PyTorchLSTMNet(input_dim=len(LSTM_FEATURE_COLS)).to(self.device)

    def prepare_sequences(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        sequences, labels = [], []
        for _, group in df.groupby("Ticker"):
            grp = group.sort_values(by="Date").copy()
            for col in LSTM_FEATURE_COLS:
                grp[col] = grp[col].fillna(0.0)

            features = grp[LSTM_FEATURE_COLS].values
            target = grp[TARGET_COLUMN].values

            if "RSI_14" in LSTM_FEATURE_COLS:
                rsi_idx = LSTM_FEATURE_COLS.index("RSI_14")
                features[:, rsi_idx] = features[:, rsi_idx] / 100.0

            for i in range(len(features) - self.lookback):
                seq_x = features[i : i + self.lookback]
                label_y = target[i + self.lookback]
                if not np.isnan(label_y):
                    sequences.append(seq_x)
                    labels.append(label_y)

        if not sequences:
            return torch.empty(0), torch.empty(0)

        return (
            torch.tensor(np.array(sequences), dtype=torch.float32),
            torch.tensor(np.array(labels), dtype=torch.float32).unsqueeze(1),
        )

    def train_lstm(self, df: pd.DataFrame, epochs: int = 6, batch_size: int = 256) -> float:
        X, y = self.prepare_sequences(df)
        if len(X) == 0:
            return 0.50

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005, weight_decay=1e-4)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        LSTM_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), LSTM_MODEL_PATH)
        return total_loss / len(loader)

    def predict_latest_ticker(self, ticker_group: pd.DataFrame) -> float:
        self.model.eval()
        grp = ticker_group.sort_values(by="Date").copy()
        for col in LSTM_FEATURE_COLS:
            grp[col] = grp[col].fillna(0.0)

        features = grp[LSTM_FEATURE_COLS].values
        if len(features) < self.lookback:
            return 0.50

        if "RSI_14" in LSTM_FEATURE_COLS:
            rsi_idx = LSTM_FEATURE_COLS.index("RSI_14")
            features[:, rsi_idx] = features[:, rsi_idx] / 100.0

        latest_window = features[-self.lookback :]
        input_tensor = torch.tensor(np.expand_dims(latest_window, axis=0), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            prob = self.model(input_tensor).item()
        return round(float(prob), 4)


# ==============================================================================
# 3. Multi-Engine Quantitative Alpha Model & Portfolio Optimizer
# ==============================================================================
class QuantitativeAlphaModel:
    """
    Multi-Engine Alpha Model with Actionable Technical Recommendations
    (Buy on Weakness, Buy on Breakout, Sell on Strength) & Portfolio Allocation Optimizer.
    """

    def __init__(self, processed_data_path: str = str(PROCESSED_DATA_FILE)):
        self.processed_data_path = processed_data_path
        self.gbdt_model = HistGradientBoostingClassifier(
            learning_rate=0.02,
            max_iter=150,
            max_depth=4,
            min_samples_leaf=40,
            l2_regularization=3.0,
            random_state=42,
        )
        self.lstm_trainer = PyTorchLSTMTrainer()
        self.arima_predictor = ARIMAPredictor()
        self.ticker_to_sector = {t: sector for sector, tickers in SECTOR_MAP.items() for t in tickers}

    def load_processed_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.processed_data_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values(by=["Date", "Ticker"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def train_and_evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        valid_df = df.dropna(subset=[TARGET_COLUMN]).copy()
        valid_df.sort_values(by="Date", inplace=True)

        for c in FEATURE_COLUMNS:
            if c not in valid_df.columns:
                valid_df[c] = 0.0

        split_idx = int(len(valid_df) * 0.8)
        train_df = valid_df.iloc[:split_idx]
        test_df = valid_df.iloc[split_idx:]

        X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN].astype(int)
        X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN].astype(int)

        logger.info("Fitting Tabular GBDT Model...")
        self.gbdt_model.fit(X_train, y_train)

        test_probs = self.gbdt_model.predict_proba(X_test)[:, 1]
        test_preds = (test_probs >= 0.5).astype(int)

        acc = accuracy_score(y_test, test_preds)
        prec = precision_score(y_test, test_preds, zero_division=0)
        auc = roc_auc_score(y_test, test_probs)

        logger.info("Fitting PyTorch LSTM Sequence Model...")
        self.lstm_trainer.train_lstm(train_df, epochs=6, batch_size=256)

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.gbdt_model, MODEL_PATH)

        return {
            "GBDT_Accuracy": round(float(acc), 4),
            "GBDT_Precision": round(float(prec), 4),
            "GBDT_ROC_AUC": round(float(auc), 4),
        }

    def _extract_latest_rows_with_ensemble(self, df: pd.DataFrame, ticker_filter: List[str]) -> pd.DataFrame:
        filtered_df = df[df["Ticker"].isin(ticker_filter)].copy()
        latest_records = []

        for ticker, group in filtered_df.groupby("Ticker"):
            grp = group.sort_values(by="Date")
            latest_row = grp.iloc[-1].to_dict()

            feat_vector = pd.DataFrame([latest_row])[FEATURE_COLUMNS]
            gbdt_prob = float(self.gbdt_model.predict_proba(feat_vector)[:, 1][0])
            arima_ret, arima_prob = self.arima_predictor.predict_ticker(grp["Close"], steps=5)
            lstm_prob = self.lstm_trainer.predict_latest_ticker(grp)

            blended_prob = round(0.50 * gbdt_prob + 0.30 * lstm_prob + 0.20 * arima_prob, 4)

            latest_row["GBDT_Prob"] = round(gbdt_prob, 4)
            latest_row["ARIMA_Prob"] = arima_prob
            latest_row["ARIMA_Expected_Return"] = arima_ret
            latest_row["LSTM_Prob"] = lstm_prob
            latest_row["Bullish_Probability"] = blended_prob
            latest_row["Conviction_Score"] = np.abs(blended_prob - 0.50)

            # Technical Action Signals
            rsi = latest_row.get("RSI_14", 50.0) or 50.0
            vol_ratio = latest_row.get("Volume_Ratio", 1.0) or 1.0
            close = latest_row.get("Close", 1000.0)
            res1 = latest_row.get("Resistance_1", close * 1.03) or (close * 1.03)
            sup1 = latest_row.get("Support_1", close * 0.97) or (close * 0.97)

            if blended_prob >= 0.52 and (rsi <= 45.0 or close <= sup1 * 1.01):
                action = "BUY ON WEAKNESS"
            elif blended_prob >= 0.52 and vol_ratio >= 1.25 and close >= res1 * 0.99:
                action = "BUY ON BREAKOUT"
            elif blended_prob >= 0.53:
                action = "TRADING BUY"
            elif blended_prob <= 0.46 or rsi >= 70.0:
                action = "SELL ON STRENGTH"
            else:
                action = "HOLD"

            # Entry, Target Price (TP), Stop Loss (SL), Risk-Reward
            entry_price = round(close, 0)
            target_price = round(close * 1.06, 0)  # TP1: +6%
            stop_loss = round(close * 0.96, 0)     # SL: -4%
            potential_gain = target_price - entry_price
            potential_risk = entry_price - stop_loss
            rr_ratio = round(potential_gain / (potential_risk + 1e-4), 2)

            latest_row["Recommendation"] = action
            latest_row["Sector"] = self.ticker_to_sector.get(ticker, "General")
            latest_row["Entry_Price"] = entry_price
            latest_row["Target_Price"] = target_price
            latest_row["Stop_Loss"] = stop_loss
            latest_row["Risk_Reward_Ratio"] = rr_ratio

            latest_records.append(latest_row)

        if not latest_records:
            return pd.DataFrame()

        latest_df = pd.DataFrame(latest_records)
        all_expected_cols = [
            "Beta_IHSG", "Sharpe_Ratio", "Max_Drawdown_1Y", "Annualized_Return_1Y",
            "Debt_to_Equity", "Current_Ratio", "RSI_14", "MACD_Hist", "MFI_14", "CMF_20",
            "Volatility_20D", "PE_Ratio", "PB_Ratio", "ROE", "Dividend_Yield",
            "GARCH_Vol", "VaR_95_1D", "Sector"
        ]
        for c in all_expected_cols:
            if c not in latest_df.columns:
                latest_df[c] = None

        return latest_df

    def generate_swing_recommendations(self, df: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
        logger.info("Generating Strategy 1: Daily Swing Trader (LQ45)...")
        latest_df = self._extract_latest_rows_with_ensemble(df, SWING_TICKERS)
        ranked_df = latest_df.sort_values(
            by=["Conviction_Score", "Bullish_Probability"], ascending=[False, False]
        ).head(top_n).reset_index(drop=True)

        ranked_df["Date"] = pd.to_datetime(ranked_df["Date"]).dt.strftime("%Y-%m-%d")
        SWING_RECOMMENDATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        ranked_df.to_csv(SWING_RECOMMENDATION_FILE, index=False)
        return ranked_df

    def generate_dividend_recommendations(self, df: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
        logger.info("Generating Strategy 2: Dividend & Value Investor...")
        latest_df = self._extract_latest_rows_with_ensemble(df, DIVIDEND_TICKERS)
        div_yield = latest_df["Dividend_Yield"].fillna(0.0) if "Dividend_Yield" in latest_df.columns else 0.0
        roe = latest_df["ROE"].fillna(0.0) if "ROE" in latest_df.columns else 0.0
        der = latest_df["Debt_to_Equity"].fillna(1.0) if "Debt_to_Equity" in latest_df.columns else 1.0

        latest_df["Dividend_Score"] = (
            div_yield * 0.4 + roe * 100.0 * 0.25 + latest_df["Bullish_Probability"] * 25.0 - der * 5.0
        )
        ranked_df = latest_df.sort_values(by=["Dividend_Score", "Dividend_Yield"], ascending=[False, False]).head(top_n).reset_index(drop=True)
        ranked_df["Date"] = pd.to_datetime(ranked_df["Date"]).dt.strftime("%Y-%m-%d")
        DIVIDEND_RECOMMENDATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        ranked_df.to_csv(DIVIDEND_RECOMMENDATION_FILE, index=False)
        return ranked_df

    def generate_favorites_recommendations(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Generating Strategy 3: 10 Favorite Stocks Portfolio...")
        latest_df = self._extract_latest_rows_with_ensemble(df, FAVORITE_TICKERS)
        ranked_df = latest_df.sort_values(by="Bullish_Probability", ascending=False).reset_index(drop=True)
        ranked_df["Date"] = pd.to_datetime(ranked_df["Date"]).dt.strftime("%Y-%m-%d")
        FAVORITES_RECOMMENDATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        ranked_df.to_csv(FAVORITES_RECOMMENDATION_FILE, index=False)
        return ranked_df

    def optimize_portfolio_allocation(self, df: pd.DataFrame, capital_idr: float = 50000000.0) -> pd.DataFrame:
        """
        Optimal Portfolio Weighting Algorithm (Sharpe-Weighted & Risk Parity with Cash Reserve).
        Calculates exact % allocation and nominal Rupiah amount per stock.
        """
        logger.info(f"Computing optimal portfolio allocation for total capital: Rp {capital_idr:,.0f}...")
        latest_df = self._extract_latest_rows_with_ensemble(df, DEFAULT_TICKERS)

        # Select candidates with positive conviction
        buy_candidates = latest_df[
            latest_df["Recommendation"].isin(["BUY ON WEAKNESS", "BUY ON BREAKOUT", "TRADING BUY"])
        ].copy()

        if len(buy_candidates) < 3:
            # Fallback to top bullish probability
            buy_candidates = latest_df.sort_values(by="Bullish_Probability", ascending=False).head(5).copy()
        else:
            buy_candidates = buy_candidates.sort_values(by="Bullish_Probability", ascending=False).head(5)

        # Calculate Risk-Parity & Sharpe Weights
        sharpe_raw = buy_candidates["Sharpe_Ratio"].fillna(0.5)
        sharpe_safe = np.clip(sharpe_raw, 0.1, 5.0)
        garch_vol = buy_candidates["GARCH_Vol"].fillna(0.30)

        # Score = Sharpe / Volatility
        raw_scores = (sharpe_safe / (garch_vol + 0.05)).values
        equity_weight_budget = 0.80  # 80% Equity Allocation, 20% Cash Reserve for risk management
        weights = (raw_scores / np.sum(raw_scores)) * equity_weight_budget

        # Cap single stock weight at 25% for institutional risk diversification
        weights = np.clip(weights, 0.08, 0.25)
        actual_equity_sum = np.sum(weights)
        cash_weight = round(1.0 - actual_equity_sum, 4)

        allocation_records = []
        for i, (_, row) in enumerate(buy_candidates.iterrows()):
            w = round(float(weights[i]), 4)
            nominal = round(capital_idr * w, 0)
            close = row["Close"]
            num_shares = int((nominal / close) // 100) * 100  # Rounded to IDX 100-shares lot
            actual_nominal = num_shares * close

            allocation_records.append({
                "Ticker": row["Ticker"],
                "Sector": row.get("Sector", "General"),
                "Recommendation": row["Recommendation"],
                "Allocation_Pct": round(w * 100.0, 1),
                "Nominal_IDR": actual_nominal if actual_nominal > 0 else nominal,
                "Shares_Lot": num_shares // 100,
                "Close_Price": close,
                "Target_Price": row["Target_Price"],
                "Stop_Loss": row["Stop_Loss"],
                "Bullish_Probability": row["Bullish_Probability"],
            })

        # Add Cash Reserve Record
        allocation_records.append({
            "Ticker": "KAS SIAGA (CASH)",
            "Sector": "Money Market / Risk Reserve",
            "Recommendation": "RESERVE",
            "Allocation_Pct": round(cash_weight * 100.0, 1),
            "Nominal_IDR": round(capital_idr * cash_weight, 0),
            "Shares_Lot": 0,
            "Close_Price": 1.0,
            "Target_Price": 1.0,
            "Stop_Loss": 1.0,
            "Bullish_Probability": 0.50,
        })

        alloc_df = pd.DataFrame(allocation_records)
        PORTFOLIO_ALLOCATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        alloc_df.to_csv(PORTFOLIO_ALLOCATION_FILE, index=False)
        logger.info(f"Portfolio allocation saved to {PORTFOLIO_ALLOCATION_FILE}")
        return alloc_df


def run_model_inference_pipeline() -> Tuple[Dict[str, float], pd.DataFrame]:
    model_engine = QuantitativeAlphaModel()
    df = model_engine.load_processed_data()
    metrics = model_engine.train_and_evaluate(df)

    swing_recs = model_engine.generate_swing_recommendations(df)
    model_engine.generate_dividend_recommendations(df)
    model_engine.generate_favorites_recommendations(df)
    model_engine.optimize_portfolio_allocation(df, capital_idr=50000000.0)

    return metrics, swing_recs


if __name__ == "__main__":
    run_model_inference_pipeline()
