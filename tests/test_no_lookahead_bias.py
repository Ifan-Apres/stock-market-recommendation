"""
Unit & Invariant Tests: Lookahead Bias & Data Leakage Prevention.
Mathematically guarantees that features at time t strictly use historical data <= t,
and future market perturbations t+1..t+k cannot leak into features at t.
"""

import importlib
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
feature_eng_mod = importlib.import_module("src.02_feature_eng")
QuantitativeFeatureEngineer = feature_eng_mod.QuantitativeFeatureEngineer

inference_mod = importlib.import_module("src.03_model_inference")
FEATURE_COLUMNS = inference_mod.FEATURE_COLUMNS
TARGET_COLUMN = inference_mod.TARGET_COLUMN
DATA_DIR = ROOT_DIR / "data" / "processed"


class TestNoLookaheadBias(unittest.TestCase):
    """Institutional Anti-Leakage & Causal Integrity Test Suite."""

    def setUp(self):
        self.engineer = QuantitativeFeatureEngineer()

    def test_feature_causality_with_future_perturbations(self):
        """
        Gold Standard Lookahead Bias Test:
        Calculate features on data up to day t=60.
        Then inject extreme price/volume/macro shocks into days 61..100.
        Verify that features at day t=60 are 100% invariant (delta < 1e-6).
        """
        np.random.seed(42)
        dates = pd.date_range("2026-01-01", periods=100, freq="B")
        base_price = 5000.0
        returns = np.random.normal(0.0005, 0.015, size=100)
        prices = base_price * np.exp(np.cumsum(returns))

        ticker_df = pd.DataFrame({
            "Date": dates,
            "Ticker": "BBCA.JK",
            "Open": prices * 0.995,
            "High": prices * 1.01,
            "Low": prices * 0.99,
            "Close": prices,
            "Adj Close": prices,
            "Volume": np.random.randint(1_000_000, 10_000_000, size=100),
            "Value": prices * np.random.randint(1_000_000, 10_000_000, size=100),
            "Foreign_Sell": np.random.randint(100_000, 500_000, size=100),
            "Foreign_Buy": np.random.randint(100_000, 500_000, size=100),
            "Net_Foreign_IDR": np.random.normal(0, 1e8, size=100),
        })

        bench_close = 7200.0 * np.exp(np.cumsum(np.random.normal(0.0002, 0.008, size=100)))
        bench_df = pd.DataFrame({
            "Date": dates,
            "Close": bench_close,
            "Adj Close": bench_close,
            "Benchmark_Return_1D": pd.Series(bench_close).pct_change(1).fillna(0.0),
            "Benchmark_Return_5D": np.log(pd.Series(bench_close).shift(-5) / pd.Series(bench_close)),
        })

        macro_df = pd.DataFrame({
            "Date": dates,
            "US_10Y_Yield_Delta": np.random.normal(0, 0.02, size=100),
            "USD_IDR_Return_1D": np.random.normal(0, 0.003, size=100),
            "Brent_Oil_Return_1D": np.random.normal(0, 0.01, size=100),
            "SP500_Return_1D": np.random.normal(0, 0.008, size=100),
        })

        # Cutoff day t = 60
        t_cutoff = 60
        cutoff_date = dates[t_cutoff]

        # Baseline: generate features using strictly unperturbed data up to t=60
        clean_features = self.engineer.generate_ticker_features(ticker_df.copy(), bench_df.copy(), macro_df.copy())
        t_row_clean = clean_features[clean_features["Date"] == cutoff_date].iloc[0]

        # Perturbed future: Inject extreme 5x price jumps, 50x volume spikes, and macro flash crash at t > 60
        perturbed_ticker = ticker_df.copy()
        perturbed_ticker.loc[t_cutoff + 1:, "Close"] *= 5.0
        perturbed_ticker.loc[t_cutoff + 1:, "Adj Close"] *= 5.0
        perturbed_ticker.loc[t_cutoff + 1:, "High"] *= 6.0
        perturbed_ticker.loc[t_cutoff + 1:, "Low"] *= 4.5
        perturbed_ticker.loc[t_cutoff + 1:, "Volume"] *= 50.0

        perturbed_macro = macro_df.copy()
        perturbed_macro.loc[t_cutoff + 1:, "US_10Y_Yield_Delta"] += 0.25
        perturbed_macro.loc[t_cutoff + 1:, "USD_IDR_Return_1D"] += 0.05
        perturbed_macro.loc[t_cutoff + 1:, "Brent_Oil_Return_1D"] += 0.10
        perturbed_macro.loc[t_cutoff + 1:, "SP500_Return_1D"] -= 0.08

        perturbed_bench = bench_df.copy()
        perturbed_bench.loc[t_cutoff + 1:, "Close"] *= 1.5
        perturbed_bench.loc[t_cutoff + 1:, "Benchmark_Return_1D"] += 0.05

        perturbed_features = self.engineer.generate_ticker_features(perturbed_ticker, perturbed_bench, perturbed_macro)
        t_row_perturbed = perturbed_features[perturbed_features["Date"] == cutoff_date].iloc[0]

        # Invariant checks: None of the features at t=60 can be altered by future t > 60 data!
        test_feature_keys = [
            "SMA_20",
            "SMA_50",
            "Dist_SMA_20",
            "Dist_SMA_50",
            "EMA_12",
            "EMA_26",
            "MACD",
            "MACD_Signal",
            "MACD_Hist",
            "RSI_14",
            "Volatility_20D",
            "daily_finbert_score",
            "sentiment_ema_3",
            "sentiment_ema_7",
            "macro_shock_flag",
            "vol_sentiment_interaction",
        ]

        for feat in test_feature_keys:
            val_clean = float(t_row_clean[feat])
            val_perturbed = float(t_row_perturbed[feat])
            delta = abs(val_clean - val_perturbed)
            self.assertLess(
                delta,
                1e-6,
                f"LOOKAHEAD LEAKAGE DETECTED in '{feat}'! Clean={val_clean}, Perturbed={val_perturbed}, Delta={delta}",
            )

    def test_target_variable_future_shift_guard(self):
        """
        Verify forward target returns (Target_Return_5D) strictly use future data,
        meaning the final 5 trading days must have NaN targets (unobservable future).
        """
        processed_file = DATA_DIR / "processed_market_features.csv"
        self.assertTrue(processed_file.exists())

        df = pd.read_csv(processed_file, usecols=["Date", "Ticker", "Target_Class_5D", "Target_Return_5D"])
        df["Date"] = pd.to_datetime(df["Date"])

        # Check for each emiten, the last 5 trading days must have NaN target
        for ticker in ["BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "ASII.JK"]:
            sub = df[df["Ticker"] == ticker].sort_values(by="Date")
            if len(sub) >= 10:
                last_5_classes = sub["Target_Class_5D"].iloc[-5:]
                last_5_returns = sub["Target_Return_5D"].iloc[-5:]
                self.assertTrue(
                    last_5_classes.isna().all(),
                    f"Target_Class_5D is not NaN in the final 5 observations for {ticker}! Lookahead leakage risk."
                )
                self.assertTrue(
                    last_5_returns.isna().all(),
                    f"Target_Return_5D is not NaN in the final 5 observations for {ticker}! Lookahead leakage risk."
                )

    def test_target_exclusion_from_feature_columns(self):
        """Verify target variables are strictly excluded from FEATURE_COLUMNS."""
        forbidden_targets = [
            TARGET_COLUMN,
            "Target_Return_5D",
            "Target_Excess_Return_5D",
            "Target_Class_5D",
            "Next_Pivot_Point",
            "Next_Support_1",
            "Next_Resistance_1",
        ]
        for forbidden in forbidden_targets:
            self.assertNotIn(
                forbidden,
                FEATURE_COLUMNS,
                f"Critical leakage: '{forbidden}' is included in FEATURE_COLUMNS!",
            )


if __name__ == "__main__":
    unittest.main()
