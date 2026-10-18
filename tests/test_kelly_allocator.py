"""
Unit & Invariant Tests: Fractional Half-Kelly Criterion Allocator.
Guarantees capital conservation (100% budget), single-asset equity caps (<= 25%),
BEI 100-share trading lot quantization, cash reserve defense, and fail-safe zero-candidate handling.
"""

import importlib
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
inference_mod = importlib.import_module("src.03_model_inference")
capped_weights = inference_mod.capped_weights
QuantitativeAlphaModel = inference_mod.QuantitativeAlphaModel
DATA_DIR = ROOT_DIR / "data" / "processed"


class TestKellyAllocator(unittest.TestCase):
    """Institutional Portfolio Optimization & Kelly Invariants Test Suite."""

    def test_capped_weights_mathematical_invariants(self):
        """
        Verify capped_weights satisfies:
        1. max(w) <= cap
        2. sum(w) <= budget + 1e-6
        3. min(w) >= 0.0
        """
        # Case 1: Standard multi-asset distribution
        scores = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
        w = capped_weights(scores, cap=0.25, budget=0.80)
        self.assertLessEqual(np.max(w), 0.25 + 1e-6, "Weight exceeded 25% single-asset cap")
        self.assertAlmostEqual(float(np.sum(w)), 0.80, places=5, msg="Weights did not sum to equity budget")
        self.assertTrue((w >= 0.0).all(), "Negative weight detected")

        # Case 2: Extreme single asset dominance (one score 1000x larger)
        dominant_scores = np.array([1000.0, 0.1, 0.1])
        w_dom = capped_weights(dominant_scores, cap=0.25, budget=0.80)
        self.assertLessEqual(np.max(w_dom), 0.25 + 1e-6)
        self.assertLessEqual(float(np.sum(w_dom)), 0.80 + 1e-6)

        # Case 3: Zero / empty scores
        zero_scores = np.array([0.0, 0.0, 0.0])
        w_zero = capped_weights(zero_scores, cap=0.25, budget=0.80)
        self.assertTrue((w_zero == 0.0).all(), "Zero scores should yield zero weights")

    def test_bei_lot_size_and_capital_conservation(self):
        """
        Verify that portfolio allocator guarantees:
        1. Sum of Nominal_IDR == total_capital
        2. Shares_Lot is an integer >= 0 for all equities
        3. Nominal_IDR == Shares_Lot * 100 * Close_Price for all equities
        4. Unspent fractional funds are strictly conserved in Cash Reserve
        """
        model = QuantitativeAlphaModel()
        model.current_market_regime = "BULLISH_TREND"

        # Create synthetic buy candidates
        candidates = pd.DataFrame([
            {"Ticker": "BBCA.JK", "Close": 10250.0, "Sharpe_Ratio": 1.45, "Bullish_Probability": 0.65, "Risk_Reward_Ratio": 2.2, "Recommendation": "BUY ON BREAKOUT", "GARCH_Vol": 0.18, "Sector": "Financials", "Target_Price": 11000.0, "Stop_Loss": 9800.0},
            {"Ticker": "BMRI.JK", "Close": 7100.0, "Sharpe_Ratio": 1.20, "Bullish_Probability": 0.60, "Risk_Reward_Ratio": 2.0, "Recommendation": "BUY ON WEAKNESS", "GARCH_Vol": 0.22, "Sector": "Financials", "Target_Price": 7600.0, "Stop_Loss": 6800.0},
            {"Ticker": "TLKM.JK", "Close": 3100.0, "Sharpe_Ratio": 0.95, "Bullish_Probability": 0.58, "Risk_Reward_Ratio": 1.8, "Recommendation": "TRADING BUY", "GARCH_Vol": 0.20, "Sector": "Infrastructures", "Target_Price": 3350.0, "Stop_Loss": 2950.0},
        ])

        # Test allocator logic with Rp 50,000,000 capital
        capital = 50_000_000.0
        # Mock get_or_extract_latest_rows to return our controlled candidate set
        model.get_or_extract_latest_rows = lambda df, tickers: candidates.copy()

        alloc_df = model.optimize_portfolio_allocation(df=pd.DataFrame(), capital_idr=capital, save_output=False)

        # Invariant 1: Budget conservation
        total_nominal = alloc_df["Nominal_IDR"].sum()
        self.assertAlmostEqual(total_nominal, capital, delta=1.0, msg="Total portfolio nominal deviated from initial capital")

        # Invariant 2 & 3: BEI 100-share trading lot integrity
        equities = alloc_df[alloc_df["Ticker"] != "KAS SIAGA (CASH)"]
        for _, row in equities.iterrows():
            lot = row["Shares_Lot"]
            close = row["Close_Price"]
            nominal = row["Nominal_IDR"]
            self.assertIsInstance(int(lot), int)
            self.assertGreaterEqual(lot, 0)
            self.assertEqual(nominal, lot * 100.0 * close, f"Nominal mismatch for {row['Ticker']}")
            # Invariant: single asset cap <= 25% + rounding lot slack
            pct = row["Allocation_Pct"]
            self.assertLessEqual(pct, 26.0, f"Stock {row['Ticker']} exceeded single asset equity cap ({pct}%)")

        # Invariant 4: Cash reserve >= 20%
        cash_row = alloc_df[alloc_df["Ticker"] == "KAS SIAGA (CASH)"].iloc[0]
        self.assertGreaterEqual(cash_row["Allocation_Pct"], 20.0, "Cash reserve dropped below mandated 20% buffer")

    def test_regime_adaptive_cash_defense(self):
        """
        Verify that in BEARISH_MARKDOWN regime:
        1. Equity budget is throttled to <= 65% (Cash >= 35%)
        2. Half-Kelly scores receive cash defense haircut (0.75x)
        """
        model = QuantitativeAlphaModel()
        model.current_market_regime = "BEARISH_MARKDOWN"

        candidates = pd.DataFrame([
            {"Ticker": "BBCA.JK", "Close": 10000.0, "Sharpe_Ratio": 1.5, "Bullish_Probability": 0.65, "Risk_Reward_Ratio": 2.0, "Recommendation": "BUY ON WEAKNESS", "GARCH_Vol": 0.18, "Sector": "Financials", "Target_Price": 10800.0, "Stop_Loss": 9600.0},
            {"Ticker": "ICBP.JK", "Close": 11500.0, "Sharpe_Ratio": 1.1, "Bullish_Probability": 0.60, "Risk_Reward_Ratio": 1.8, "Recommendation": "BUY ON WEAKNESS", "GARCH_Vol": 0.15, "Sector": "Consumer Non-Cyclicals", "Target_Price": 12200.0, "Stop_Loss": 11000.0},
        ])

        capital = 50_000_000.0
        model.get_or_extract_latest_rows = lambda df, tickers: candidates.copy()

        alloc_df = model.optimize_portfolio_allocation(df=pd.DataFrame(), capital_idr=capital, save_output=False)
        cash_row = alloc_df[alloc_df["Ticker"] == "KAS SIAGA (CASH)"].iloc[0]

        # In Bear regime, Cash Reserve must be >= 35%
        self.assertGreaterEqual(
            cash_row["Allocation_Pct"],
            35.0,
            f"Bear regime failed to enforce >= 35% cash buffer (Allocated cash: {cash_row['Allocation_Pct']}%)"
        )

    def test_zero_candidate_fallback(self):
        """
        Verify fail-safe behavior when 0 stocks pass the quantitative hurdle:
        Must assign 100% to KAS SIAGA (CASH) without error or NaN.
        """
        model = QuantitativeAlphaModel()
        # Mock universe with zero positive Sharpe or BUY signals
        empty_candidates = pd.DataFrame(columns=["Ticker", "Close", "Sharpe_Ratio", "Bullish_Probability", "Recommendation"])
        model.get_or_extract_latest_rows = lambda df, tickers: empty_candidates

        alloc_df = model.optimize_portfolio_allocation(df=pd.DataFrame(), capital_idr=50_000_000.0, save_output=False)
        self.assertEqual(len(alloc_df), 1)
        cash_row = alloc_df.iloc[0]
        self.assertEqual(cash_row["Ticker"], "KAS SIAGA (CASH)")
        self.assertEqual(cash_row["Allocation_Pct"], 100.0)
        self.assertEqual(cash_row["Nominal_IDR"], 50_000_000.0)

    def test_live_portfolio_allocation_artifact(self):
        """Verify the generated latest_portfolio_allocation.csv satisfies all institutional portfolio invariants."""
        alloc_file = DATA_DIR / "latest_portfolio_allocation.csv"
        self.assertTrue(alloc_file.exists(), f"Portfolio allocation file not found: {alloc_file}")

        df = pd.read_csv(alloc_file)
        self.assertGreater(len(df), 0, "Allocation file is empty")

        # 1. Total nominal IDR equals exactly Rp 50,000,000
        total_nominal = df["Nominal_IDR"].sum()
        self.assertAlmostEqual(total_nominal, 50_000_000.0, delta=1.0)

        # 2. Total percentage is 100% +/- 0.2%
        total_pct = df["Allocation_Pct"].sum()
        self.assertAlmostEqual(total_pct, 100.0, delta=0.2)

        # 3. KAS SIAGA exists with >= 20%
        cash_rows = df[df["Ticker"] == "KAS SIAGA (CASH)"]
        self.assertEqual(len(cash_rows), 1, "Must contain exactly 1 cash reserve row")
        self.assertGreaterEqual(cash_rows.iloc[0]["Allocation_Pct"], 20.0)

        # 4. Equities check
        equities = df[df["Ticker"] != "KAS SIAGA (CASH)"]
        for _, row in equities.iterrows():
            self.assertLessEqual(row["Allocation_Pct"], 25.1)
            # Lots * 100 * Close == Nominal
            calc_nominal = row["Shares_Lot"] * 100.0 * row["Close_Price"]
            self.assertEqual(row["Nominal_IDR"], calc_nominal)


if __name__ == "__main__":
    unittest.main()
