"""
Unit & Integration Tests: Data Contracts & Schema Integrity.
Verifies the 37 quantitative feature matrix, zero-NaN inference invariants,
boundary constraints, and snapshot/recommendations schema fidelity.
"""

import importlib
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
inference_mod = importlib.import_module("src.03_model_inference")
FEATURE_COLUMNS = inference_mod.FEATURE_COLUMNS
DATA_DIR = ROOT_DIR / "data" / "processed"


class TestDataContracts(unittest.TestCase):
    """Institutional Data Contract & Schema Integrity Test Suite."""

    def test_feature_columns_dimension_and_schema(self):
        """Verify feature contract specifies exactly 37 distinct features with all layers."""
        self.assertEqual(len(FEATURE_COLUMNS), 37, f"Expected 37 features, got {len(FEATURE_COLUMNS)}")
        self.assertEqual(len(set(FEATURE_COLUMNS)), 37, "Duplicate feature names detected in FEATURE_COLUMNS")

        # Verify Layer 2: FinBERT Macro Sentiment features are present
        sentiment_features = [
            "daily_finbert_score",
            "sentiment_ema_3",
            "sentiment_ema_7",
            "macro_shock_flag",
            "vol_sentiment_interaction",
        ]
        for feat in sentiment_features:
            self.assertIn(feat, FEATURE_COLUMNS, f"Missing critical sentiment feature: {feat}")

        # Verify Foreign Flow features are present
        foreign_features = [
            "Foreign_Flow_Norm_1D",
            "Foreign_Flow_5D_Accum",
            "Foreign_Participation",
            "Foreign_Flow_Momentum",
            "Foreign_Accum_Divergence",
            "Foreign_Flow_Intensity",
            "Foreign_Consistent_Buy_5D",
        ]
        for feat in foreign_features:
            self.assertIn(feat, FEATURE_COLUMNS, f"Missing foreign flow feature: {feat}")

        # Verify Macro sensitivity features are present
        macro_features = [
            "IHSG_Return_1D",
            "US_10Y_Yield_Delta",
            "USD_IDR_Return_1D",
            "Brent_Oil_Return_1D",
            "Oil_Energy_Tailwind",
            "Rate_Bank_Sensitivity",
            "FX_Consumer_Headwind",
        ]
        for feat in macro_features:
            self.assertIn(feat, FEATURE_COLUMNS, f"Missing macro feature: {feat}")

    def test_processed_market_features_integrity(self):
        """Verify processed feature matrix exists, contains all 37 features, and maintains zero-NaN in valid training/inference rows."""
        processed_file = DATA_DIR / "processed_market_features.csv"
        self.assertTrue(processed_file.exists(), f"Processed data file not found: {processed_file}")

        # Read only required columns and valid subset to optimize memory & speed
        sample_df = pd.read_csv(processed_file, nrows=5000)
        for col in FEATURE_COLUMNS:
            self.assertIn(col, sample_df.columns, f"Feature column '{col}' missing from processed_market_features.csv")

        # Check latest observation slice per ticker has 0% NaN
        full_cols = ["Date", "Ticker"] + FEATURE_COLUMNS
        df_latest = pd.read_csv(processed_file, usecols=full_cols)
        latest_rows = df_latest.sort_values(by="Date").groupby("Ticker").last()

        nan_counts = latest_rows[FEATURE_COLUMNS].isna().sum()
        failing_cols = nan_counts[nan_counts > 0]
        self.assertTrue(
            failing_cols.empty,
            f"Detected NaN values in latest inference slice for features: {failing_cols.to_dict()}"
        )

    def test_sentiment_feature_boundary_constraints(self):
        """Verify mathematical boundary invariants for sentiment and shock signals."""
        processed_file = DATA_DIR / "processed_market_features.csv"
        cols_to_check = [
            "daily_finbert_score",
            "sentiment_ema_3",
            "sentiment_ema_7",
            "macro_shock_flag",
        ]
        df = pd.read_csv(processed_file, usecols=cols_to_check, nrows=10000)

        # FinBERT scores must be in [-1.0, 1.0]
        self.assertTrue(
            (df["daily_finbert_score"] >= -1.0 - 1e-6).all() and (df["daily_finbert_score"] <= 1.0 + 1e-6).all(),
            "daily_finbert_score violated [-1.0, 1.0] boundary constraint"
        )
        self.assertTrue(
            (df["sentiment_ema_3"] >= -1.0 - 1e-6).all() and (df["sentiment_ema_3"] <= 1.0 + 1e-6).all(),
            "sentiment_ema_3 violated [-1.0, 1.0] boundary constraint"
        )
        self.assertTrue(
            (df["sentiment_ema_7"] >= -1.0 - 1e-6).all() and (df["sentiment_ema_7"] <= 1.0 + 1e-6).all(),
            "sentiment_ema_7 violated [-1.0, 1.0] boundary constraint"
        )

        # Macro shock flag must be binary {0.0, 1.0}
        unique_flags = set(df["macro_shock_flag"].unique())
        self.assertTrue(
            unique_flags.issubset({0.0, 1.0}),
            f"macro_shock_flag contains non-binary values: {unique_flags}"
        )

    def test_latest_recommendations_contract(self):
        """Verify latest_recommendations.json schema, ticker completeness, rank uniqueness, and valid price levels."""
        recs_file = DATA_DIR / "latest_recommendations.json"
        self.assertTrue(recs_file.exists(), f"Recommendations file not found: {recs_file}")

        with open(recs_file, "r", encoding="utf-8") as f:
            recs = json.load(f)

        self.assertEqual(len(recs), 66, f"Expected 66 emiten in active universe, got {len(recs)}")

        tickers = [r["Ticker"] for r in recs]
        self.assertEqual(len(set(tickers)), 66, "Duplicate tickers found in recommendations")

        ranks = [r["Cross_Sectional_Rank"] for r in recs]
        self.assertEqual(sorted(ranks), list(range(1, 67)), "Cross_Sectional_Rank must be sequential 1 to 66")

        allowed_actions = {"BUY ON WEAKNESS", "BUY ON BREAKOUT", "TRADING BUY", "SELL ON STRENGTH", "HOLD"}
        allowed_tiers = {
            "Tier 1: Top Decile (Alpha Outperformer)",
            "Tier 2: Upper Mid (Strong Beta / Accumulation)",
            "Tier 3: Neutral (Market Performer)",
            "Tier 4: Bottom Decile (Underperformer / Avoid)",
        }

        for r in recs:
            self.assertIn(r["Recommendation"], allowed_actions, f"Invalid recommendation: {r['Recommendation']}")
            self.assertIn(r["Alpha_Tier"], allowed_tiers, f"Invalid alpha tier: {r['Alpha_Tier']}")
            self.assertGreaterEqual(r["Bullish_Probability"], 0.0)
            self.assertLessEqual(r["Bullish_Probability"], 1.0)
            self.assertGreater(r["Entry_Price"], 0)
            self.assertGreater(r["Target_Price"], 0)
            self.assertGreater(r["Stop_Loss"], 0)
            self.assertGreaterEqual(r["Risk_Reward_Ratio"], 0.5)

    def test_snapshot_json_contract(self):
        """Verify snapshot.json root schema, market regime state, and component payloads."""
        snap_file = DATA_DIR / "snapshot.json"
        self.assertTrue(snap_file.exists(), f"Snapshot file not found: {snap_file}")

        with open(snap_file, "r", encoding="utf-8") as f:
            snap = json.load(f)

        self.assertEqual(snap.get("status"), "online")
        self.assertIn("market_regime", snap)
        self.assertIn(snap["market_regime"], ["BULLISH_TREND", "BEARISH_MARKDOWN", "NEUTRAL_SIDEWAYS"])

        expected_keys = [
            "data_asof",
            "generated_at",
            "metrics",
            "morning_brief",
            "portfolio_allocation",
            "recommendations_swing",
            "recommendations_dividend",
            "recommendations_favorites",
            "all_recommendations",
        ]
        for key in expected_keys:
            self.assertIn(key, snap, f"Missing key '{key}' in snapshot.json")

        self.assertEqual(len(snap["all_recommendations"]), 66)
        self.assertTrue(len(snap["portfolio_allocation"]) > 0)


if __name__ == "__main__":
    unittest.main()
