"""
Integration Tests: FastAPI Endpoints, Authentication, & Schema Contracts (Langkah 2).
Validates HTTP responses, Bearer token security, anti-bot WAF filters,
snapshot payload schemas, and cross-sectional recommendation serving.
"""

import importlib
import json
from pathlib import Path
import unittest
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent
api_mod = importlib.import_module("api.main")
app = api_mod.app

auth_mod = importlib.import_module("src.auth")
create_token = auth_mod.create_token


class TestAPIIntegration(unittest.TestCase):
    """Institutional API Endpoint & Schema Integration Test Suite."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.browser_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "application/json",
        }
        # Generate valid authenticated session token
        cls.test_user = {
            "sub": "quant_researcher_01",
            "name": "Alexander Vance",
            "email": "vance@alphatech-quant.com",
            "role": "Lead Quantitative Researcher",
        }
        cls.auth_token = create_token(cls.test_user)
        cls.auth_headers = {
            **cls.browser_headers,
            "Authorization": f"Bearer {cls.auth_token}",
        }

    def test_status_endpoint(self):
        """Verify /status and /api/status return online system state with 11 IDX sectors."""
        res = self.client.get("/api/status", headers=self.browser_headers)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data.get("status"), "online")
        self.assertIn("engine", data)
        self.assertIn("sectors_covered", data)
        self.assertEqual(len(data["sectors_covered"]), 11)

    def test_snapshot_endpoint_contract(self):
        """Verify /api/snapshot returns unified institutional snapshot with market regime and alpha tiers."""
        res = self.client.get("/api/snapshot", headers=self.browser_headers)
        self.assertEqual(res.status_code, 200)
        snap = res.json()

        self.assertEqual(snap.get("status"), "online")
        self.assertIn("market_regime", snap)
        self.assertIn(snap["market_regime"], ["BULLISH_TREND", "BEARISH_MARKDOWN", "NEUTRAL_SIDEWAYS"])

        # Check all_recommendations
        all_recs = snap.get("all_recommendations", [])
        self.assertEqual(len(all_recs), 66, f"Expected 66 emiten in snapshot, got {len(all_recs)}")

        # Check first stock has cross-sectional rank 1 and Tier 1 badge
        first_stock = all_recs[0]
        self.assertEqual(first_stock.get("Cross_Sectional_Rank"), 1)
        self.assertIn("Tier 1", first_stock.get("Alpha_Tier", ""))
        self.assertIn("Recommendation", first_stock)
        self.assertGreater(first_stock.get("Bullish_Probability", 0), 0)

        # Check portfolio allocation in snapshot
        alloc = snap.get("portfolio_allocation", [])
        self.assertTrue(len(alloc) > 0)
        cash_item = [a for a in alloc if a.get("Ticker") == "KAS SIAGA (CASH)"]
        self.assertEqual(len(cash_item), 1)

    def test_recommendations_endpoint_auth_protection(self):
        """Verify unauthenticated requests are rejected (401), while authenticated requests succeed."""
        # 1. Unauthenticated request must return 401
        unauth_res = self.client.get("/api/recommendations", headers=self.browser_headers)
        self.assertEqual(unauth_res.status_code, 401)

        # 2. Authenticated request with Bearer token succeeds with 200
        auth_res = self.client.get("/api/recommendations?mode=all", headers=self.auth_headers)
        self.assertEqual(auth_res.status_code, 200)
        recs = auth_res.json()
        self.assertEqual(len(recs), 66)

        # Verify enriched schema fields are served cleanly
        sample = recs[0]
        self.assertIn("Ticker", sample)
        self.assertIn("Cross_Sectional_Rank", sample)
        self.assertIn("Alpha_Tier", sample)
        self.assertIn("Market_Regime", sample)
        self.assertIn("Bullish_Probability", sample)
        self.assertIn("Entry_Price", sample)
        self.assertIn("Target_Price", sample)
        self.assertIn("Stop_Loss", sample)

    def test_portfolio_allocate_endpoint(self):
        """Verify /api/portfolio/allocate scales capital and returns valid lot sizes."""
        test_capital = 100_000_000.0
        res = self.client.get(f"/api/portfolio/allocate?capital={test_capital}", headers=self.auth_headers)
        self.assertEqual(res.status_code, 200)
        items = res.json()
        self.assertTrue(len(items) > 0)

        total_alloc_nominal = sum(item["Nominal_IDR"] for item in items)
        self.assertAlmostEqual(total_alloc_nominal, test_capital, delta=500.0)

        # Ensure Cash Reserve item is present
        cash_items = [i for i in items if i["Ticker"] == "KAS SIAGA (CASH)"]
        self.assertEqual(len(cash_items), 1)

    def test_morning_brief_endpoint(self):
        """Verify /api/morning-brief returns institutional IHSG market brief."""
        res = self.client.get("/api/morning-brief", headers=self.browser_headers)
        self.assertEqual(res.status_code, 200)
        brief = res.json()
        self.assertIn("headline", brief)
        self.assertIn("brief_content", brief)
        self.assertIn("cards", brief)

    def test_emiten_financials_and_history(self):
        """Verify /api/financials/{ticker} and /api/history/{ticker} return structured fundamental & price data."""
        # 1. Financials
        fin_res = self.client.get("/api/financials/BBCA", headers=self.auth_headers)
        self.assertEqual(fin_res.status_code, 200)
        fin = fin_res.json()
        self.assertEqual(fin.get("ticker"), "BBCA")
        self.assertIn("metrics", fin)
        self.assertIn("health", fin)

        # 2. 30-Day Price History
        hist_res = self.client.get("/api/history/BBCA", headers=self.auth_headers)
        self.assertEqual(hist_res.status_code, 200)
        hist = hist_res.json()
        self.assertIn("prices", hist)
        self.assertIn("volumes", hist)

    def test_security_waf_bot_blocking(self):
        """Verify that automated scraper signatures (python-requests, curl, httpx) are blocked with 403 Forbidden on protected data routes."""
        bot_headers = {
            "User-Agent": "python-requests/2.31.0",
            "Authorization": f"Bearer {self.auth_token}",
        }
        res = self.client.get("/api/recommendations", headers=bot_headers)
        self.assertEqual(res.status_code, 403)
        self.assertEqual(res.json().get("error_code"), "BOT_SCRAPING_FORBIDDEN")


if __name__ == "__main__":
    unittest.main()
