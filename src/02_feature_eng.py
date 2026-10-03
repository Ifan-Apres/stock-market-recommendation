import json
import logging
from pathlib import Path
import sys
from typing import Optional, Dict, Any, List

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# pyrefly: ignore [missing-import]
import numpy as np  # type: ignore # pyrefly: ignore [missing-import]
import pandas as pd  # type: ignore # pyrefly: ignore [missing-import]
from arch import arch_model  # type: ignore # pyrefly: ignore [missing-import]
from scipy import stats  # type: ignore # pyrefly: ignore [missing-import]

# pyrefly: ignore [missing-import]
from src.config import (  # type: ignore # pyrefly: ignore [missing-import]
    ADVANCED_METRICS_FILE,
    BENCHMARK_DATA_FILE,
    FUNDAMENTAL_DATA_FILE,
    GLOBAL_MACRO_FILE,
    HISTORICAL_MACRO_FILE,
    LOG_FORMAT,
    PROCESSED_DATA_FILE,
    RAW_DATA_FILE,
    RISK_FREE_RATE,
    SECTOR_MAP,
)
from src.foreign_flow import FOREIGN_WEIGHTS, DEFAULT_FOREIGN_WEIGHT

# Configure logger
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("FeatureEngineering")

# BEI Seasonality Dividend Calendar (Historical distribution windows: Interim & Final)
BEI_DIVIDEND_MONTHS = {
    "BBCA": [3, 4, 12],
    "BBRI": [1, 3],
    "BMRI": [3, 4],
    "BBNI": [3, 4],
    "ASII": [5, 10],
    "HEXA": [9, 10],
    "PTBA": [6],
    "ADRO": [1, 5, 6, 12],
    "ITMG": [4, 9],
    "UNTR": [4, 5, 10],
    "TLKM": [6, 7],
    "ICBP": [7, 8],
    "INDF": [7, 8],
    "BRIS": [5],
    "ISAT": [6],
    "AADI": [5, 12],
    "KLBF": [5, 6],
    "CPIN": [6],
    "AMRT": [5],
    "PGAS": [6],
    "MEDC": [7, 11],
    "INKP": [7],
    "ANTM": [6],
}
DEFAULT_DIVIDEND_MONTHS = [4, 5, 6]


class QuantitativeFeatureEngineer:
    """
    Quantitative Feature & Alpha Signal Generation Engine.
    Engineers technical indicators, GARCH(1,1) conditional volatility,
    Value at Risk (VaR 95%), Pivot Support/Resistance levels, sector classification,
    and institutional portfolio metrics (Beta, Sharpe, Max Drawdown).
    """

    def __init__(
        self,
        raw_data_path: str = str(RAW_DATA_FILE),
        benchmark_path: str = str(BENCHMARK_DATA_FILE),
        macro_path: str = str(HISTORICAL_MACRO_FILE),
        risk_free_rate: float = RISK_FREE_RATE,
    ):
        self.raw_data_path = raw_data_path
        self.benchmark_path = benchmark_path
        self.macro_path = macro_path
        self.risk_free_rate = risk_free_rate
        # Invert sector map for fast ticker -> sector lookup
        self.ticker_to_sector = {}
        for sector, tickers in SECTOR_MAP.items():
            for t in tickers:
                self.ticker_to_sector[t] = sector

        # Load cached official IDX daily records for high-fidelity foreign flow features
        self.cached_idx_flows: Dict[str, Dict[str, Any]] = {}
        idx_daily_dir = ROOT_DIR / "data" / "raw" / "idx_daily"
        if idx_daily_dir.exists():
            for f in idx_daily_dir.glob("idx_summary_*.json"):
                d_str = f.stem.replace("idx_summary_", "")
                if len(d_str) == 8:
                    fmt_date = f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:]}"
                    try:
                        with open(f, "r", encoding="utf-8") as fp:
                            self.cached_idx_flows[fmt_date] = json.load(fp)
                    except Exception:
                        pass
        logger.info(f"Loaded {len(self.cached_idx_flows)} official IDX daily flow snapshots for feature engineering.")

    def load_raw_data(self) -> pd.DataFrame:
        """
        Loads ingested raw OHLCV market data from CSV.
        """
        logger.info(f"Loading raw market data from {self.raw_data_path}...")
        df = pd.read_csv(self.raw_data_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values(by=["Ticker", "Date"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def load_benchmark_data(self) -> pd.DataFrame:
        """
        Loads benchmark (^JKSE / IHSG) market data and computes 1D and 5D forward return.
        """
        if Path(self.benchmark_path).exists():
            bench_df = pd.read_csv(self.benchmark_path)
            bench_df["Date"] = pd.to_datetime(bench_df["Date"])
            bench_df.sort_values(by="Date", inplace=True)
            bench_df.reset_index(drop=True, inplace=True)
            bench_df["Benchmark_Return_1D"] = bench_df["Adj Close"].pct_change(1)
            bench_df["Benchmark_Return_5D"] = np.log(bench_df["Adj Close"].shift(-5) / bench_df["Adj Close"])
            return bench_df
        return pd.DataFrame()

    def load_macro_data(self) -> pd.DataFrame:
        """
        Loads historical multi-year cross-asset macro dataset.
        """
        if Path(self.macro_path).exists():
            macro_df = pd.read_csv(self.macro_path)
            macro_df["Date"] = pd.to_datetime(macro_df["Date"])
            macro_df.sort_values(by="Date", inplace=True)
            macro_df.reset_index(drop=True, inplace=True)
            return macro_df
        return pd.DataFrame()

    def load_fundamental_data(self) -> pd.DataFrame:
        """
        Loads fundamental financial summary dataset.
        """
        if FUNDAMENTAL_DATA_FILE.exists():
            return pd.read_csv(FUNDAMENTAL_DATA_FILE)
        return pd.DataFrame()

    @staticmethod
    def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_var_es_garch_t(return_series: pd.Series, alpha: float = 0.05, min_obs: int = 250, lam: float = 0.94) -> dict:
        """
        Estimates Student-t GARCH(1,1) conditional volatility, VaR (95% and 99%),
        and Expected Shortfall (ES) with RiskMetrics EWMA fallback (lambda=0.94).
        Avoids silent failures and parameter divergence.
        """
        clean_ret = pd.Series(return_series).dropna().to_numpy()
        x = clean_ret[-1000:] * 100.0  # Scale to percent for GARCH numerical stability

        if len(x) < 30:
            return {
                "garch_vol": 0.25,
                "var_95_1d": 0.025,
                "var_99_1d": 0.035,
                "es_95_1d": 0.031,
                "model": "prior_default",
                "nu": 8.0,
            }

        if len(x) < min_obs:
            # Fallback to EWMA RiskMetrics when sample is shorter than min_obs
            s2 = float(np.var(x[:30])) if len(x) >= 30 else float(np.var(x))
            for r in x:
                s2 = lam * s2 + (1.0 - lam) * (r ** 2)
            sig = np.sqrt(max(s2, 1e-6))
            z = stats.norm.ppf(alpha)
            var_pct = -z * sig
            es_pct = sig * stats.norm.pdf(z) / alpha
            ann_vol = (sig / 100.0) * np.sqrt(252)
            var_95_dec = float(np.clip(var_pct / 100.0, 0.005, 0.25))
            var_99_dec = float(np.clip((-stats.norm.ppf(0.01) * sig) / 100.0, 0.008, 0.35))
            es_95_dec = float(np.clip(es_pct / 100.0, 0.006, 0.35))
            return {
                "garch_vol": round(float(ann_vol), 4),
                "var_95_1d": round(var_95_dec, 4),
                "var_99_1d": round(var_99_dec, 4),
                "es_95_1d": round(es_95_dec, 4),
                "model": "ewma",
                "nu": 30.0,
            }

        try:
            am = arch_model(x, mean="Constant", vol="GARCH", p=1, q=1, dist="t", rescale=False)
            res = am.fit(disp="off", show_warning=False, options={"maxiter": 500})
            params = res.params
            a = float(params.get("alpha[1]", 0.0))
            b = float(params.get("beta[1]", 0.0))
            nu = float(params.get("nu", 8.0))
            mu = float(params.get("mu", 0.0))

            healthy = (
                res.convergence_flag == 0
                and a > 1e-3
                and (a + b) < 0.999
                and np.isfinite(getattr(res, "std_err", [0.0])).all()
                and 2.5 < nu < 60
            )

            if not healthy:
                # Fallback to EWMA RiskMetrics if GARCH parameters are degenerate or non-converged
                s2 = float(np.var(x[:30]))
                for r in x:
                    s2 = lam * s2 + (1.0 - lam) * (r ** 2)
                sig = np.sqrt(max(s2, 1e-6))
                z = stats.norm.ppf(alpha)
                var_pct = -z * sig
                es_pct = sig * stats.norm.pdf(z) / alpha
                ann_vol = (sig / 100.0) * np.sqrt(252)
                var_95_dec = float(np.clip(var_pct / 100.0, 0.005, 0.25))
                var_99_dec = float(np.clip((-stats.norm.ppf(0.01) * sig) / 100.0, 0.008, 0.35))
                es_95_dec = float(np.clip(es_pct / 100.0, 0.006, 0.35))
                return {
                    "garch_vol": round(float(ann_vol), 4),
                    "var_95_1d": round(var_95_dec, 4),
                    "var_99_1d": round(var_99_dec, 4),
                    "es_95_1d": round(es_95_dec, 4),
                    "model": "ewma_unhealthy_guard",
                    "nu": round(float(nu), 2),
                }

            forecast = res.forecast(horizon=1)
            sig = np.sqrt(forecast.variance.iloc[-1, 0])
            k = np.sqrt((nu - 2.0) / nu)
            tq = stats.t.ppf(alpha, nu)
            var_pct = -(mu + sig * k * tq)
            es_pct = -mu + sig * k * stats.t.pdf(tq, nu) / alpha * (nu + (tq ** 2)) / (nu - 1.0)
            tq99 = stats.t.ppf(0.01, nu)
            var99_pct = -(mu + sig * k * tq99)
            ann_vol = (sig / 100.0) * np.sqrt(252)

            var_95_dec = float(np.clip(var_pct / 100.0, 0.005, 0.25))
            var_99_dec = float(np.clip(var99_pct / 100.0, 0.008, 0.35))
            es_95_dec = float(np.clip(es_pct / 100.0, 0.006, 0.35))

            return {
                "garch_vol": round(float(ann_vol), 4),
                "var_95_1d": round(var_95_dec, 4),
                "var_99_1d": round(var_99_dec, 4),
                "es_95_1d": round(es_95_dec, 4),
                "model": "garch-t",
                "nu": round(float(nu), 2),
                "persist": round(float(a + b), 4),
            }
        except Exception:
            # Fallback to EWMA RiskMetrics upon numerical error
            s2 = float(np.var(x[:30])) if len(x) >= 30 else float(np.var(x))
            for r in x:
                s2 = lam * s2 + (1.0 - lam) * (r ** 2)
            sig = np.sqrt(max(s2, 1e-6))
            z = stats.norm.ppf(alpha)
            var_pct = -z * sig
            es_pct = sig * stats.norm.pdf(z) / alpha
            var_95_dec = float(np.clip(var_pct / 100.0, 0.005, 0.25))
            var_99_dec = float(np.clip((-stats.norm.ppf(0.01) * sig) / 100.0, 0.008, 0.35))
            es_95_dec = float(np.clip(es_pct / 100.0, 0.006, 0.35))
            return {
                "garch_vol": round(float((sig / 100.0) * np.sqrt(252)), 4),
                "var_95_1d": round(var_95_dec, 4),
                "var_99_1d": round(var_99_dec, 4),
                "es_95_1d": round(es_95_dec, 4),
                "model": "ewma_exception_fallback",
                "nu": 8.0,
            }

    def generate_ticker_features(
        self,
        group: pd.DataFrame,
        bench_df: pd.DataFrame,
        macro_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Constructs technical, institutional money flow, support/resistance, and GARCH volatility metrics.
        """
        df = group.copy()
        df.sort_values(by="Date", inplace=True)
        ticker = df["Ticker"].iloc[0]
        df["Sector"] = self.ticker_to_sector.get(ticker, "General")

        # 1. Price Returns & Momentum
        df["Log_Return"] = np.log(df["Adj Close"] / df["Adj Close"].shift(1))
        df["Return_1D"] = df["Adj Close"].pct_change(1)
        df["Return_5D"] = df["Adj Close"].pct_change(5)
        df["Return_20D"] = df["Adj Close"].pct_change(20)

        # 2. Moving Averages
        df["SMA_10"] = df["Adj Close"].rolling(window=10).mean()
        df["SMA_20"] = df["Adj Close"].rolling(window=20).mean()
        df["SMA_50"] = df["Adj Close"].rolling(window=50).mean()
        df["SMA_200"] = df["Adj Close"].rolling(window=200).mean()
        df["Dist_SMA_20"] = (df["Adj Close"] - df["SMA_20"]) / (df["SMA_20"] + 1e-9)
        df["Dist_SMA_50"] = (df["Adj Close"] - df["SMA_50"]) / (df["SMA_50"] + 1e-9)
        df["Dist_SMA_200"] = (df["Adj Close"] - df["SMA_200"]) / (df["SMA_200"] + 1e-9)

        df["EMA_12"] = df["Adj Close"].ewm(span=12, adjust=False).mean()
        df["EMA_26"] = df["Adj Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = df["EMA_12"] - df["EMA_26"]
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

        # 3. Oscillators & Volatility
        df["RSI_14"] = self.calculate_rsi(df["Adj Close"], window=14)
        df["Volatility_20D"] = df["Log_Return"].rolling(window=20).std() * np.sqrt(252)

        bb_std = df["Adj Close"].rolling(window=20).std()
        df["Bollinger_Upper"] = df["SMA_20"] + (bb_std * 2)
        df["Bollinger_Lower"] = df["SMA_20"] - (bb_std * 2)

        # 4. Volume Dynamics & Institutional Money Flow (Bandarmologi)
        df["Volume_SMA_20"] = df["Volume"].rolling(window=20).mean()
        df["Volume_Ratio"] = df["Volume"] / (df["Volume_SMA_20"] + 1e-9)

        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
        raw_money_flow = typical_price * df["Volume"]
        tp_diff = typical_price.diff()
        pos_flow = raw_money_flow.where(tp_diff > 0, 0.0).rolling(window=14).sum()
        neg_flow = raw_money_flow.where(tp_diff < 0, 0.0).rolling(window=14).sum()
        mfi_ratio = pos_flow / (neg_flow + 1e-9)
        df["MFI_14"] = 100.0 - (100.0 / (1.0 + mfi_ratio))

        mf_multiplier = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / ((df["High"] - df["Low"]) + 1e-9)
        mf_volume = mf_multiplier * df["Volume"]
        df["CMF_20"] = mf_volume.rolling(window=20).sum() / (df["Volume"].rolling(window=20).sum() + 1e-9)

        # 4.1 Official & Hybrid Foreign Money Flow (Arus Modal Asing BEI)
        ticker_clean = str(ticker).replace(".JK", "").strip().upper()
        fw = FOREIGN_WEIGHTS.get(ticker_clean, DEFAULT_FOREIGN_WEIGHT)

        df["Value"] = df["Close"] * df["Volume"]
        df["Value_SMA_20"] = df["Value"].rolling(window=20).mean()

        net_foreign_list = []
        part_list = []

        for _, row in df.iterrows():
            d_str = row["Date"].strftime("%Y-%m-%d")
            if d_str in self.cached_idx_flows and ticker_clean in self.cached_idx_flows[d_str]:
                item = self.cached_idx_flows[d_str][ticker_clean]
                net_foreign_list.append(float(item.get("net_foreign_idr", 0.0)))
                part_list.append(float(item.get("foreign_participation", fw)))
            else:
                ret = row.get("Return_1D", 0.0)
                val = row.get("Value", 0.0)
                scaled_pressure = np.clip(ret * 7.5, -0.65, 0.65)
                fallback_flow = val * fw * scaled_pressure
                net_foreign_list.append(fallback_flow)
                part_list.append(fw)

        df["Net_Foreign_IDR"] = net_foreign_list
        df["Foreign_Participation"] = np.clip(part_list, 0.0, 1.0)
        df["Foreign_Flow_Norm_1D"] = np.clip(df["Net_Foreign_IDR"] / (df["Value_SMA_20"] + 1e-9), -3.0, 3.0).fillna(0.0)
        df["Foreign_Flow_5D_Accum"] = np.clip(df["Net_Foreign_IDR"].rolling(5).sum() / ((df["Value_SMA_20"] * 5.0) + 1e-9), -3.0, 3.0).fillna(0.0)
        df["Foreign_Flow_Momentum"] = (df["Foreign_Flow_Norm_1D"] - df["Foreign_Flow_Norm_1D"].shift(3)).fillna(0.0)

        # 4.2 Enhanced Smart Money Flow & Divergence (RND Features)
        df["Foreign_Accum_Divergence"] = np.clip(df["Foreign_Flow_5D_Accum"] - df["Return_5D"], -3.0, 3.0).fillna(0.0)
        df["Foreign_Flow_Intensity"] = np.clip(df["Foreign_Flow_Norm_1D"] * df["Foreign_Participation"], -3.0, 3.0).fillna(0.0)
        df["Foreign_Consistent_Buy_5D"] = (df["Net_Foreign_IDR"] > 0).astype(float).rolling(5).mean().fillna(0.5)

        # 4.3 Dividend Seasonality & Cyclical Dynamics (RND Features)
        season_months = BEI_DIVIDEND_MONTHS.get(ticker_clean, DEFAULT_DIVIDEND_MONTHS)
        row_months = df["Date"].dt.month
        df["Is_Dividend_Season"] = row_months.isin(season_months).astype(float)
        df["Dividend_Season_Momentum"] = df["Is_Dividend_Season"] * df["Return_20D"].fillna(0.0)

        # 5. Technical Floor Pivot Levels (Support & Resistance for BoW & BoB)
        # Next-day pivot levels (projected from completed session t for next session t+1)
        curr_h = df["High"]
        curr_l = df["Low"]
        curr_c = df["Close"]
        next_pivot = (curr_h + curr_l + curr_c) / 3.0
        df["Next_Pivot_Point"] = next_pivot
        df["Next_Support_1"] = (2.0 * next_pivot) - curr_h
        df["Next_Support_2"] = next_pivot - (curr_h - curr_l)
        df["Next_Resistance_1"] = (2.0 * next_pivot) - curr_l
        df["Next_Resistance_2"] = next_pivot + (curr_h - curr_l)

        # Shifted pivot levels (for historical backtest rows without lookahead)
        df["Pivot_Point"] = next_pivot.shift(1)
        df["Support_1"] = df["Next_Support_1"].shift(1)
        df["Support_2"] = df["Next_Support_2"].shift(1)
        df["Resistance_1"] = df["Next_Resistance_1"].shift(1)
        df["Resistance_2"] = df["Next_Resistance_2"].shift(1)

        # 6. Advanced Institutional Portfolio Risk Metrics
        df["Annualized_Return_1Y"] = df["Adj Close"].pct_change(252)
        df["Annualized_Vol_252D"] = df["Log_Return"].rolling(window=252).std() * np.sqrt(252)

        rolling_peak = df["Adj Close"].rolling(window=252, min_periods=20).max()
        daily_drawdown = (df["Adj Close"] - rolling_peak) / (rolling_peak + 1e-9)
        df["Max_Drawdown_1Y"] = daily_drawdown.rolling(window=252, min_periods=20).min()

        if not bench_df.empty:
            merged_bench = df[["Date", "Return_1D"]].merge(
                bench_df[["Date", "Benchmark_Return_1D"]], on="Date", how="left"
            )
            merged_bench["Benchmark_Return_1D"] = merged_bench["Benchmark_Return_1D"].fillna(0.0)
            cov = merged_bench["Return_1D"].rolling(window=252, min_periods=60).cov(merged_bench["Benchmark_Return_1D"])
            var = merged_bench["Benchmark_Return_1D"].rolling(window=252, min_periods=60).var()
            beta_vals = np.clip((cov / (var + 1e-9)).fillna(1.0), -1.0, 4.0).to_numpy()
            df["Beta_IHSG"] = beta_vals
        else:
            df["Beta_IHSG"] = 1.0


        df["Sharpe_Ratio"] = np.clip(
            (df["Annualized_Return_1Y"] - self.risk_free_rate) / (df["Annualized_Vol_252D"] + 1e-6), -5.0, 10.0
        ).fillna(0.0)

        # GARCH-t Volatility & Value at Risk (VaR 95% and 99%, ES)
        garch_res = self.calculate_var_es_garch_t(df["Return_1D"])
        df["GARCH_Vol"] = garch_res["garch_vol"]
        df["VaR_95_1D"] = garch_res["var_95_1d"]
        df["VaR_99_1D"] = garch_res["var_99_1d"]
        df["ES_95_1D"] = garch_res["es_95_1d"]
        df["GARCH_Model"] = garch_res["model"]

        # 7. Merge Benchmark & Global Macro Features
        if not bench_df.empty:
            bench_sub = bench_df[["Date", "Benchmark_Return_1D", "Benchmark_Return_5D"]].copy()
            df = df.merge(bench_sub, on="Date", how="left")
            df["IHSG_Return_1D"] = df["Benchmark_Return_1D"].fillna(0.0)
        else:
            df["Benchmark_Return_1D"] = 0.0
            df["Benchmark_Return_5D"] = 0.0
            df["IHSG_Return_1D"] = 0.0

        if macro_df is not None and not macro_df.empty:
            macro_cols = ["Date", "US_10Y_Yield_Delta", "USD_IDR_Return_1D", "Brent_Oil_Return_1D", "SP500_Return_1D"]
            avail_cols = [c for c in macro_cols if c in macro_df.columns]
            df = df.merge(macro_df[avail_cols], on="Date", how="left")
            for c in ["US_10Y_Yield_Delta", "USD_IDR_Return_1D", "Brent_Oil_Return_1D", "SP500_Return_1D"]:
                if c in df.columns:
                    df[c] = df[c].ffill().bfill().fillna(0.0)
                else:
                    df[c] = 0.0
        else:
            df["US_10Y_Yield_Delta"] = 0.0
            df["USD_IDR_Return_1D"] = 0.0
            df["Brent_Oil_Return_1D"] = 0.0
            df["SP500_Return_1D"] = 0.0

        # 8. Sector-Macro Interaction Features (Sector Sensitivity)
        sector_name = str(df["Sector"].iloc[0]) if "Sector" in df.columns else "General"
        # Energy Tailwind: Oil return active specifically for Energy sector
        df["Oil_Energy_Tailwind"] = np.where(sector_name == "Energy", df["Brent_Oil_Return_1D"], 0.0)
        # Bank Yield Sensitivity: 10Y Yield delta active specifically for Financials
        df["Rate_Bank_Sensitivity"] = np.where(sector_name == "Financials", df["US_10Y_Yield_Delta"], 0.0)
        # Consumer FX Headwind: USD/IDR depreciation active for Consumer and Healthcare
        consumer_sectors = ["Consumer Non-Cyclicals", "Consumer Cyclicals", "Healthcare"]
        df["FX_Consumer_Headwind"] = np.where(sector_name in consumer_sectors, df["USD_IDR_Return_1D"], 0.0)

        # 9. Forward Target: Excess Return (Alpha Relatif vs IHSG)
        df["Target_Return_5D"] = np.log(df["Adj Close"].shift(-5) / df["Adj Close"])
        df["Target_Excess_Return_5D"] = df["Target_Return_5D"] - df["Benchmark_Return_5D"]
        excess_val = df["Target_Excess_Return_5D"].fillna(df["Target_Return_5D"])
        df["Target_Class_5D"] = np.where(
            df["Target_Return_5D"].isna(),
            np.nan,
            (excess_val > 0.0).astype(float),
        )

        return df

    def engineer_features(self) -> pd.DataFrame:
        """
        Applies feature transformations across all tickers.
        """
        raw_df = self.load_raw_data()
        bench_df = self.load_benchmark_data()
        macro_df = self.load_macro_data()
        fund_df = self.load_fundamental_data()

        logger.info(f"Generating quant features across {raw_df['Ticker'].nunique()} emiten with GARCH, Excess Alpha & Macro Context...")
        processed_groups = []
        for _, group in raw_df.groupby("Ticker"):
            ticker_features = self.generate_ticker_features(group, bench_df, macro_df)
            processed_groups.append(ticker_features)

        feature_df = pd.concat(processed_groups, ignore_index=True)

        if not fund_df.empty:
            feature_df = feature_df.merge(fund_df, on="Ticker", how="left")
            feature_df["Market_Cap"] = feature_df["Market_Cap"].astype(float).fillna(1e9)
            feature_df["Log_Market_Cap"] = np.log(feature_df["Market_Cap"] + 1e-9)
            raw_dy = feature_df.get("Dividend_Yield", pd.Series(0.0, index=feature_df.index)).fillna(0.0)
            feature_df["Dividend_Yield_Norm"] = np.clip(raw_dy / (15.0 if raw_dy.max() > 1.0 else 0.15), 0.0, 2.0).fillna(0.0)
        else:
            feature_df["Dividend_Yield_Norm"] = 0.0

        clean_df = feature_df.dropna(subset=["SMA_200", "RSI_14"]).copy()
        clean_df.reset_index(drop=True, inplace=True)

        self.generate_advanced_metrics_snapshot(clean_df)
        return clean_df

    def generate_advanced_metrics_snapshot(self, df: pd.DataFrame) -> None:
        latest_records = []
        for ticker, group in df.groupby("Ticker"):
            latest_row = group.sort_values(by="Date").iloc[-1].to_dict()
            if "Next_Pivot_Point" in latest_row and pd.notna(latest_row["Next_Pivot_Point"]):
                latest_row["Pivot_Point"] = latest_row["Next_Pivot_Point"]
                latest_row["Support_1"] = latest_row["Next_Support_1"]
                latest_row["Support_2"] = latest_row["Next_Support_2"]
                latest_row["Resistance_1"] = latest_row["Next_Resistance_1"]
                latest_row["Resistance_2"] = latest_row["Next_Resistance_2"]
            latest_records.append(latest_row)

        if not latest_records:
            return


        latest_df = pd.DataFrame(latest_records)
        cols = [
            "Ticker", "Sector", "Date", "Close", "Annualized_Return_1Y", "Annualized_Vol_252D",
            "Beta_IHSG", "Sharpe_Ratio", "Max_Drawdown_1Y", "GARCH_Vol", "VaR_95_1D",
            "VaR_99_1D", "ES_95_1D", "GARCH_Model",
            "Pivot_Point", "Support_1", "Resistance_1", "PE_Ratio", "PB_Ratio", "ROE",
            "Dividend_Yield", "Dividend_Yield_Norm", "Is_Dividend_Season", "Dividend_Season_Momentum",
            "Debt_to_Equity", "Current_Ratio",
            "Foreign_Flow_Norm_1D", "Foreign_Participation", "Foreign_Flow_5D_Accum", "Foreign_Flow_Momentum",
            "Foreign_Accum_Divergence", "Foreign_Flow_Intensity", "Foreign_Consistent_Buy_5D"
        ]
        available_cols = [c for c in cols if c in latest_df.columns]
        snapshot_df = latest_df[available_cols].copy()
        snapshot_df["Date"] = pd.to_datetime(snapshot_df["Date"]).dt.strftime("%Y-%m-%d")

        ADVANCED_METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
        snapshot_df.to_csv(ADVANCED_METRICS_FILE, index=False)
        logger.info(f"Advanced quant & GARCH snapshot saved to {ADVANCED_METRICS_FILE}")

    def save_processed_data(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        PROCESSED_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_DATA_FILE, index=False)
        logger.info(f"Processed feature matrix saved to {PROCESSED_DATA_FILE}")


def run_feature_engineering_pipeline() -> pd.DataFrame:
    engineer = QuantitativeFeatureEngineer()
    feature_df = engineer.engineer_features()
    engineer.save_processed_data(feature_df)
    return feature_df


if __name__ == "__main__":
    run_feature_engineering_pipeline()
