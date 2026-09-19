import logging
from pathlib import Path
import sys

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# pyrefly: ignore [missing-import]
import numpy as np  # type: ignore # pyrefly: ignore [missing-import]
import pandas as pd  # type: ignore # pyrefly: ignore [missing-import]
from arch import arch_model  # type: ignore # pyrefly: ignore [missing-import]

# pyrefly: ignore [missing-import]
from src.config import (  # type: ignore # pyrefly: ignore [missing-import]
    ADVANCED_METRICS_FILE,
    BENCHMARK_DATA_FILE,
    FUNDAMENTAL_DATA_FILE,
    GLOBAL_MACRO_FILE,
    LOG_FORMAT,
    PROCESSED_DATA_FILE,
    RAW_DATA_FILE,
    RISK_FREE_RATE,
    SECTOR_MAP,
)

# Configure logger
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("FeatureEngineering")


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
        risk_free_rate: float = RISK_FREE_RATE,
    ):
        self.raw_data_path = raw_data_path
        self.benchmark_path = benchmark_path
        self.risk_free_rate = risk_free_rate
        # Invert sector map for fast ticker -> sector lookup
        self.ticker_to_sector = {}
        for sector, tickers in SECTOR_MAP.items():
            for t in tickers:
                self.ticker_to_sector[t] = sector

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
        Loads benchmark (^JKSE / IHSG) market data.
        """
        if BENCHMARK_DATA_FILE.exists():
            bench_df = pd.read_csv(self.benchmark_path)
            bench_df["Date"] = pd.to_datetime(bench_df["Date"])
            bench_df.sort_values(by="Date", inplace=True)
            bench_df.reset_index(drop=True, inplace=True)
            return bench_df
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
    def estimate_garch_volatility(return_series: pd.Series) -> float:
        """
        Estimates GARCH(1,1) conditional volatility on 120-day return window.
        """
        clean_ret = return_series.dropna().values
        if len(clean_ret) < 60:
            return float(np.std(clean_ret) * np.sqrt(252)) if len(clean_ret) > 0 else 0.25

        recent_ret = clean_ret[-120:] * 100.0  # Scale to percent for GARCH numerical stability
        try:
            am = arch_model(recent_ret, vol="Garch", p=1, q=1, rescale=False)
            res = am.fit(disp="off", show_warning=False)
            forecast = res.forecast(horizon=1)
            var_1d = forecast.variance.iloc[-1, 0]
            vol_daily = np.sqrt(var_1d) / 100.0
            return float(round(vol_daily * np.sqrt(252), 4))
        except Exception:
            return float(round(np.std(recent_ret / 100.0) * np.sqrt(252), 4))

    def generate_ticker_features(self, group: pd.DataFrame, bench_df: pd.DataFrame) -> pd.DataFrame:
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

        # 5. Technical Floor Pivot Levels (Support & Resistance for BoW & BoB)
        prev_h = df["High"].shift(1)
        prev_l = df["Low"].shift(1)
        prev_c = df["Close"].shift(1)
        pivot = (prev_h + prev_l + prev_c) / 3.0
        df["Pivot_Point"] = pivot
        df["Support_1"] = (2.0 * pivot) - prev_h
        df["Support_2"] = pivot - (prev_h - prev_l)
        df["Resistance_1"] = (2.0 * pivot) - prev_l
        df["Resistance_2"] = pivot + (prev_h - prev_l)

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
            cov = merged_bench["Return_1D"].rolling(window=252).cov(merged_bench["Benchmark_Return_1D"])
            var = merged_bench["Benchmark_Return_1D"].rolling(window=252).var()
            df["Beta_IHSG"] = np.clip((cov / (var + 1e-9)).fillna(1.0), -1.0, 4.0)
        else:
            df["Beta_IHSG"] = 1.0

        df["Sharpe_Ratio"] = np.clip(
            (df["Annualized_Return_1Y"] - self.risk_free_rate) / (df["Annualized_Vol_252D"] + 1e-6), -5.0, 10.0
        ).fillna(0.0)

        # GARCH Volatility & Value at Risk (VaR 95%) for latest row
        latest_garch_vol = self.estimate_garch_volatility(df["Return_1D"])
        df["GARCH_Vol"] = latest_garch_vol
        df["VaR_95_1D"] = round((latest_garch_vol / np.sqrt(252)) * 1.645 * 100.0, 2)  # 1-day 95% VaR in %

        # 7. Forward Target
        df["Target_Return_5D"] = np.log(df["Adj Close"].shift(-5) / df["Adj Close"])
        df["Target_Class_5D"] = np.where(
            df["Target_Return_5D"].isna(),
            np.nan,
            (df["Target_Return_5D"] > 0).astype(float),
        )

        return df

    def engineer_features(self) -> pd.DataFrame:
        """
        Applies feature transformations across all tickers.
        """
        raw_df = self.load_raw_data()
        bench_df = self.load_benchmark_data()
        fund_df = self.load_fundamental_data()

        logger.info(f"Generating quant features across {raw_df['Ticker'].nunique()} emiten with GARCH...")
        processed_groups = []
        for _, group in raw_df.groupby("Ticker"):
            ticker_features = self.generate_ticker_features(group, bench_df)
            processed_groups.append(ticker_features)

        feature_df = pd.concat(processed_groups, ignore_index=True)

        if not fund_df.empty:
            feature_df = feature_df.merge(fund_df, on="Ticker", how="left")
            feature_df["Market_Cap"] = feature_df["Market_Cap"].astype(float).fillna(1e9)
            feature_df["Log_Market_Cap"] = np.log(feature_df["Market_Cap"] + 1e-9)

        clean_df = feature_df.dropna(subset=["SMA_200", "RSI_14"]).copy()
        clean_df.reset_index(drop=True, inplace=True)

        self.generate_advanced_metrics_snapshot(clean_df)
        return clean_df

    def generate_advanced_metrics_snapshot(self, df: pd.DataFrame) -> None:
        latest_records = []
        for ticker, group in df.groupby("Ticker"):
            latest_row = group.sort_values(by="Date").iloc[-1]
            latest_records.append(latest_row)

        if not latest_records:
            return

        latest_df = pd.DataFrame(latest_records)
        cols = [
            "Ticker", "Sector", "Date", "Close", "Annualized_Return_1Y", "Annualized_Vol_252D",
            "Beta_IHSG", "Sharpe_Ratio", "Max_Drawdown_1Y", "GARCH_Vol", "VaR_95_1D",
            "Pivot_Point", "Support_1", "Resistance_1", "PE_Ratio", "PB_Ratio", "ROE",
            "Dividend_Yield", "Debt_to_Equity", "Current_Ratio"
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
