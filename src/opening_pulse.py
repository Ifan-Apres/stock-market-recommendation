"""
Opening Pulse Engine - 1 Jam Setelah Pembukaan Pasar BEI (10:00 WIB)
Mengambil data harga pembukaan (Open), pergerakan harga berjalan (Intraday Close jam 10:00),
volume awal bursa, dan memperbarui status eksekusi rencana trading (Area Beli, TP, SL).
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import yfinance as yf

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    DEFAULT_TICKERS,
    LOG_FORMAT,
    PRICE_HISTORY_FILE,
    FINANCIALS_SUMMARY_FILE,
    SWING_RECOMMENDATION_FILE,
    DIVIDEND_RECOMMENDATION_FILE,
    FAVORITES_RECOMMENDATION_FILE,
    RECOMMENDATION_FILE,
    RECOMMENDATIONS_JSON_FILE,
    SNAPSHOT_FILE,
    PROCESSED_DATA_DIR,
)

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("OpeningPulseEngine")

MORNING_BRIEF_FILE = PROCESSED_DATA_DIR / "latest_morning_brief.json"


def get_current_wib_time() -> datetime:
    """Mengembalikan waktu WIB (UTC+7) saat ini."""
    return datetime.now(timezone(timedelta(hours=7)))


def run_opening_pulse(tickers: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Eksekusi Pipeline Fase 1: Opening Pulse (10:00 WIB / 03:00 UTC).
    1. Unduh data intraday harga pembukaan & pergerakan 1 jam pertama BEI.
    2. Cek status trigger eksekusi (Buy Zone, TP, SL).
    3. Update file CSV & JSON rekomendasi.
    4. Update candlestick bar hari ini di Terminal Grafik Teknikal.
    """
    wib_now = get_current_wib_time()
    timestamp_str = wib_now.strftime("%Y-%m-%d %H:%M:%S WIB")
    logger.info("==================================================================")
    logger.info(f"  OPENING PULSE (10:00 WIB) - INTRADAY PRICE & EXECUTION TRIGGER  ")
    logger.info(f"  Waktu Eksekusi: {timestamp_str}")
    logger.info("==================================================================")

    target_tickers = tickers or DEFAULT_TICKERS
    logger.info(f"Mengunduh pergerakan pembukaan bursa untuk {len(target_tickers)} emiten...")

    # 1. Batch download 5-day historical to get today's open + previous close
    try:
        raw_df = yf.download(
            tickers=target_tickers,
            period="5d",
            interval="1d",
            progress=False,
            auto_adjust=False,
        )
    except Exception as e:
        logger.error(f"Gagal mengunduh data pasar via yfinance: {e}")
        return {"status": "error", "message": str(e)}

    if raw_df.empty:
        logger.warning("Data download kosong dari Yahoo Finance.")
        return {"status": "empty"}

    intraday_data: Dict[str, Dict[str, Any]] = {}
    latest_market_date = None

    # Handle multi-index vs single-ticker columns
    is_multi = isinstance(raw_df.columns, pd.MultiIndex)

    for ticker in target_tickers:
        try:
            if is_multi:
                if ticker not in raw_df["Close"].columns:
                    continue
                closes = raw_df["Close"][ticker].dropna()
                opens = raw_df["Open"][ticker].dropna()
                highs = raw_df["High"][ticker].dropna()
                lows = raw_df["Low"][ticker].dropna()
                vols = raw_df["Volume"][ticker].dropna()
            else:
                closes = raw_df["Close"].dropna()
                opens = raw_df["Open"].dropna()
                highs = raw_df["High"].dropna()
                lows = raw_df["Low"].dropna()
                vols = raw_df["Volume"].dropna()

            if len(closes) == 0:
                continue

            last_dt = closes.index[-1]
            last_date_str = last_dt.strftime("%Y-%m-%d") if hasattr(last_dt, "strftime") else str(last_dt)[:10]
            latest_market_date = last_date_str

            cur_close = float(closes.iloc[-1])
            cur_open = float(opens.iloc[-1]) if len(opens) else cur_close
            cur_high = float(highs.iloc[-1]) if len(highs) else cur_close
            cur_low = float(lows.iloc[-1]) if len(lows) else cur_close
            cur_vol = int(vols.iloc[-1]) if len(vols) else 0

            # Prev close for return calculation
            prev_close = float(closes.iloc[-2]) if len(closes) >= 2 else cur_close
            return_1d = (cur_close - prev_close) / prev_close if prev_close > 0 else 0.0

            clean_ticker = ticker.replace(".JK", "").upper()
            intraday_data[clean_ticker] = {
                "ticker_full": ticker,
                "ticker_clean": clean_ticker,
                "date": last_date_str,
                "open": cur_open,
                "high": cur_high,
                "low": cur_low,
                "close": cur_close,
                "volume": cur_vol,
                "prev_close": prev_close,
                "return_1d": return_1d,
            }
        except Exception as ex:
            logger.debug(f"Error parsing ticker {ticker}: {ex}")

    logger.info(f"Berhasil memproses {len(intraday_data)} emiten. Sesi Bursa: {latest_market_date}")

    # 2. Update CSV files (Swing, Dividend, Favorites, Main Recommendation)
    csv_targets = [
        (SWING_RECOMMENDATION_FILE, "Swing Recommendations"),
        (DIVIDEND_RECOMMENDATION_FILE, "Dividend Recommendations"),
        (FAVORITES_RECOMMENDATION_FILE, "Favorites Recommendations"),
        (RECOMMENDATION_FILE, "Main Recommendations"),
    ]

    all_updated_rows = []
    triggers_count = {"tp_hit": 0, "sl_hit": 0, "buy_zone": 0, "monitoring": 0}

    for file_path, label in csv_targets:
        if not file_path.exists():
            continue
        try:
            df = pd.read_csv(file_path)
            for idx, row in df.iterrows():
                t = str(row.get("Ticker", "")).replace(".JK", "").upper()
                if t in intraday_data:
                    data = intraday_data[t]
                    df.at[idx, "Date"] = data["date"]
                    df.at[idx, "Open"] = data["open"]
                    df.at[idx, "High"] = max(float(row.get("High", data["high"])), data["high"])
                    df.at[idx, "Low"] = min(float(row.get("Low", data["low"])), data["low"])
                    df.at[idx, "Close"] = data["close"]
                    df.at[idx, "Volume"] = data["volume"]
                    df.at[idx, "Return_1D"] = data["return_1d"]

                    # Check Intraday Status
                    tp = float(row.get("Target_Price", 0))
                    sl = float(row.get("Stop_Loss", 0))
                    entry = float(row.get("Entry_Price", 0))
                    curr = data["close"]

                    status = "DALAM PEMANTAUAN"
                    if tp > 0 and (data["high"] >= tp or curr >= tp):
                        status = "TARGET PROFIT TERCAPAI"
                        triggers_count["tp_hit"] += 1
                    elif sl > 0 and (data["low"] <= sl or curr <= sl):
                        status = "STOP LOSS ALERT"
                        triggers_count["sl_hit"] += 1
                    elif entry > 0 and (abs(curr - entry) / entry <= 0.015 or (data["low"] <= entry <= data["high"])):
                        status = "AREA BELI AKTIF"
                        triggers_count["buy_zone"] += 1
                    else:
                        triggers_count["monitoring"] += 1

                    df.at[idx, "Intraday_Status"] = status
                    df.at[idx, "Intraday_Updated_At"] = "10:00 WIB"

            df.to_csv(file_path, index=False)
            logger.info(f"Updated {label} ({file_path.name}) with opening pulse data.")

            if file_path == SWING_RECOMMENDATION_FILE or not all_updated_rows:
                all_updated_rows = df.to_dict(orient="records")
        except Exception as e:
            logger.error(f"Failed updating {file_path}: {e}")

    # 3. Export latest_recommendations.json
    if all_updated_rows:
        try:
            with open(RECOMMENDATIONS_JSON_FILE, "w", encoding="utf-8") as f:
                json.dump(all_updated_rows, f, indent=2, ensure_ascii=False)
            logger.info(f"Updated unified {RECOMMENDATIONS_JSON_FILE.name}")
        except Exception as e:
            logger.error(f"Error saving {RECOMMENDATIONS_JSON_FILE}: {e}")

    # 4. Update Candlestick Bar in Terminal Charts (price_history_30d.json & financials)
    if PRICE_HISTORY_FILE.exists():
        try:
            with open(PRICE_HISTORY_FILE, "r", encoding="utf-8") as f:
                history_store = json.load(f)

            for clean_t, data in intraday_data.items():
                if clean_t in history_store:
                    h = history_store[clean_t]
                    f_dates = h.get("full_dates", [])
                    d_dates = h.get("dates", [])
                    opens = h.get("opens", [])
                    highs = h.get("highs", [])
                    lows = h.get("lows", [])
                    prices = h.get("prices", [])
                    vols = h.get("volumes", [])

                    dt_obj = datetime.strptime(data["date"], "%Y-%m-%d")
                    short_date_str = dt_obj.strftime("%d %b")

                    if f_dates and f_dates[-1] == data["date"]:
                        # Update bar hari ini yang sudah ada
                        opens[-1] = data["open"]
                        highs[-1] = max(highs[-1], data["high"])
                        lows[-1] = min(lows[-1], data["low"])
                        prices[-1] = data["close"]
                        vols[-1] = data["volume"]
                    else:
                        # Bar hari baru (misal tanggal bursa baru dibuka hari ini)
                        f_dates.append(data["date"])
                        d_dates.append(short_date_str)
                        opens.append(data["open"])
                        highs.append(data["high"])
                        lows.append(data["low"])
                        prices.append(data["close"])
                        vols.append(data["volume"])

                        # Maintain rolling 65 days
                        if len(f_dates) > 65:
                            h["full_dates"] = f_dates[-65:]
                            h["dates"] = d_dates[-65:]
                            h["opens"] = opens[-65:]
                            h["highs"] = highs[-65:]
                            h["lows"] = lows[-65:]
                            h["prices"] = prices[-65:]
                            h["volumes"] = vols[-65:]

                    h["last_price"] = data["close"]
                    h["last_date"] = data["date"]

            with open(PRICE_HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(history_store, f, indent=2, ensure_ascii=False)
            logger.info(f"Integrated opening candlestick bars into {PRICE_HISTORY_FILE.name}")

            # Sinkronkan juga ke financial_statements_summary.json jika ada
            if FINANCIALS_SUMMARY_FILE.exists():
                try:
                    with open(FINANCIALS_SUMMARY_FILE, "r", encoding="utf-8") as f:
                        fin_data = json.load(f)
                    for clean_t, h in history_store.items():
                        if clean_t in fin_data:
                            fin_data[clean_t]["price_history"] = h
                    with open(FINANCIALS_SUMMARY_FILE, "w", encoding="utf-8") as f:
                        json.dump(fin_data, f, indent=2, ensure_ascii=False)
                    logger.info(f"Synchronized price history into {FINANCIALS_SUMMARY_FILE.name}")
                except Exception as ef:
                    logger.warning(f"Warning synchronizing financials summary: {ef}")

        except Exception as e:
            logger.error(f"Error updating price history file: {e}")

    # 5. Update snapshot metadata
    if SNAPSHOT_FILE.exists():
        try:
            with open(SNAPSHOT_FILE, "r", encoding="utf-8") as f:
                snap = json.load(f)
            snap["pipeline_phase"] = "OPENING_PULSE_10_WIB"
            snap["session_label"] = "Sesi Pagi (10:00 WIB) - Harga Pembukaan & Intraday Aktif"
            snap["last_updated"] = timestamp_str
            with open(SNAPSHOT_FILE, "w", encoding="utf-8") as f:
                json.dump(snap, f, indent=2, ensure_ascii=False)
            logger.info("Updated snapshot.json metadata.")
        except Exception as e:
            logger.debug(f"Error updating snapshot: {e}")

    logger.info("==================================================================")
    logger.info(f"OPENING PULSE SELESAI: {len(intraday_data)} Emiten Diperbarui.")
    logger.info(f"Trigger Summary -> Area Beli Aktif: {triggers_count['buy_zone']} | TP Hit: {triggers_count['tp_hit']} | SL Hit: {triggers_count['sl_hit']}")
    logger.info("==================================================================")

    return {
        "status": "success",
        "timestamp": timestamp_str,
        "market_date": latest_market_date,
        "tickers_updated": len(intraday_data),
        "triggers": triggers_count,
    }


if __name__ == "__main__":
    run_opening_pulse()
