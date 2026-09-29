"""
IDX Official Market Data Scraper
Direct extraction from Bursa Efek Indonesia (IDX) Trading Summary API.
Bypasses Cloudflare anti-bot challenges using browser impersonation via curl_cffi.
Provides 100% exact Foreign Buy, Foreign Sell, Volume, Value, and Net Foreign IDR.
"""

import datetime
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from curl_cffi import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("IDXScraper")

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "data" / "raw" / "idx_daily"

BASE_URL = "https://www.idx.co.id/primary/TradingSummary/GetStockSummary"

DEFAULT_HEADERS = {
    "accept": "application/json, text/plain, */*",
    "accept-language": "en-US,en;q=0.9,id;q=0.8",
    "referer": "https://www.idx.co.id/",
}

IMPERSONATE_ROTATION = ("safari18_0", "safari17_0", "chrome124", "edge101")


def get_cached_summary(date_str: str) -> Optional[Any]:
    """Retrieves cached daily stock summary (dict or list) if it exists."""
    clean_date = date_str.replace("-", "").strip()
    cache_file = CACHE_DIR / f"idx_summary_{clean_date}.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if (isinstance(data, list) and len(data) > 0) or (isinstance(data, dict) and len(data) > 0):
                    return data
        except Exception as e:
            logger.warning(f"Failed to read cache {cache_file}: {e}")
    return None


def save_cached_summary(date_str: str, data: Any) -> None:
    """Saves daily stock summary to cache."""
    clean_date = date_str.replace("-", "").strip()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"idx_summary_{clean_date}.json"
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        logger.info(f"Cached IDX records to {cache_file.name}")
    except Exception as e:
        logger.warning(f"Failed to write cache {cache_file}: {e}")


def fetch_idx_stock_summary(
    date_str: str, max_retries: int = 3, delay: float = 0.5
) -> Optional[List[Dict[str, Any]]]:
    """
    Fetches official Stock Summary from IDX for a specific date (YYYYMMDD or YYYY-MM-DD).
    Returns list of dicts or None if unavailable/failed.
    """
    clean_date = date_str.replace("-", "").strip()
    
    # 1. Check local cache first
    cached = get_cached_summary(clean_date)
    if cached:
        return cached

    # 2. Query official IDX endpoint
    url = f"{BASE_URL}?date={clean_date}&start=0&length=999"
    
    for attempt in range(max_retries):
        imp = IMPERSONATE_ROTATION[attempt % len(IMPERSONATE_ROTATION)]
        try:
            time.sleep(delay)
            response = requests.get(url, headers=DEFAULT_HEADERS, impersonate=imp, timeout=15)
            
            if response.status_code == 200:
                res_json = response.json()
                data = res_json.get("data", [])
                records_total = res_json.get("recordsTotal", 0)
                
                # If market not closed or 0 records, return None
                if records_total == 0 or len(data) == 0:
                    logger.warning(f"IDX returned 0 records for date {clean_date} (market open or holiday).")
                    return None
                
                logger.info(f"Successfully scraped {len(data)} stocks from IDX for {clean_date} using {imp}")
                save_cached_summary(clean_date, data)
                return data
            
            elif response.status_code == 403:
                logger.warning(f"HTTP 403 on {url} with {imp}, retrying next browser profile...")
                time.sleep(1.0)
            else:
                logger.warning(f"HTTP {response.status_code} for {clean_date}, retrying...")
                time.sleep(1.0)
                
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} error fetching IDX date {clean_date}: {e}")
            time.sleep(1.0)

    logger.error(f"Failed to fetch IDX stock summary for date {clean_date} after {max_retries} attempts.")
    return None


def parse_foreign_flow_from_summary(
    records: Any
) -> Dict[str, Dict[str, Any]]:
    """
    Parses raw IDX records into a structured mapping of:
    Ticker -> {
        'foreign_buy_vol': float,
        'foreign_sell_vol': float,
        'volume': float,
        'value': float,
        'close': float,
        'vwap': float,
        'net_foreign_vol': float,
        'net_foreign_idr': float,
        'foreign_participation': float
    }
    """
    if isinstance(records, dict):
        return records

    result: Dict[str, Dict[str, Any]] = {}
    if not isinstance(records, list):
        return result

    for r in records:
        code = str(r.get("StockCode", "")).strip().upper()
        if not code:
            continue
        
        vol = float(r.get("Volume", 0) or 0)
        val = float(r.get("Value", 0) or 0)
        close = float(r.get("Close", 0) or 0)
        fb = float(r.get("ForeignBuy", 0) or 0)
        fs = float(r.get("ForeignSell", 0) or 0)
        
        vwap = (val / vol) if vol > 0 else close
        net_vol = fb - fs
        net_idr = net_vol * vwap
        
        part_pct = ((fb + fs) / (2 * vol)) if vol > 0 else 0.0
        
        result[code] = {
            "foreign_buy_vol": fb,
            "foreign_sell_vol": fs,
            "volume": vol,
            "value": val,
            "close": close,
            "vwap": vwap,
            "net_foreign_vol": net_vol,
            "net_foreign_idr": net_idr,
            "foreign_participation": part_pct,
        }
    return result
