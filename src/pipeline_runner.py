"""
Master Pipeline Runner - Two-Phase Quantitative Execution Engine
Menjalankan pipeline otomatis dalam 2 fase:
1. Opening Pulse (10:00 WIB / 03:00 UTC): Pembaruan harga Open, volume awal, dan alert trigger area beli.
2. EOD Settlement (17:15 WIB / 10:15 UTC): Full ingestion, multi-model inference, Foreign Flow, dan riset AI Gemini.
"""

import argparse
from datetime import datetime, timezone, timedelta
import importlib
import logging
from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import LOG_FORMAT

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("PipelineRunner")


def get_current_wib_datetime() -> datetime:
    """Mengembalikan objek datetime zona waktu Indonesia Barat (WIB / UTC+7)."""
    return datetime.now(timezone(timedelta(hours=7)))


def run_opening_pulse_phase():
    """Menjalankan Fase 1: Opening Pulse (10:00 WIB)."""
    logger.info(">>> MEMULAI FASE 1: OPENING PULSE (HARGA OPEN & INTRADAY TRIGGER) <<<")
    opening_mod = importlib.import_module("src.opening_pulse")
    return opening_mod.run_opening_pulse()


def run_eod_settlement_phase():
    """Menjalankan Fase 2: Full End-of-Day Settlement (17:15 WIB)."""
    start_time = time.time()
    logger.info(">>> MEMULAI FASE 2: FULL EOD SETTLEMENT & MULTI-MODEL INFERENCE <<<")

    # Step 0: Universe Manager
    logger.info("[Step 0/5] Dynamic Universe Capacity & Health Check...")
    univ_mod = importlib.import_module("src.universe_manager")
    mgr = univ_mod.get_universe_manager()
    diag = mgr.get_universe_diagnostics()
    logger.info(f"Active Universe: {diag['active_tickers_count']} emiten terverifikasi.")

    # Step 1: Ingestion
    logger.info("[Step 1/5] Ingesting Official Daily Market Data & Macro...")
    ingest_mod = importlib.import_module("src.01_data_ingestion")
    ingest_mod.run_ingestion_pipeline()

    # Step 2: Feature Engineering
    logger.info("[Step 2/5] Engineering Technical Indicators & GARCH Volatility...")
    fe_mod = importlib.import_module("src.02_feature_eng")
    fe_mod.run_feature_engineering_pipeline()

    # Step 3: Model Inference & Portfolio Allocation
    logger.info("[Step 3/5] Running Multi-Engine AI Consensus (GBDT + LSTM + ARIMA)...")
    infer_mod = importlib.import_module("src.03_model_inference")
    infer_mod.run_model_inference_pipeline()

    # Step 3.5: Institutional Foreign Flow
    logger.info("[Step 3.5/5] Tracking Institutional Foreign Flow (Arus Modal Asing BEI)...")
    try:
        ff_mod = importlib.import_module("src.foreign_flow")
        ff_mod.compute_foreign_flow()
    except Exception as e:
        logger.warning(f"Foreign Flow engine issue (non-fatal): {e}")

    # Step 4: Morning Market Brief via Gemini AI
    logger.info("[Step 4/5] Generating Institutional Morning Market Brief...")
    try:
        mb_mod = importlib.import_module("src.morning_brief")
        mb_mod.run_morning_brief_pipeline()
    except Exception as e:
        logger.warning(f"Morning Brief generation issue (non-fatal): {e}")

    # Step 5: Multi-Year Financial Statements & Stock Archetypes
    logger.info("[Step 5/5] Building Multi-Year Financial Summary & 65-Day Indicators...")
    try:
        fin_mod = importlib.import_module("scripts.generate_financials_summary")
        fin_mod.build_summary()
    except Exception as e:
        logger.warning(f"Financial summary generation issue (non-fatal): {e}")

    elapsed = round(time.time() - start_time, 2)
    logger.info(f"FASE 2 EOD SETTLEMENT SELESAI DALAM {elapsed} DETIK!")
    return {"status": "success", "elapsed_seconds": elapsed}


# Hari Libur Resmi Bursa Efek Indonesia (BEI / IDX) Tahun 2026
IDX_HOLIDAYS_2026 = {
    "2026-01-01": "Tahun Baru Masehi 2026",
    "2026-01-16": "Isra Mi'raj Nabi Muhammad SAW",
    "2026-02-17": "Tahun Baru Imlek 2577 Kongzili",
    "2026-03-20": "Hari Suci Nyepi (Tahun Baru Saka 1948)",
    "2026-03-21": "Hari Raya Idul Fitri 1447 H",
    "2026-03-22": "Hari Raya Idul Fitri 1447 H",
    "2026-03-23": "Cuti Bersama Idul Fitri 1447 H",
    "2026-03-24": "Cuti Bersama Idul Fitri 1447 H",
    "2026-04-03": "Wafat Yesus Kristus (Jumat Agung)",
    "2026-05-01": "Hari Buruh Internasional",
    "2026-05-14": "Kenaikan Yesus Kristus",
    "2026-05-27": "Hari Raya Idul Adha 1447 H",
    "2026-05-31": "Hari Raya Waisak 2570 BE",
    "2026-06-01": "Hari Lahir Pancasila",
    "2026-06-16": "Tahun Baru Islam 1448 H",
    "2026-08-17": "Hari Kemerdekaan RI Ke-81",
    "2026-08-25": "Maulid Nabi Muhammad SAW",
    "2026-12-25": "Hari Raya Natal",
}


def is_idx_market_day(wib_dt: datetime) -> tuple[bool, str]:
    """
    Memeriksa apakah tanggal saat ini merupakan hari aktif bursa BEI.
    Mengembalikan (is_active, alasan).
    """
    # 1. Cek Akhir Pekan (Sabtu = 5, Minggu = 6)
    if wib_dt.weekday() == 5:
        return False, "Hari Sabtu (Bursa Tutup - Akhir Pekan)"
    if wib_dt.weekday() == 6:
        return False, "Hari Minggu (Bursa Tutup - Akhir Pekan)"

    # 2. Cek Tanggal Merah / Libur Bursa Resmi BEI
    date_str = wib_dt.strftime("%Y-%m-%d")
    if date_str in IDX_HOLIDAYS_2026:
        holiday_name = IDX_HOLIDAYS_2026[date_str]
        return False, f"Hari Libur Bursa Resmi BEI: {holiday_name}"

    return True, "Hari Kerja Bursa Aktif (Senin - Jumat)"


def main():
    parser = argparse.ArgumentParser(description="AlphaTech Two-Phase Quantitative Execution Runner")
    parser.add_argument(
        "--mode",
        choices=["auto", "opening_pulse", "eod_settlement"],
        default="auto",
        help="Mode eksekusi: 'opening_pulse' (10:00 WIB), 'eod_settlement' (17:15 WIB), atau 'auto' (deteksi waktu)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Paksa eksekusi pipeline meskipun hari libur bursa atau akhir pekan (bypass market day guard)",
    )
    args = parser.parse_args()

    wib_now = get_current_wib_datetime()
    current_hour_wib = wib_now.hour
    date_str = wib_now.strftime("%Y-%m-%d")
    logger.info(f"Pipeline Runner diaktifkan. Mode: {args.mode} | Waktu WIB: {wib_now.strftime('%Y-%m-%d %H:%M:%S')}")

    # Pengecekan Hari Aktif Bursa
    is_open, market_status_reason = is_idx_market_day(wib_now)
    if not is_open:
        if args.force:
            logger.warning(f"MARKET GUARD BYPASSED (--force aktif): {market_status_reason}. Melanjutkan eksekusi atas permintaan paksa.")
        else:
            logger.info(f"==================================================================")
            logger.info(f"  BURSA EFEK INDONESIA SEDANG TUTUP / LIBUR                      ")
            logger.info(f"  Status: {market_status_reason}                                 ")
            logger.info(f"  Tanggal: {date_str} (WIB)                                      ")
            logger.info(f"  Aksi: Pipeline otomatis dilewati dengan aman (skip).           ")
            logger.info(f"  Tip: Gunakan flag --force jika ingin mengeksekusi manual.      ")
            logger.info(f"==================================================================")
            return

    if args.mode == "opening_pulse":
        run_opening_pulse_phase()
    elif args.mode == "eod_settlement":
        run_eod_settlement_phase()
    else:
        # Auto Mode: Deteksi berdasarkan jam bursa WIB
        # Sesi pagi (09:00 - 13:59 WIB) -> Opening Pulse
        # Sesi sore / malam (14:00 - 08:59 WIB) -> EOD Settlement
        if 9 <= current_hour_wib < 14:
            logger.info(f"Jam {current_hour_wib}:00 WIB terdeteksi dalam rentang Sesi Pagi (09:00 - 13:59 WIB).")
            run_opening_pulse_phase()
        else:
            logger.info(f"Jam {current_hour_wib}:00 WIB terdeteksi dalam rentang Penutupan/EOD (>= 14:00 WIB).")
            run_eod_settlement_phase()


if __name__ == "__main__":
    main()
