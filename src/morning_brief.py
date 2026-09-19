import json
import logging
import os
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# pyrefly: ignore [missing-import]
from dotenv import load_dotenv  # type: ignore # pyrefly: ignore [missing-import]
import google.generativeai as genai  # type: ignore # pyrefly: ignore [missing-import]
import pandas as pd  # type: ignore # pyrefly: ignore [missing-import]

# pyrefly: ignore [missing-import]
from src.config import (  # type: ignore # pyrefly: ignore [missing-import]
    ADVANCED_METRICS_FILE,
    BENCHMARK_DATA_FILE,
    GLOBAL_MACRO_FILE,
    LOG_FORMAT,
    MORNING_BRIEF_FILE,
    PROCESSED_DATA_FILE,
)

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("MorningBriefGenerator")


class MorningBriefGenerator:
    """
    Automated Institutional Market Morning Brief Generator.
    Aggregates Wall Street closes, US Treasury yields, global oil prices, Rupiah exchange rate,
    and IHSG technical levels into an institutional-grade daily brief using Gemini 1.5 Flash.
    """

    def __init__(self):
        self.api_key = GEMINI_API_KEY

    def collect_market_snapshot(self) -> Dict[str, Any]:
        """
        Gathers overnight global market closes and IHSG technical status.
        """
        snapshot = {
            "date": datetime.today().strftime("%Y-%m-%d"),
            "ihsg_close": 7150.0,
            "ihsg_change_pct": 0.35,
            "ihsg_ma20": 7120.0,
            "ihsg_support": "7.080 - 7.120",
            "ihsg_resistance": "7.200 - 7.240",
            "sp500_change": "+0.65%",
            "nasdaq_change": "+1.15%",
            "dow_change": "+0.40%",
            "ust_10y_yield": "4.25%",
            "brent_oil": "US$ 78.50 / barel",
            "usd_idr": "Rp 15.850 / US$",
        }

        # Read actual benchmark data if available
        if BENCHMARK_DATA_FILE.exists():
            try:
                b_df = pd.read_csv(BENCHMARK_DATA_FILE)
                if not b_df.empty:
                    last_row = b_df.iloc[-1]
                    prev_row = b_df.iloc[-2] if len(b_df) > 1 else last_row
                    close = round(float(last_row["Close"]), 2)
                    prev_close = float(prev_row["Close"])
                    chg_pct = round(((close - prev_close) / prev_close) * 100.0, 2)
                    snapshot["ihsg_close"] = close
                    snapshot["ihsg_change_pct"] = chg_pct
                    snapshot["date"] = str(last_row["Date"])[:10]
                    # Calculate simple support & resistance range
                    sup = round(close * 0.99, 0)
                    res = round(close * 1.012, 0)
                    snapshot["ihsg_support"] = f"{int(sup):,}"
                    snapshot["ihsg_resistance"] = f"{int(res):,}"
            except Exception as e:
                logger.warning(f"Error reading benchmark data: {str(e)}")

        # Read actual global macro data if available
        if GLOBAL_MACRO_FILE.exists():
            try:
                m_df = pd.read_csv(GLOBAL_MACRO_FILE)
                for asset, symbol in [("SP500", "SP500"), ("Nasdaq", "Nasdaq"), ("DowJones", "DowJones")]:
                    sub = m_df[m_df["Asset_Name"] == asset]
                    if len(sub) >= 2:
                        c1 = sub.iloc[-1]["Close"]
                        c0 = sub.iloc[-2]["Close"]
                        pct = round(((c1 - c0) / c0) * 100.0, 2)
                        snapshot[f"{asset.lower()}_change"] = f"{'+' if pct > 0 else ''}{pct}%"

                oil_sub = m_df[m_df["Asset_Name"] == "Oil_Brent"]
                if not oil_sub.empty:
                    snapshot["brent_oil"] = f"US$ {oil_sub.iloc[-1]['Close']:.2f} / barel"

                usd_sub = m_df[m_df["Asset_Name"] == "USD_IDR"]
                if not usd_sub.empty:
                    snapshot["usd_idr"] = f"Rp {usd_sub.iloc[-1]['Close']:,.0f} / US$"

                ust_sub = m_df[m_df["Asset_Name"] == "US_Treasury_10Y"]
                if not ust_sub.empty:
                    snapshot["ust_10y_yield"] = f"{ust_sub.iloc[-1]['Close']:.2f}%"
            except Exception as e:
                logger.warning(f"Error reading macro data: {str(e)}")

        return snapshot

    def generate_brief_text(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates editorial Morning Brief text using Gemini 3.6 Flash with clean institutional formatting.
        """
        title = f"Morning Brief IHSG: Katalis Wall Street & Arah Pasar Hari Ini ({snapshot['date']})"

        prompt = f"""
Bertindaklah sebagai Senior Institutional Equity Research Analyst di pasar modal Indonesia.
Susun laporan Morning Market Brief harian untuk para investor institusi dan profesional sebelum bursa BEI dibuka pagi ini.

Data Fakta Pasar Hari Ini:
- Tanggal: {snapshot['date']}
- IHSG Terakhir: {snapshot['ihsg_close']} (Perubahan harian: {snapshot['ihsg_change_pct']}%)
- Estimasi Range Hari Ini: Support {snapshot['ihsg_support']} | Resistance {snapshot['ihsg_resistance']}
- Wall Street: S&P 500 ({snapshot['sp500_change']}), Nasdaq ({snapshot['nasdaq_change']}), Dow Jones ({snapshot['dow_change']})
- Yield US Treasury 10 Tahun: {snapshot['ust_10y_yield']}
- Minyak Mentah Brent: {snapshot['brent_oil']}
- Kurs Rupiah: {snapshot['usd_idr']}

PANDUAN PENULISAN:
1. JANGAN mencantumkan header judul berulang seperti "INSTITUTIONAL EQUITY RESEARCH" atau "Tanggal: ...", karena sistem sudah memiliki header tersendiri.
2. JANGAN menggunakan tanda asterisk tebal ganda berlebihan (**) pada setiap kata. Tulis dalam paragraf naratif berita pasar yang mengalir alami dan profesional.
3. Struktur Analisis:
   - Paragraf 1: Ringkasan arah pembukaan IHSG hari ini dan proyeksi rentang pergerakan support-resisten.
   - Paragraf 2: Analisis sentimen global (penutupan Wall Street, yield obligasi AS, komoditas minyak, dan transmisi dampaknya ke bursa domestik).
   - Paragraf 3: Fundamental domestik, stabilitas kurs rupiah, dan ekspektasi arus dana asing (foreign inflow).
   - Paragraf 4: Posisi teknikal IHSG dan batas risiko pivot harian.
   - Paragraf 5: Panduan Taktis Hari Ini (misal: Selective Accumulation on Weakness atau Profit Taking bertahap).
"""

        if not self.api_key:
            logger.warning("No Gemini API key configured. Generating deterministic fallback brief.")
            brief_body = (
                f"Untuk perdagangan hari ini, IHSG diperkirakan bergerak fluktuatif dengan kecenderungan konsolidasi menguat "
                f"pada rentang {snapshot['ihsg_support']} hingga {snapshot['ihsg_resistance']}. Sentimen eksternal terdorong oleh "
                f"pergerakan Wall Street (S&P 500 {snapshot['sp500_change']}, Nasdaq {snapshot['nasdaq_change']}), sementara "
                f"yield US Treasury berada di kisaran {snapshot['ust_10y_yield']} dan minyak mentah Brent bertengger di {snapshot['brent_oil']}.\n\n"
                f"Dari domestik, kurs rupiah berada di kisaran {snapshot['usd_idr']}. Investor disarankan menerapkan strategi "
                f"Selective Buy on Weakness pada saham-saham likuid berfundamental solid dengan disiplin level stop loss."
            )
            return {
                "date": snapshot["date"],
                "headline": title,
                "brief_content": brief_body,
                "snapshot": snapshot,
            }

        try:
            model = genai.GenerativeModel("gemini-3.6-flash")
            response = model.generate_content(prompt)
            brief_body = response.text.strip()
            
            # Clean up any remaining repetitive headers
            lines = brief_body.split("\n")
            cleaned_lines = []
            for line in lines:
                l_strip = line.strip()
                if l_strip.startswith("**INSTITUTIONAL") or l_strip.startswith("INSTITUTIONAL") or l_strip.startswith("**Tanggal"):
                    continue
                cleaned_lines.append(line)
            brief_body = "\n".join(cleaned_lines).strip()

            return {
                "date": snapshot["date"],
                "headline": title,
                "brief_content": brief_body,
                "snapshot": snapshot,
            }
        except Exception as e:
            logger.error(f"Failed to generate brief via Gemini API: {str(e)}")
            return {
                "date": snapshot["date"],
                "headline": title,
                "brief_content": (
                    f"IHSG diperkirakan menguji area {snapshot['ihsg_support']} - {snapshot['ihsg_resistance']}. "
                    f"Sentimen pasar dipengaruhi penutupan Wall Street ({snapshot['sp500_change']}) dan kurs {snapshot['usd_idr']}. "
                    f"Panduan taktis: Buy on Weakness pada emiten berbobot pasar defensif."
                ),
                "snapshot": snapshot,
            }

    def execute_and_save(self) -> Dict[str, Any]:
        snapshot = self.collect_market_snapshot()
        brief_data = self.generate_brief_text(snapshot)

        MORNING_BRIEF_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(MORNING_BRIEF_FILE, "w", encoding="utf-8") as f:
            json.dump(brief_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Morning Brief successfully saved to {MORNING_BRIEF_FILE}")
        return brief_data


def run_morning_brief_pipeline() -> Dict[str, Any]:
    generator = MorningBriefGenerator()
    return generator.execute_and_save()


if __name__ == "__main__":
    run_morning_brief_pipeline()
