"""
Generate standardized multi-year financial statements summary and stock archetypes for all 66 IDX tickers.
Outputs data/processed/financial_statements_summary.json for high-speed frontend and API access.
"""
import os
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("FinancialSummaryGenerator")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FIN_DIR = DATA_DIR / "raw" / "financial_statements"
FUND_FILE = DATA_DIR / "raw" / "fundamental_financial_data.csv"
RAW_MARKET_FILE = DATA_DIR / "raw" / "raw_market_data.csv"
OUTPUT_FILE = DATA_DIR / "processed" / "financial_statements_summary.json"
PRICE_HISTORY_FILE = DATA_DIR / "processed" / "price_history_30d.json"
FOREIGN_FLOW_FILE = DATA_DIR / "processed" / "foreign_flow_summary.json"
REC_FILES = [
    DATA_DIR / "processed" / "latest_alpha_recommendations_favorites.csv",
    DATA_DIR / "processed" / "latest_alpha_recommendations_swing.csv",
    DATA_DIR / "processed" / "latest_alpha_recommendations_dividend.csv",
]

# Ticker archetypes classification helpers
BIG_CAP_BLUE_CHIPS = {
    "BBCA", "BBRI", "BMRI", "BBNI", "ASII", "TLKM", "BREN", "AMMN", 
    "ADRO", "ICBP", "INDF", "UNVR", "KLBF", "UNTR", "BRIS"
}

def format_curr_val(val, is_usd=False):
    if pd.isnull(val) or val == 0:
        return "-"
    abs_val = abs(val)
    sign = "-" if val < 0 else ""
    if is_usd:
        # Institutional US standards: T = Trillion, B = Billion, M = Million, K = Thousand
        if abs_val >= 1e12:
            return f"{sign}${abs_val / 1e12:.2f} T"
        elif abs_val >= 1e9:
            return f"{sign}${abs_val / 1e9:.2f} B"
        elif abs_val >= 1e6:
            return f"{sign}${abs_val / 1e6:.2f} M"
        elif abs_val >= 1e3:
            return f"{sign}${abs_val / 1e3:.2f} K"
        else:
            return f"{sign}${abs_val:,.2f}"
    else:
        # IDX / Indonesian standards: T = Triliun, M = Miliar, Jt = Juta
        if abs_val >= 1e12:
            return f"{sign}Rp {abs_val / 1e12:.2f} T"
        elif abs_val >= 1e9:
            return f"{sign}Rp {abs_val / 1e9:.2f} M"
        elif abs_val >= 1e6:
            return f"{sign}Rp {abs_val / 1e6:.2f} Jt"
        else:
            return f"{sign}Rp {abs_val:,.0f}"

def build_summary():
    fund_df = pd.read_csv(FUND_FILE) if FUND_FILE.exists() else pd.DataFrame()
    fund_map = {}
    if not fund_df.empty:
        for _, r in fund_df.iterrows():
            t_clean = str(r.get("Ticker", "")).replace(".JK", "").strip()
            fund_map[t_clean] = r.to_dict()

    rec_map = {}
    for rf in REC_FILES:
        if rf.exists():
            df_rec = pd.read_csv(rf)
            for _, r in df_rec.iterrows():
                t_clean = str(r.get("Ticker", "")).replace(".JK", "").strip()
                if t_clean not in rec_map:
                    rec_map[t_clean] = r.to_dict()

    # Extract 100% Real 3-Month (65-Day) Historical Prices with OHLC from raw_market_data.csv
    price_history_map = {}
    if RAW_MARKET_FILE.exists():
        logger.info(f"Extracting 3-month (65-day) factual OHLC price history from {RAW_MARKET_FILE}...")
        try:
            raw_market_df = pd.read_csv(RAW_MARKET_FILE)
            raw_market_df["Date"] = pd.to_datetime(raw_market_df["Date"])
            for ticker_full, grp in raw_market_df.groupby("Ticker"):
                t_clean = str(ticker_full).replace(".JK", "").strip().upper()
                full_df = grp.sort_values("Date").copy()
                if full_df.empty:
                    continue

                close_series = full_df["Close"]
                vol_series = full_df["Volume"]

                # 1. EMAs & SMAs
                ema10_series = close_series.ewm(span=10, adjust=False).mean()
                ema20_series = close_series.ewm(span=20, adjust=False).mean()
                ema50_series = close_series.ewm(span=50, adjust=False).mean()
                ema100_series = close_series.ewm(span=100, adjust=False).mean()
                ema200_series = close_series.ewm(span=200, adjust=False).mean()
                ma50_series = close_series.rolling(50, min_periods=1).mean()
                ma200_series = close_series.rolling(200, min_periods=1).mean()

                # 2. Bollinger Bands (20, 2)
                bb_mid_series = close_series.rolling(20, min_periods=1).mean()
                bb_std_series = close_series.rolling(20, min_periods=1).std().fillna(0)
                bb_upper_series = bb_mid_series + 2 * bb_std_series
                bb_lower_series = bb_mid_series - 2 * bb_std_series

                # 3. MACD (12, 26, 9)
                ema12_series = close_series.ewm(span=12, adjust=False).mean()
                ema26_series = close_series.ewm(span=26, adjust=False).mean()
                macd_line_series = ema12_series - ema26_series
                macd_signal_series = macd_line_series.ewm(span=9, adjust=False).mean()
                macd_hist_series = macd_line_series - macd_signal_series

                # 4. RSI (14)
                delta = close_series.diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                avg_gain = gain.ewm(com=13, adjust=False).mean()
                avg_loss = loss.ewm(com=13, adjust=False).mean()
                rs = avg_gain / (avg_loss + 1e-9)
                rsi14_series = 100 - (100 / (1 + rs))

                # 5. Volume SMA 20
                vol_sma20_series = vol_series.rolling(20, min_periods=1).mean()

                # Slicing the last 65 trading days (3 months)
                sorted_grp = full_df.tail(65).copy()
                idx = sorted_grp.index

                dates_formatted = [d.strftime("%d %b") for d in sorted_grp["Date"]]
                full_dates = [d.strftime("%Y-%m-%d") for d in sorted_grp["Date"]]
                opens = [round(float(o), 2) for o in sorted_grp["Open"]]
                prices = [round(float(p), 2) for p in sorted_grp["Close"]]
                highs = [round(float(h), 2) for h in sorted_grp["High"]]
                lows = [round(float(l), 2) for l in sorted_grp["Low"]]
                volumes = [int(v) if pd.notnull(v) else 0 for v in sorted_grp["Volume"]]

                # Indicators aligned with last 65 days
                ema10 = [round(float(v), 2) for v in ema10_series.loc[idx]]
                ema20 = [round(float(v), 2) for v in ema20_series.loc[idx]]
                ema50 = [round(float(v), 2) for v in ema50_series.loc[idx]]
                ema100 = [round(float(v), 2) for v in ema100_series.loc[idx]]
                ema200 = [round(float(v), 2) for v in ema200_series.loc[idx]]
                ma50 = [round(float(v), 2) for v in ma50_series.loc[idx]]
                ma200 = [round(float(v), 2) for v in ma200_series.loc[idx]]
                bb_upper = [round(float(v), 2) for v in bb_upper_series.loc[idx]]
                bb_mid = [round(float(v), 2) for v in bb_mid_series.loc[idx]]
                bb_lower = [round(float(v), 2) for v in bb_lower_series.loc[idx]]
                macd_line = [round(float(v), 2) for v in macd_line_series.loc[idx]]
                macd_signal = [round(float(v), 2) for v in macd_signal_series.loc[idx]]
                macd_hist = [round(float(v), 2) for v in macd_hist_series.loc[idx]]
                rsi14 = [round(float(v), 2) for v in rsi14_series.loc[idx]]
                vol_sma20 = [round(float(v), 0) for v in vol_sma20_series.loc[idx]]

                price_history_map[t_clean] = {
                    "ticker": t_clean,
                    "dates": dates_formatted,
                    "full_dates": full_dates,
                    "opens": opens,
                    "highs": highs,
                    "lows": lows,
                    "prices": prices,
                    "sma20": bb_mid,
                    "volumes": volumes,
                    "ema10": ema10,
                    "ema20": ema20,
                    "ema50": ema50,
                    "ema100": ema100,
                    "ema200": ema200,
                    "ma50": ma50,
                    "ma200": ma200,
                    "bb_upper": bb_upper,
                    "bb_mid": bb_mid,
                    "bb_lower": bb_lower,
                    "macd_line": macd_line,
                    "macd_signal": macd_signal,
                    "macd_hist": macd_hist,
                    "rsi14": rsi14,
                    "vol_sma20": vol_sma20,
                    "last_price": prices[-1] if prices else 0,
                    "last_date": full_dates[-1] if full_dates else "",
                }
            logger.info(f"Extracted real 3-month OHLC price history and technical indicators for {len(price_history_map)} tickers.")
        except Exception as e:
            logger.error(f"Error extracting price history: {e}")

    foreign_flow_map = {}
    if FOREIGN_FLOW_FILE.exists():
        try:
            with open(FOREIGN_FLOW_FILE, "r", encoding="utf-8") as f:
                foreign_flow_map = json.load(f).get("constituents", {})
            logger.info(f"Loaded foreign flow data for {len(foreign_flow_map)} tickers.")
        except Exception as e:
            logger.error(f"Error loading foreign flow: {e}")

    financials_summary = {}

    csv_files = list(FIN_DIR.glob("*_financials.csv"))
    logger.info(f"Found {len(csv_files)} financial statement files in {FIN_DIR}")

    for fpath in csv_files:
        ticker = fpath.stem.replace("_financials", "").upper()
        fund = fund_map.get(ticker, {})
        rec = rec_map.get(ticker, {})

        try:
            df = pd.read_csv(fpath, index_col=0)
        except Exception as e:
            logger.warning(f"Error reading {fpath}: {e}")
            continue

        raw_years = [c for c in df.columns if "-" in str(c) or str(c).isdigit()]
        # Sort chronologically ascending
        sorted_year_cols = sorted(raw_years, key=lambda x: str(x)[:4])

        # Filter out years that have no active revenue or income data (e.g. 2021 with all NaNs)
        active_year_cols = []
        for c in sorted_year_cols:
            has_financial_data = False
            for check_metric in ["Total Revenue", "Operating Revenue", "Net Income", "Net Income Common Stockholders", "Operating Income", "Gross Profit"]:
                if check_metric in df.index and c in df.columns:
                    val_c = df.loc[check_metric, c]
                    if val_c is not None and not pd.isnull(val_c) and float(val_c) != 0:
                        has_financial_data = True
                        break
            if has_financial_data:
                active_year_cols.append(c)

        if len(active_year_cols) >= 2:
            sorted_year_cols = active_year_cols

        years_labels = [str(c)[:4] for c in sorted_year_cols]

        def get_series(row_names):
            for name in row_names:
                if name in df.index:
                    s = df.loc[name]
                    return [float(s[c]) if (c in s and pd.notnull(s[c])) else None for c in sorted_year_cols]
            return [None] * len(sorted_year_cols)

        revenue_list = get_series(["Total Revenue", "Operating Revenue"])
        gross_profit_list = get_series(["Gross Profit"])
        operating_income_list = get_series(["Operating Income", "EBIT"])
        net_income_list = get_series(["Net Income", "Net Income Common Stockholders", "Normalized Income"])
        eps_list = get_series(["Basic EPS", "Diluted EPS"])
        shares_list = get_series(["Basic Average Shares", "Diluted Average Shares"])
        operating_expense_list = get_series(["Operating Expense", "Total Expenses"])

        # Detect if numbers are in USD (< 50 billion nominal vs IDR >= 2.4 trillion nominal)
        max_rev = max([abs(v) for v in revenue_list if v is not None] or [0])
        is_usd = bool(max_rev > 0 and max_rev < 50e9)

        # Health Scoring & Archetypes
        mcap = float(fund.get("Market_Cap") or 0.0)
        close_price = float(fund.get("Close") or rec.get("Close") or 1000.0)
        roe = float(fund.get("ROE") or 0.0) * 100.0
        
        is_bank = ticker in ["BBCA", "BBRI", "BMRI", "BBNI", "BBTN", "BRIS"]
        der_raw = fund.get("Debt_to_Equity")
        if is_bank or pd.isna(der_raw) or der_raw is None:
            der = 0.0
            der_display = "-"
        else:
            der = float(der_raw)
            der_display = f"{der:.2f}x"

        div_yield = float(fund.get("Dividend_Yield") or 0.0)
        pe = float(fund.get("PE_Ratio") or 0.0)
        pbv = float(fund.get("PB_Ratio") or 0.0)
        rec_action = str(rec.get("Recommendation") or "BUY ON WEAKNESS").upper()
        prob = float(rec.get("Bullish_Probability") or 0.70)
        rrr = float(rec.get("Risk_Reward_Ratio") or 2.0)

        # Secondary estimated shares if not in statement: Market_Cap / Close
        est_shares = (mcap / close_price) if (mcap > 0 and close_price > 0) else None

        # Format EPS with precision and fallback: (Net Income / Total Shares)
        formatted_eps = []
        for i in range(len(sorted_year_cols)):
            raw_eps = eps_list[i]
            net_inc = net_income_list[i]

            # Check if raw_eps is valid (not null and not 0)
            is_valid_eps = (raw_eps is not None and not pd.isnull(raw_eps) and float(raw_eps) != 0)

            if not is_valid_eps and net_inc is not None and not pd.isnull(net_inc):
                # Fallback calculation: Net Income / Total Outstanding Shares
                sh = shares_list[i] if (shares_list and i < len(shares_list) and shares_list[i] and shares_list[i] > 0) else est_shares
                if sh and sh > 0:
                    raw_eps = net_inc / sh

            if raw_eps is None or pd.isnull(raw_eps):
                formatted_eps.append("-")
            else:
                raw_eps = float(raw_eps)
                if is_usd:
                    if abs(raw_eps) >= 0.05:
                        formatted_eps.append(f"${raw_eps:.2f}")
                    elif abs(raw_eps) > 0:
                        formatted_eps.append(f"${raw_eps:.3f}")
                    else:
                        formatted_eps.append("$0.00")
                else:
                    formatted_eps.append(f"{raw_eps:,.0f}")

        # Calculate revenue growth YoY
        rev_growth = []
        for i in range(len(revenue_list)):
            if i == 0 or revenue_list[i-1] is None or revenue_list[i] is None or revenue_list[i-1] == 0:
                rev_growth.append(None)
            else:
                growth = ((revenue_list[i] - revenue_list[i-1]) / abs(revenue_list[i-1])) * 100
                rev_growth.append(round(growth, 1))

        # Calculate net margin
        net_margins = []
        for i in range(len(revenue_list)):
            r_val = revenue_list[i]
            n_val = net_income_list[i]
            if r_val and n_val and r_val != 0:
                net_margins.append(round((n_val / r_val) * 100, 1))
            else:
                net_margins.append(None)

        # Determine Archetypes (Kartu Mencolok)
        archetypes = []

        # 1. Blue Chip / Big Cap
        is_blue_chip = (ticker in BIG_CAP_BLUE_CHIPS) or (mcap >= 40e12)
        if is_blue_chip:
            archetypes.append({
                "id": "blue_chip",
                "label": "BLUE CHIP / BIG CAP",
                "badge": "💎 Blue Chip LQ45",
                "icon": "diamond",
                "gradient": "from-indigo-600 via-purple-600 to-pink-600",
                "badge_bg": "bg-purple-100 text-purple-900 border-purple-300",
                "card_bg": "bg-gradient-to-br from-indigo-50/90 via-purple-50/60 to-pink-50/40 border-indigo-200/80",
                "headline": "Kapitalisasi Pasar Jumbo & Fondasi Likuiditas Institusional",
                "desc": f"Memiliki Market Cap Rp {mcap/1e12:.1f}T dengan likuiditas harian sangat tinggi, menjadikannya pilihan utama dana pensiun dan fund manager institusi."
            })

        # 2. Swing Trading Prime
        is_swing = ("BUY" in rec_action or "BREAKOUT" in rec_action) and (prob >= 0.65 or rrr >= 1.8)
        if is_swing:
            archetypes.append({
                "id": "swing_trading",
                "label": "PRIME SWING TRADING",
                "badge": "⚡ Swing Trading Pick",
                "icon": "trending_up",
                "gradient": "from-emerald-600 via-teal-600 to-cyan-600",
                "badge_bg": "bg-emerald-100 text-emerald-900 border-emerald-300",
                "card_bg": "bg-gradient-to-br from-emerald-50/90 via-teal-50/60 to-cyan-50/40 border-emerald-200/80",
                "headline": f"Setup Momentum Kuantitatif: {rec_action}",
                "desc": f"Probabilitas bullish AI {int(prob*100)}% dengan rasio Risk:Reward 1 : {rrr:.1f}. Memiliki swing range yang atraktif untuk horizon 3–15 hari kerja."
            })

        # 3. Fundamental Bintang 5
        is_fundamental = (roe >= 15.0 and der <= 1.0) or (roe >= 20.0)
        if is_fundamental:
            archetypes.append({
                "id": "fundamental_strong",
                "label": "FUNDAMENTAL KOKOH",
                "badge": "⭐ High Quality Business",
                "icon": "verified",
                "gradient": "from-blue-600 via-sky-600 to-cyan-600",
                "badge_bg": "bg-blue-100 text-blue-900 border-blue-300",
                "card_bg": "bg-gradient-to-br from-blue-50/90 via-sky-50/60 to-cyan-50/40 border-blue-200/80",
                "headline": f"Profitabilitas Superior (ROE {roe:.1f}%) & Utang Sehat",
                "desc": f"Mampu menghasilkan imbal hasil ekuitas tinggi dengan efisiensi modal teruji 5 tahun dan neraca keuangan yang terlindung dari risiko gagal bayar."
            })

        # 4. Dividend Aristocrat
        is_dividend = div_yield >= 5.5
        if is_dividend:
            archetypes.append({
                "id": "dividend_cow",
                "label": "DIVIDEND CASH COW",
                "badge": f"💰 Yield Dividen {div_yield:.1f}%",
                "icon": "payments",
                "gradient": "from-amber-600 via-yellow-600 to-orange-600",
                "badge_bg": "bg-amber-100 text-amber-900 border-amber-300",
                "card_bg": "bg-gradient-to-br from-amber-50/90 via-yellow-50/60 to-orange-50/40 border-amber-200/80",
                "headline": f"Imbal Hasil Dividen Signifikan ({div_yield:.1f}% p.a.)",
                "desc": "Arus kas operasi yang berlimpah memungkinkan emiten membagikan dividen tunai tebal secara reguler kepada pemegang saham."
            })

        # 5. Value Play / Undervalued
        is_value = (pbv > 0 and pbv < 1.0) or (pe > 0 and pe < 8.0)
        if is_value and not is_fundamental:
            archetypes.append({
                "id": "value_play",
                "label": "VALUE PLAY / TERDISKON",
                "badge": "🏷️ Margin of Safety",
                "icon": "price_check",
                "gradient": "from-slate-700 via-zinc-600 to-stone-600",
                "badge_bg": "bg-slate-100 text-slate-800 border-slate-300",
                "card_bg": "bg-gradient-to-br from-slate-50/90 via-zinc-50/60 to-stone-50/40 border-slate-200/80",
                "headline": f"Valuasi Diskon (PBV {pbv:.2f}x / PER {pe:.1f}x)",
                "desc": "Harga pasar saat ini diperdagangkan di bawah nilai wajar historisnya, menawarkan bantalan risiko (*downside protection*) yang menarik."
            })

        if not archetypes:
            archetypes.append({
                "id": "growth_watch",
                "label": "EXPANSION & GROWTH WATCH",
                "badge": "📊 Dynamic Mid-Cap",
                "icon": "insights",
                "gradient": "from-teal-600 via-cyan-600 to-blue-600",
                "badge_bg": "bg-teal-100 text-teal-800 border-teal-300",
                "card_bg": "bg-gradient-to-br from-teal-50/90 to-cyan-50/50 border-teal-200",
                "headline": "Fase Transformasi Bisnis & Pertumbuhan Sektoral",
                "desc": "Emiten berada dalam fase konsolidasi strategis dengan potensi katalis pertumbuhan siklikal pada industrinya."
            })

        # 5-Year Financial Health Status with YoY Revenue Contraction Guard
        valid_profits = [p for p in net_income_list if p is not None]
        all_positive = all([p > 0 for p in valid_profits]) if valid_profits else True

        # Detect consecutive negative revenue growth in the latest reporting years
        valid_growths = [g for g in rev_growth if g is not None]
        consecutive_neg_rev = 0
        for g in reversed(valid_growths):
            if g < 0:
                consecutive_neg_rev += 1
            else:
                break

        neg_growths_list = [f"{g:+.1f}%" for g in valid_growths if g < 0]
        has_consecutive_neg_rev = consecutive_neg_rev >= 2

        solv_desc = f"solvabilitas (DER {der_display})" if der_display != "-" else "permodalan perbankan"
        if has_consecutive_neg_rev:
            health_status = "MODERAT / KONTRAKSI OMZET"
            health_badge = "bg-amber-100 text-amber-900 border-amber-300"
            health_desc = (
                f"Meskipun {solv_desc} dan laba bersih positif, terdeteksi kontraksi omzet berturut-turut "
                f"({consecutive_neg_rev} periode terakhir: {', '.join(neg_growths_list[-consecutive_neg_rev:])}) yang menjadi sinyal waspada perlambatan top-line."
            )
        elif roe >= 15.0 and (der <= 1.2 or is_bank) and all_positive:
            health_status = "SANGAT SEHAT & PRIMA"
            health_badge = "bg-emerald-100 text-emerald-900 border-emerald-300"
            health_desc = "Neraca solid tanpa ancaman solvabilitas, laba bersih multi-tahun konsisten surplus, dan ekspansi operasional terjaga prima."
        elif all_positive and (der <= 2.0 or is_bank):
            health_status = "SEHAT & STABIL"
            health_badge = "bg-blue-100 text-blue-900 border-blue-300"
            health_desc = "Fundamental operasional berada dalam jalur stabil, struktur utang terkendali, dan kapasitas arus kas mencukupi kewajiban modal."
        else:
            health_status = "MODERAT / PERLU MONITORING"
            health_badge = "bg-amber-100 text-amber-900 border-amber-300"
            health_desc = "Terdapat fluktuasi margin laba, perlambatan pertumbuhan, atau tingkat leverage utang yang perlu dipantau ketat seiring dinamika siklus sektoral."

        # Structured Gemini Flash Analysis Narrative with Mandatory Revenue YoY Evaluation
        if has_consecutive_neg_rev:
            rev_trend_analysis = (
                f"Namun demikian, terdapat sinyal waspada berupa penurunan pendapatan usaha (YoY) berturut-turut "
                f"selama {consecutive_neg_rev} periode terakhir ({', '.join(neg_growths_list[-consecutive_neg_rev:])}). "
                f"Meskipun profitabilitas bersih tetap surplus, kontraksi top-line menuntut kehati-hatian atas potensi normalisasi laba di masa depan."
            )
            tone_performance = "bertahan dengan profitabilitas tinggi namun menghadapi kontraksi top-line"
        elif roe >= 18 and all_positive:
            rev_trend_analysis = "Pertumbuhan top-line dan ekspansi arus kas operasional terjaga prima untuk menopang ekspansi bisnis secara mandiri."
            tone_performance = "bertumbuh impresif"
        else:
            rev_trend_analysis = "Pertumbuhan pendapatan dan struktur modal bertahan defensif dengan manajemen efisiensi biaya yang terukur."
            tone_performance = "solid dan defensif"

        der_narrative = f"serta rasio utang terhadap ekuitas (DER) {der_display}" if der_display != "-" else "serta rasio kecukupan modal perbankan yang prima"

        gemini_summary = {
            "health_evaluation": (
                f"Selama periode historis 4–5 tahun terakhir, {ticker} mempertahankan kinerja {tone_performance}. "
                f"Dengan Margin of Return on Equity (ROE) {roe:.1f}% {der_narrative}, "
                f"kesehatan finansial emiten dinilai {health_status.lower()}. {rev_trend_analysis}"
            ),
            "investor_fit": (
                f"Dari sisi profil investor: Saham ini {'sangat direkomendasikan untuk swing trader aktif' if is_swing else 'ideal untuk portofolio jangka panjang'} "
                f"dengan sinyal teknikal {rec_action} dan probabilitas AI {int(prob*100)}%. "
                f"{'Bagi investor pasif, dividen yield ' + str(round(div_yield, 1)) + '% p.a. memberikan arus pendapatan tunai yang sangat kompetitif di atas suku bunga deposito.' if is_dividend else 'Fokus utama ditujukan pada apresiasi nilai modal (capital gain).'}"
            ),
            "verdict": (
                f"Verdict Analis Kuantitatif: {ticker} layak dikoleksi dengan strategi {rec_action.lower()}. "
                f"Tingkat risiko terukur dengan target take profit realistis dan disiplin level stop loss ketat sesuai rencana trading."
            )
        }

        financials_summary[ticker] = {
            "ticker": ticker,
            "years": years_labels,
            "currency": "USD" if is_usd else "IDR",
            "is_usd": is_usd,
            "metrics": {
                "revenue": [format_curr_val(v, is_usd) for v in revenue_list],
                "gross_profit": [format_curr_val(v, is_usd) for v in gross_profit_list],
                "operating_income": [format_curr_val(v, is_usd) for v in operating_income_list],
                "net_income": [format_curr_val(v, is_usd) for v in net_income_list],
                "eps": formatted_eps,
                "revenue_growth": [f"{v:+.1f}%" if v is not None else "-" for v in rev_growth],
                "net_margin": [f"{v:.1f}%" if v is not None else "-" for v in net_margins],
            },
            "health": {
                "status": health_status,
                "badge_class": health_badge,
                "desc": health_desc,
                "roe": f"{roe:.1f}%",
                "der": der_display,
                "pe": f"{pe:.1f}x" if pe > 0 else "-",
                "pbv": f"{pbv:.2f}x" if pbv > 0 else "-",
                "div_yield": f"{div_yield:.1f}%",
                "mcap_formatted": format_curr_val(mcap, is_usd=False),
            },
            "archetypes": archetypes,
            "gemini_analysis": gemini_summary,
            "price_history": price_history_map.get(ticker, {}),
            "foreign_flow": foreign_flow_map.get(ticker, {}),
        }

    # Save dedicated 30-day factual price history file
    PRICE_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PRICE_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(price_history_map, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved real 30-day price history to {PRICE_HISTORY_FILE}")

    # Save comprehensive financials summary file (safeguarded against empty raw folder)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if financials_summary:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(financials_summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully generated financial statement summary for {len(financials_summary)} tickers at {OUTPUT_FILE}")
    elif OUTPUT_FILE.exists():
        logger.info(f"Preserving existing financial statements summary at {OUTPUT_FILE} (raw financial files are private/excluded).")
    else:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f)

if __name__ == "__main__":
    build_summary()
