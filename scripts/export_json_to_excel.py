"""
AlphaTech JSON to Excel Exporter
Extracts all processed JSON market datasets, model recommendations, financial statements,
foreign flow tracking, technical levels, and universe telemetry into formatted Excel (.xlsx) workbooks.
Saves output directly to D:\\Portofolio\\data\\
"""

import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# Set directories
ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR = Path(r"D:\Portofolio\data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("JsonToExcelExporter")

HEADER_FILL = PatternFill(start_color="0F172A", end_color="0F172A", fill_type="solid")
HEADER_FONT = Font(name="Calibri", size=11, bold=True, color="F8FAFC")
BORDER_THIN = Border(
    left=Side(style="thin", color="E2E8F0"),
    right=Side(style="thin", color="E2E8F0"),
    top=Side(style="thin", color="E2E8F0"),
    bottom=Side(style="thin", color="E2E8F0")
)


def load_json(filepath: Path) -> Any:
    if not filepath.exists():
        logger.warning(f"File not found: {filepath}")
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return None


def style_worksheet(ws):
    """Applies professional styling, auto-column widths, and header borders."""
    ws.views.sheetView[0].showGridLines = True
    
    # Header row
    for col_idx in range(1, ws.max_column + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    
    # Auto-adjust column widths
    for col in ws.columns:
        col_letter = get_column_letter(col[0].column)
        max_len = 0
        for cell in col:
            val_str = str(cell.value or "")
            if "\n" in val_str:
                val_str = max(val_str.split("\n"), key=len)
            max_len = max(max_len, len(val_str))
        ws.column_dimensions[col_letter].width = max(max_len + 3, 11)
        
        # Data rows alignment and borders
        for cell in col[1:]:
            cell.border = BORDER_THIN
            if isinstance(cell.value, (int, float)):
                cell.alignment = Alignment(horizontal="right", vertical="center")
            else:
                cell.alignment = Alignment(vertical="center")


# ==============================================================================
# 1. EXPORT REKOMENDASI SAHAM & ALOKASI PORTOFOLIO (AlphaTech_Rekomendasi_Saham.xlsx)
# ==============================================================================
def export_recommendations_workbook():
    target_path = OUTPUT_DIR / "AlphaTech_Rekomendasi_Saham.xlsx"
    logger.info(f"Generating {target_path.name}...")
    
    snapshot = load_json(PROCESSED_DIR / "snapshot.json") or {}
    all_recs = load_json(PROCESSED_DIR / "latest_recommendations.json") or snapshot.get("all_recommendations", [])
    swing_recs = snapshot.get("recommendations_swing", [])
    div_recs = snapshot.get("recommendations_dividend", [])
    fav_recs = snapshot.get("recommendations_favorites", [])
    alloc_recs = snapshot.get("portfolio_allocation", [])
    metrics = snapshot.get("metrics", {})

    with pd.ExcelWriter(target_path, engine="openpyxl") as writer:
        # Sheet 1: Semua Rekomendasi (66 Emiten)
        if all_recs:
            df_all = pd.DataFrame(all_recs)
            # Reorder key presentation columns
            key_cols = [
                "Date", "Ticker", "Sector", "Close", "Recommendation", "Entry_Price", 
                "Target_Price", "Stop_Loss", "Risk_Reward_Ratio", "Bullish_Probability",
                "Tree_Blend_Prob", "CatBoost_Prob", "GBDT_Prob", "LSTM_Prob", "ARIMA_Prob",
                "Sharpe_Ratio", "GARCH_Vol", "VaR_95_1D", "Max_Drawdown_1Y", "Beta_IHSG",
                "RSI_14", "MACD_Hist", "CMF_20", "MFI_14", "PE_Ratio", "PB_Ratio", "ROE",
                "Dividend_Yield", "Net_Foreign_IDR", "Foreign_Accum_Divergence", "Is_Dividend_Season"
            ]
            ordered = [c for c in key_cols if c in df_all.columns] + [c for c in df_all.columns if c not in key_cols]
            df_all[ordered].to_excel(writer, sheet_name="Semua_Rekomendasi_66", index=False)
            style_worksheet(writer.sheets["Semua_Rekomendasi_66"])

        # Sheet 2: Swing Trader
        if swing_recs:
            df_swing = pd.DataFrame(swing_recs)
            df_swing.to_excel(writer, sheet_name="Swing_Trader_LQ45", index=False)
            style_worksheet(writer.sheets["Swing_Trader_LQ45"])

        # Sheet 3: Dividen & Value
        if div_recs:
            df_div = pd.DataFrame(div_recs)
            df_div.to_excel(writer, sheet_name="Dividen_dan_Value", index=False)
            style_worksheet(writer.sheets["Dividen_dan_Value"])

        # Sheet 4: 10 Pilihan Favorit
        if fav_recs:
            df_fav = pd.DataFrame(fav_recs)
            df_fav.to_excel(writer, sheet_name="Portofolio_10_Favorit", index=False)
            style_worksheet(writer.sheets["Portofolio_10_Favorit"])

        # Sheet 5: Alokasi Portofolio Modal
        if alloc_recs:
            df_alloc = pd.DataFrame(alloc_recs)
            df_alloc.to_excel(writer, sheet_name="Alokasi_Portofolio_Modal", index=False)
            style_worksheet(writer.sheets["Alokasi_Portofolio_Modal"])

        # Sheet 6: Metrik Evaluasi Model
        if metrics:
            metric_rows = [{"Metrik_Evaluasi": k, "Nilai_Pengujian_Out_of_Sample": v} for k, v in metrics.items()]
            df_metrics = pd.DataFrame(metric_rows)
            df_metrics.to_excel(writer, sheet_name="Metrik_Evaluasi_Model", index=False)
            style_worksheet(writer.sheets["Metrik_Evaluasi_Model"])

    logger.info(f"Successfully created: {target_path}")


# ==============================================================================
# 2. EXPORT ANALISIS TEKNIKAL & FUNDAMENTAL (AlphaTech_Watchlist_Analysis.xlsx)
# ==============================================================================
def export_watchlist_workbook():
    target_path = OUTPUT_DIR / "AlphaTech_Watchlist_Analysis.xlsx"
    logger.info(f"Generating {target_path.name}...")
    
    watchlist = load_json(PROCESSED_DIR / "watchlist_analysis.json") or {}
    financials = load_json(PROCESSED_DIR / "financial_statements_summary.json") or {}

    levels_rows = []
    fund_rows = []
    pillars_rows = []
    narrative_rows = []

    for ticker, item in watchlist.items():
        comp_name = item.get("company_name", ticker)
        tek = item.get("teknikal", {})
        levels = tek.get("levels", {})
        fund = item.get("fundamental", {})
        
        # 1. Level Eksekusi
        levels_rows.append({
            "Ticker": ticker,
            "Nama_Emiten": comp_name,
            "Buy_on_Weakness": levels.get("buy_on_weakness", "-"),
            "Buy_on_Breakout": levels.get("buy_on_breakout", "-"),
            "Take_Profit_1": levels.get("tp_1", "-"),
            "Take_Profit_2": levels.get("tp_2", "-"),
            "Target_Utama": levels.get("target_utama", "-"),
            "Cut_Loss": levels.get("cut_loss", "-"),
        })

        # 2. Naratif Teknikal
        narrative_rows.append({
            "Ticker": ticker,
            "Nama_Emiten": comp_name,
            "Naratif_Tren_dan_Price_Action": tek.get("narrative", "-"),
        })

        # 3. Empat Pilar Bisnis
        pillars_rows.append({
            "Ticker": ticker,
            "Nama_Emiten": comp_name,
            "Pilar_1_Laba_Bersih_EPS": fund.get("1_kinerja_laba_bersih", "-"),
            "Pilar_2_Pendapatan_dan_Operasional": fund.get("2_pendapatan_dan_laba_operasional", "-"),
            "Pilar_3_EBITDA_dan_Margin": fund.get("3_ebitda_dan_margin", "-"),
            "Pilar_4_Struktur_Keuangan_Neraca": fund.get("4_struktur_keuangan", "-"),
        })

        # 4. Kesehatan Finansial
        fin_item = financials.get(ticker, {})
        health = fin_item.get("health", {})
        fund_rows.append({
            "Ticker": ticker,
            "Nama_Emiten": comp_name,
            "Status_Kesehatan": health.get("status", "-"),
            "ROE": health.get("roe", "-"),
            "DER": health.get("der", "-"),
            "PER": health.get("pe", "-"),
            "PBV": health.get("pbv", "-"),
            "Dividend_Yield": health.get("div_yield", "-"),
            "Market_Cap": health.get("mcap_formatted", "-"),
            "Deskripsi_Kesehatan": health.get("desc", "-"),
        })

    with pd.ExcelWriter(target_path, engine="openpyxl") as writer:
        if levels_rows:
            pd.DataFrame(levels_rows).to_excel(writer, sheet_name="Matriks_Level_Eksekusi", index=False)
            style_worksheet(writer.sheets["Matriks_Level_Eksekusi"])
        if fund_rows:
            pd.DataFrame(fund_rows).to_excel(writer, sheet_name="Kesehatan_dan_Valuasi", index=False)
            style_worksheet(writer.sheets["Kesehatan_dan_Valuasi"])
        if pillars_rows:
            pd.DataFrame(pillars_rows).to_excel(writer, sheet_name="Analisis_4_Pilar_Bisnis", index=False)
            style_worksheet(writer.sheets["Analisis_4_Pilar_Bisnis"])
        if narrative_rows:
            pd.DataFrame(narrative_rows).to_excel(writer, sheet_name="Naratif_Analisis_Teknikal", index=False)
            style_worksheet(writer.sheets["Naratif_Analisis_Teknikal"])

    logger.info(f"Successfully created: {target_path}")


# ==============================================================================
# 3. EXPORT ARUS MODAL ASING (AlphaTech_Arus_Modal_Asing.xlsx)
# ==============================================================================
def export_foreign_flow_workbook():
    target_path = OUTPUT_DIR / "AlphaTech_Arus_Modal_Asing.xlsx"
    logger.info(f"Generating {target_path.name}...")
    
    data = load_json(PROCESSED_DIR / "foreign_flow_summary.json") or {}
    constituents = data.get("constituents", {})
    macro = data.get("macro", {})

    const_rows = []
    for ticker, val in constituents.items():
        const_rows.append({
            "Ticker": ticker,
            "Status_Aliran_Asing": val.get("status_id", val.get("status", "-")),
            "Partisipasi_Asing_Pct": val.get("foreign_participation_pct", "-"),
            "Net_Foreign_1D_IDR": val.get("net_foreign_1d", 0),
            "Net_Foreign_1D_Format": val.get("net_foreign_1d_formatted", "-"),
            "Net_Foreign_5D_IDR": val.get("net_foreign_5d", 0),
            "Net_Foreign_5D_Format": val.get("net_foreign_5d_formatted", "-"),
            "Net_Foreign_20D_IDR": val.get("net_foreign_20d", 0),
            "Net_Foreign_20D_Format": val.get("net_foreign_20d_formatted", "-"),
            "Net_Foreign_65D_IDR": val.get("net_foreign_65d", 0),
            "Net_Foreign_65D_Format": val.get("net_foreign_65d_formatted", "-"),
            "Tanggal_Terakhir": val.get("last_date", "-"),
        })

    with pd.ExcelWriter(target_path, engine="openpyxl") as writer:
        if const_rows:
            df_c = pd.DataFrame(const_rows)
            df_c.sort_values(by="Net_Foreign_65D_IDR", ascending=False, inplace=True)
            df_c.to_excel(writer, sheet_name="Ringkasan_Asing_Konstituen", index=False)
            style_worksheet(writer.sheets["Ringkasan_Asing_Konstituen"])

        if macro:
            df_macro = pd.DataFrame([{
                "Tanggal": macro.get("date", "-"),
                "Net_Foreign_1D_IHSG_IDR": macro.get("ihsg_net_foreign_1d", 0),
                "Net_Foreign_1D_IHSG_Format": macro.get("ihsg_net_foreign_1d_formatted", "-"),
            }])
            df_macro.to_excel(writer, sheet_name="Makro_Asing_IHSG", index=False)
            style_worksheet(writer.sheets["Makro_Asing_IHSG"])

    logger.info(f"Successfully created: {target_path}")


# ==============================================================================
# 4. EXPORT LAPORAN KEUANGAN MULTI-TAHUN (AlphaTech_Laporan_Keuangan_Multi_Tahun.xlsx)
# ==============================================================================
def export_financials_summary_workbook():
    target_path = OUTPUT_DIR / "AlphaTech_Laporan_Keuangan_Multi_Tahun.xlsx"
    logger.info(f"Generating {target_path.name}...")
    
    data = load_json(PROCESSED_DIR / "financial_statements_summary.json") or {}

    statements_rows = []
    ratio_rows = []
    ai_rows = []

    for ticker, item in data.items():
        curr = item.get("currency", "IDR")
        is_usd = item.get("is_usd", False)
        years = item.get("years", [])
        metrics = item.get("metrics", {})
        rev = metrics.get("revenue", [])
        gp = metrics.get("gross_profit", [])
        op = metrics.get("operating_income", [])
        ni = metrics.get("net_income", [])
        eps = metrics.get("eps", [])
        growth = metrics.get("revenue_growth", [])
        npm = metrics.get("net_margin", [])

        for idx, yr in enumerate(years):
            statements_rows.append({
                "Ticker": ticker,
                "Tahun_Buku": yr,
                "Mata_Uang": curr,
                "Basis_USD": "Ya" if is_usd else "Tidak",
                "Pendapatan_Revenue": rev[idx] if idx < len(rev) else "-",
                "Laba_Kotor_Gross_Profit": gp[idx] if idx < len(gp) else "-",
                "Laba_Operasional": op[idx] if idx < len(op) else "-",
                "Laba_Bersih_Net_Income": ni[idx] if idx < len(ni) else "-",
                "EPS": eps[idx] if idx < len(eps) else "-",
                "Pertumbuhan_Omzet_YoY": growth[idx] if idx < len(growth) else "-",
                "Net_Profit_Margin": npm[idx] if idx < len(npm) else "-",
            })

        archetypes_list = [
            x.get("title") or x.get("label") or str(x)
            if isinstance(x, dict) else str(x)
            for x in item.get("archetypes", [])
        ]
        health = item.get("health", {})
        ratio_rows.append({
            "Ticker": ticker,
            "Status_Kesehatan": health.get("status", "-"),
            "ROE": health.get("roe", "-"),
            "DER": health.get("der", "-"),
            "PER": health.get("pe", "-"),
            "PBV": health.get("pbv", "-"),
            "Dividend_Yield": health.get("div_yield", "-"),
            "Market_Cap": health.get("mcap_formatted", "-"),
            "Archetypes": ", ".join(archetypes_list),
        })

        ai = item.get("gemini_analysis", {})
        ai_rows.append({
            "Ticker": ticker,
            "Evaluasi_Kesehatan_Finansial": ai.get("health_evaluation", "-"),
            "Kesesuaian_Profil_Investor": ai.get("investor_fit", "-"),
            "Putusan_Akhir_Verdict": ai.get("verdict", "-"),
        })

    with pd.ExcelWriter(target_path, engine="openpyxl") as writer:
        if statements_rows:
            pd.DataFrame(statements_rows).to_excel(writer, sheet_name="Laporan_Keuangan_Tahunan", index=False)
            style_worksheet(writer.sheets["Laporan_Keuangan_Tahunan"])
        if ratio_rows:
            pd.DataFrame(ratio_rows).to_excel(writer, sheet_name="Rasio_dan_Valuasi", index=False)
            style_worksheet(writer.sheets["Rasio_dan_Valuasi"])
        if ai_rows:
            pd.DataFrame(ai_rows).to_excel(writer, sheet_name="Sintesis_Riset_AI", index=False)
            style_worksheet(writer.sheets["Sintesis_Riset_AI"])

    logger.info(f"Successfully created: {target_path}")


# ==============================================================================
# 5. EXPORT RIWAYAT HARGA 30 HARI (AlphaTech_Riwayat_Harga_30D.xlsx)
# ==============================================================================
def export_price_history_workbook():
    target_path = OUTPUT_DIR / "AlphaTech_Riwayat_Harga_30D.xlsx"
    logger.info(f"Generating {target_path.name}...")
    
    data = load_json(PROCESSED_DIR / "price_history_30d.json") or {}
    price_rows = []

    for ticker, item in data.items():
        dates = item.get("full_dates") or item.get("dates", [])
        opens = item.get("opens", [])
        highs = item.get("highs", [])
        lows = item.get("lows", [])
        prices = item.get("prices", [])
        vols = item.get("volumes", [])
        sma20 = item.get("sma20", [])
        rsi14 = item.get("rsi14", [])
        macd_h = item.get("macd_hist", [])
        bb_u = item.get("bb_upper", [])
        bb_l = item.get("bb_lower", [])

        for i, dt in enumerate(dates):
            price_rows.append({
                "Ticker": ticker,
                "Tanggal": dt,
                "Open": opens[i] if i < len(opens) else None,
                "High": highs[i] if i < len(highs) else None,
                "Low": lows[i] if i < len(lows) else None,
                "Close": prices[i] if i < len(prices) else None,
                "Volume": vols[i] if i < len(vols) else None,
                "SMA_20": sma20[i] if i < len(sma20) else None,
                "RSI_14": rsi14[i] if i < len(rsi14) else None,
                "MACD_Hist": macd_h[i] if i < len(macd_h) else None,
                "BB_Upper": bb_u[i] if i < len(bb_u) else None,
                "BB_Lower": bb_l[i] if i < len(bb_l) else None,
            })

    with pd.ExcelWriter(target_path, engine="openpyxl") as writer:
        if price_rows:
            df_p = pd.DataFrame(price_rows)
            df_p.to_excel(writer, sheet_name="Riwayat_Harga_OHLCV_30D", index=False)
            style_worksheet(writer.sheets["Riwayat_Harga_OHLCV_30D"])

    logger.info(f"Successfully created: {target_path}")


# ==============================================================================
# 6. EXPORT MORNING BRIEF & ACTIVE UNIVERSE
# ==============================================================================
def export_brief_and_universe_workbooks():
    # Morning Brief
    brief_target = OUTPUT_DIR / "AlphaTech_Morning_Market_Brief.xlsx"
    brief_data = load_json(PROCESSED_DIR / "latest_morning_brief.json")
    if brief_data:
        logger.info(f"Generating {brief_target.name}...")
        brief_rows = [{
            "Tanggal_Sesi": brief_data.get("session_date", "-"),
            "Status_Bursa": brief_data.get("market_status", "-"),
            "IHSG_Last_Close": brief_data.get("ihsg_summary", {}).get("close", "-"),
            "IHSG_Change_Pct": brief_data.get("ihsg_summary", {}).get("change_pct", "-"),
            "Judul_Editorial": brief_data.get("editorial_title", "-"),
            "Ringkasan_Eksekutif": brief_data.get("executive_summary", "-"),
            "Arus_Dana_Asing": brief_data.get("foreign_flow_narrative", "-"),
            "Sentimen_Global": brief_data.get("global_sentiment", "-"),
            "Narasi_Lengkap": brief_data.get("full_narrative", "-"),
        }]
        with pd.ExcelWriter(brief_target, engine="openpyxl") as writer:
            pd.DataFrame(brief_rows).to_excel(writer, sheet_name="Morning_Brief_IHSG", index=False)
            style_worksheet(writer.sheets["Morning_Brief_IHSG"])
        logger.info(f"Successfully created: {brief_target}")

    # Active Universe
    univ_target = OUTPUT_DIR / "AlphaTech_Semesta_Saham_Aktif_66.xlsx"
    univ_data = load_json(PROCESSED_DIR / "active_universe.json")
    if univ_data:
        logger.info(f"Generating {univ_target.name}...")
        sec_map = univ_data.get("sector_map", {})
        health_map = univ_data.get("health_status", {})
        
        univ_rows = []
        for sec, tickers in sec_map.items():
            for tick in tickers:
                clean = tick.replace(".JK", "")
                univ_rows.append({
                    "Ticker": clean,
                    "Ticker_Yahoo": tick,
                    "Sektor_BEI": sec,
                    "Status_Kesehatan": health_map.get(tick, "HEALTHY"),
                    "Kapasitas_Semesta": univ_data.get("capacity", 66),
                    "Tanggal_Rebalance": univ_data.get("last_rebalance_date", "-"),
                })
        with pd.ExcelWriter(univ_target, engine="openpyxl") as writer:
            pd.DataFrame(univ_rows).to_excel(writer, sheet_name="66_Saham_Aktif_BEI", index=False)
            style_worksheet(writer.sheets["66_Saham_Aktif_BEI"])
        logger.info(f"Successfully created: {univ_target}")


def run_all_exports():
    logger.info("==================================================================")
    logger.info("  STARTING MASTER JSON TO EXCEL EXPORT TO D:\\Portofolio\\data\\  ")
    logger.info("==================================================================")
    export_recommendations_workbook()
    export_watchlist_workbook()
    export_foreign_flow_workbook()
    export_financials_summary_workbook()
    export_price_history_workbook()
    export_brief_and_universe_workbooks()
    logger.info("==================================================================")
    logger.info("  ALL EXCEL WORKBOOKS GENERATED SUCCESSFULLY IN D:\\Portofolio\\data\\")
    logger.info("==================================================================")


if __name__ == "__main__":
    run_all_exports()
